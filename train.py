#!/usr/bin/env python3
"""
Fine-tune google/gemma-4-E2B for prompt injection detection using QLoRA
Optimized for RTX 4060 Ti (16GB VRAM)
"""
import os
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from load_data import load_prompt_injection_data, prepare_data_for_training

# Model configuration
MODEL_NAME = "google/gemma-4-E2B"
OUTPUT_DIR = "./models/gemma-4-prompt-injection"
MAX_LENGTH = 512

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def tokenize_function(examples, tokenizer):
    """Tokenize the text data"""
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH
    )

def main():
    print("="*60)
    print("Gemma-4-E2B Prompt Injection Detection Training")
    print("="*60)

    # Load data
    print("\n[1/6] Loading dataset...")
    train_df, test_df = load_prompt_injection_data()
    train_dataset, test_dataset = prepare_data_for_training(train_df, test_df)

    # Configure 4-bit quantization for memory efficiency
    print("\n[2/6] Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    print("\n[3/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize datasets
    print("\n[4/6] Tokenizing datasets...")
    def tokenize_and_keep_labels(examples):
        tokenized = tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH
        )
        # Keep labels
        tokenized['labels'] = examples['label']
        return tokenized

    train_dataset = train_dataset.map(
        tokenize_and_keep_labels,
        batched=True,
        remove_columns=['text']  # Remove only text, keep label
    )
    test_dataset = test_dataset.map(
        tokenize_and_keep_labels,
        batched=True,
        remove_columns=['text']  # Remove only text, keep label
    )

    # Load model with quantization
    print("\n[5/6] Loading model with 4-bit quantization...")
    print("This may take several minutes...")

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )

    # Prepare model for k-bit training
    base_model = prepare_model_for_kbit_training(base_model)

    # Configure LoRA
    print("\n[6/6] Configuring LoRA adapters...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # LoRA rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules="all-linear",  # Use all linear layers for Gemma4
        inference_mode=False,
    )

    model = get_peft_model(base_model, lora_config)

    # Add classification head
    print("Adding classification head...")
    model.config.pad_token_id = tokenizer.pad_token_id

    # Create a simple classification wrapper
    class SequenceClassificationModel(nn.Module):
        def __init__(self, base_model, num_labels=2):
            super().__init__()
            self.model = base_model
            self.num_labels = num_labels
            # Get text config for Gemma4 multimodal model
            text_config = base_model.config.get_text_config() if hasattr(base_model.config, 'get_text_config') else base_model.config
            hidden_size = text_config.hidden_size
            self.classifier = nn.Linear(hidden_size, num_labels)
            self.config = base_model.config

        def forward(self, input_ids, attention_mask=None, labels=None):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            # Get last hidden state and pool (use last token)
            hidden_states = outputs.hidden_states[-1]
            # Use mean pooling
            pooled = hidden_states.mean(dim=1)
            logits = self.classifier(pooled)

            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            # Return dict-like object that supports both dict access and attribute access
            from transformers.modeling_outputs import SequenceClassifierOutput
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
            )

        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
            """Enable gradient checkpointing on the base model"""
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

        def gradient_checkpointing_disable(self):
            """Disable gradient checkpointing on the base model"""
            if hasattr(self.model, 'gradient_checkpointing_disable'):
                self.model.gradient_checkpointing_disable()

    model = SequenceClassificationModel(model)
    model.to(base_model.device)

    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Training arguments
    # Note: bf16 disabled due to CUDA driver version issue - using fp32 instead
    # Note: save_strategy="no" to avoid safetensors shared memory error with LoRA
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,  # Reduced for fp32
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,  # Increased to maintain effective batch size
        eval_strategy="steps",  # Updated from evaluation_strategy
        eval_steps=50,
        save_strategy="no",  # Disabled due to shared tensor issue - save manually at end
        load_best_model_at_end=False,  # Disabled since we're not saving checkpoints
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=10,
        fp16=False,
        bf16=False,  # Disabled due to CUDA detection issue
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to=[],  # Disable all reporting integrations
        use_cpu=False,  # Still use GPU even though bf16 is off
    )

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")

    trainer.train()

    # Evaluate
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60 + "\n")

    results = trainer.evaluate()
    print("\nFinal Test Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")

    # Save the model
    print("\n" + "="*60)
    print("Saving model...")

    # Save LoRA adapters and classifier
    model.model.save_pretrained(OUTPUT_DIR)
    torch.save(model.classifier.state_dict(), f"{OUTPUT_DIR}/classifier.pt")
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save config for easy loading
    import json
    text_config = base_model.config.get_text_config() if hasattr(base_model.config, 'get_text_config') else base_model.config
    config = {
        "base_model": MODEL_NAME,
        "num_labels": 2,
        "hidden_size": text_config.hidden_size
    }
    with open(f"{OUTPUT_DIR}/training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nModel saved to: {OUTPUT_DIR}")
    print("="*60)
    print("Training complete!")
    print("="*60)

if __name__ == "__main__":
    main()

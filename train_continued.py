#!/usr/bin/env python3
"""
Continued QLoRA fine-tuning for prompt-injection detection.

Resumes from the existing LoRA adapter + classifier head at
./models/gemma-4-prompt-injection/ and trains further on a combined corpus
of public prompt-injection datasets. Saves to ./models/gemma-4-prompt-injection-v2/.
"""
import os
import json
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
)
from peft import PeftModel, prepare_model_for_kbit_training
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from load_data import load_combined_datasets

MODEL_NAME = "google/gemma-4-E2B"
RESUME_DIR = "./models/gemma-4-prompt-injection"
OUTPUT_DIR = "./models/gemma-4-prompt-injection-v2"
MAX_LENGTH = 512
LR = 1e-4          # lower than v1 (2e-4) to reduce catastrophic forgetting
EPOCHS = 2
WARMUP_STEPS = 200


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


class SequenceClassificationModel(nn.Module):
    """Mean-pooled hidden state + linear head, matching train.py's wrapper."""

    def __init__(self, base_model, num_labels=2, class_weights=None):
        super().__init__()
        self.model = base_model
        self.num_labels = num_labels
        text_config = base_model.config.get_text_config() if hasattr(base_model.config, 'get_text_config') else base_model.config
        hidden_size = text_config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.config = base_model.config
        self.class_weights = class_weights

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        pooled = hidden_states.mean(dim=1)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
            loss_fct = nn.CrossEntropyLoss(weight=weight)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(loss=loss, logits=logits)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()


def main():
    print("="*60)
    print("Continued fine-tuning (resume from %s)" % RESUME_DIR)
    print("="*60)

    if not os.path.isdir(RESUME_DIR):
        raise FileNotFoundError(
            f"Existing adapter not found at {RESUME_DIR}. "
            "Run train.py first or pull the v1 adapter from HF."
        )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("\n[1/6] Loading combined datasets...")
    train_df, test_df = load_combined_datasets()

    # Class weights for imbalance > 1.5x
    n_safe = int((train_df['label'] == 0).sum())
    n_inj = int((train_df['label'] == 1).sum())
    class_weights = None
    if n_safe > 0 and n_inj > 0:
        ratio = max(n_safe, n_inj) / max(1, min(n_safe, n_inj))
        if ratio > 1.5:
            total = n_safe + n_inj
            w0 = total / (2.0 * n_safe)
            w1 = total / (2.0 * n_inj)
            class_weights = torch.tensor([w0, w1], dtype=torch.float32)
            print(f"Imbalance {ratio:.2f}x detected — using class_weights={class_weights.tolist()}")

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    print("\n[2/6] Loading tokenizer (from resume dir)...")
    tokenizer = AutoTokenizer.from_pretrained(RESUME_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n[3/6] Tokenizing datasets...")

    def tokenize_fn(examples):
        tokenized = tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
        )
        tokenized['labels'] = examples['label']
        return tokenized

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=['text'], num_proc=2)
    test_ds = test_ds.map(tokenize_fn, batched=True, remove_columns=['text'], num_proc=2)

    # ------------------------------------------------------------------
    # Model: 4-bit base -> kbit prep -> attach existing PEFT adapter (trainable)
    # ------------------------------------------------------------------
    print("\n[4/6] Loading 4-bit base model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    base_model = prepare_model_for_kbit_training(base_model)

    print("\n[5/6] Attaching existing LoRA adapter (is_trainable=True)...")
    peft_model = PeftModel.from_pretrained(base_model, RESUME_DIR, is_trainable=True)
    peft_model.print_trainable_parameters()
    if not any(p.requires_grad for p in peft_model.parameters()):
        raise RuntimeError("No trainable parameters after PEFT attach — adapter loaded frozen.")

    peft_model.config.pad_token_id = tokenizer.pad_token_id

    # Wrap with classification head and load saved head weights
    model = SequenceClassificationModel(peft_model, num_labels=2, class_weights=class_weights)
    classifier_path = os.path.join(RESUME_DIR, "classifier.pt")
    classifier_state = torch.load(classifier_path, map_location=base_model.device)
    model.classifier.load_state_dict(classifier_state)
    model.to(base_model.device)
    print(f"Loaded classifier head from {classifier_path}")
    print(f"Classifier weight shape: {tuple(model.classifier.weight.shape)}")

    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ------------------------------------------------------------------
    # Training arguments — mirror train.py with adjusted lr/epochs/warmup
    # ------------------------------------------------------------------
    print("\n[6/6] Configuring training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="no",            # manual save at end (safetensors shared-tensor bug)
        load_best_model_at_end=False,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_steps=WARMUP_STEPS,
        logging_steps=50,
        fp16=False,
        bf16=False,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        report_to=[],
        use_cpu=False,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # ------------------------------------------------------------------
    # Pre-training eval (forgetting baseline on held-out)
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Pre-training eval on held-out (baseline for forgetting check):")
    print("="*60)
    pre_metrics = trainer.evaluate()
    for k, v in pre_metrics.items():
        try:
            print(f"  pre_{k}: {float(v):.4f}")
        except (TypeError, ValueError):
            pass

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Starting continued training...")
    print("="*60 + "\n")
    trainer.train()

    # ------------------------------------------------------------------
    # Final eval
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Post-training eval on held-out:")
    print("="*60)
    post_metrics = trainer.evaluate()
    for k, v in post_metrics.items():
        try:
            print(f"  post_{k}: {float(v):.4f}")
        except (TypeError, ValueError):
            pass

    pre_acc = pre_metrics.get('eval_accuracy')
    post_acc = post_metrics.get('eval_accuracy')
    if pre_acc is not None and post_acc is not None:
        delta = post_acc - pre_acc
        print(f"\nAccuracy delta (post - pre): {delta:+.4f}")
        if delta < -0.02:
            print("WARNING: held-out accuracy dropped >2pp — possible catastrophic forgetting.")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Saving model to %s ..." % OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.model.save_pretrained(OUTPUT_DIR)
    torch.save(model.classifier.state_dict(), f"{OUTPUT_DIR}/classifier.pt")
    tokenizer.save_pretrained(OUTPUT_DIR)

    text_config = base_model.config.get_text_config() if hasattr(base_model.config, 'get_text_config') else base_model.config
    config = {
        "base_model": MODEL_NAME,
        "num_labels": 2,
        "hidden_size": text_config.hidden_size,
        "resumed_from": RESUME_DIR,
    }
    with open(f"{OUTPUT_DIR}/training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nModel saved to: {OUTPUT_DIR}")
    print("="*60)
    print("Continued training complete!")
    print("="*60)


if __name__ == "__main__":
    main()

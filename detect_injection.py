#!/usr/bin/env python3
"""
Prompt Injection Detection using fine-tuned Gemma-4-E2B model
Simple CLI interface to test if a prompt is safe or contains injection attempts
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys
import os
import json

# Configuration
MODEL_PATH = "./models/gemma-4-prompt-injection"
BASE_MODEL = "google/gemma-4-E2B"

class SequenceClassificationModel(nn.Module):
    """Wrapper to add classification head to causal LM"""
    def __init__(self, base_model, num_labels=2):
        super().__init__()
        self.model = base_model
        self.num_labels = num_labels
        hidden_size = base_model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.config = base_model.config

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        pooled = hidden_states.mean(dim=1)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return type('Output', (), {'loss': loss, 'logits': logits})()

class PromptInjectionDetector:
    def __init__(self, model_path=MODEL_PATH):
        """Initialize the detector with the trained model"""
        print("Loading prompt injection detector...")

        # Check if we have a fine-tuned model
        if not os.path.exists(model_path):
            print(f"Warning: No fine-tuned model found at {model_path}")
            print("Please train the model first using train.py")
            sys.exit(1)

        print(f"Loading fine-tuned model from {model_path}")

        # Load config
        with open(f"{model_path}/training_config.json", "r") as f:
            config = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            config["base_model"],
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        # Load LoRA adapters
        base_model = PeftModel.from_pretrained(base_model, model_path)
        base_model.eval()

        # Create classification model and load classifier weights
        self.model = SequenceClassificationModel(base_model, num_labels=config["num_labels"])
        classifier_state = torch.load(f"{model_path}/classifier.pt", map_location=base_model.device)
        self.model.classifier.load_state_dict(classifier_state)
        self.model.to(base_model.device)
        self.model.eval()

        print("Model loaded successfully!\n")

    def detect(self, prompt, verbose=True):
        """
        Detect if a prompt contains injection attempts

        Args:
            prompt (str): The prompt to analyze
            verbose (bool): Print detailed output

        Returns:
            dict: {'is_safe': bool, 'confidence': float, 'label': str}
        """
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.model.device)

        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][prediction].item()

        # Interpret results
        is_safe = (prediction == 0)  # Assuming 0 is safe, 1 is injection
        label = "SAFE" if is_safe else "INJECTION DETECTED"

        result = {
            'is_safe': is_safe,
            'confidence': confidence,
            'label': label,
            'prediction': prediction
        }

        if verbose:
            self._print_result(prompt, result)

        return result

    def _print_result(self, prompt, result):
        """Print formatted results"""
        print("="*60)
        print("PROMPT INJECTION DETECTION RESULTS")
        print("="*60)
        print(f"\nPrompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
        print(f"\nStatus: {result['label']}")
        print(f"Confidence: {result['confidence']*100:.2f}%")

        if not result['is_safe']:
            print("\n⚠️  WARNING: This prompt may contain injection attempts!")
        else:
            print("\n✓ This prompt appears to be safe.")

        print("="*60 + "\n")

def main():
    """Main CLI interface"""
    detector = PromptInjectionDetector()

    print("="*60)
    print("Prompt Injection Detector - Interactive Mode")
    print("="*60)
    print("Enter prompts to check if they're safe or contain injections")
    print("Commands: 'quit' or 'exit' to stop")
    print("="*60 + "\n")

    while True:
        try:
            # Get user input
            print("Enter prompt (or 'quit' to exit):")
            user_prompt = input("> ").strip()

            if user_prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not user_prompt:
                print("Please enter a prompt.\n")
                continue

            # Detect
            detector.detect(user_prompt)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    # Check if prompt provided as command line argument
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        detector = PromptInjectionDetector()
        detector.detect(prompt)
    else:
        main()

#!/usr/bin/env python3
"""
Simple script to run Google Gemma-4-26B-A4B-it model
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_name = "google/gemma-4-26B-A4B-it"

    print(f"Loading model: {model_name}")
    print("This may take a while on first run as it downloads the model...")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    print("\nModel loaded successfully!")
    print("\n" + "="*50)
    print("Interactive Chat Mode")
    print("Type 'exit' or 'quit' to stop")
    print("="*50 + "\n")

    # Interactive chat loop
    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        if not user_input:
            continue

        # Format the input for the instruction-tuned model
        messages = [
            {"role": "user", "content": user_input}
        ]

        # Tokenize and generate
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(model.device)

        print("Assistant: ", end="", flush=True)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        # Decode and print response
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        print(response)
        print()

if __name__ == "__main__":
    main()

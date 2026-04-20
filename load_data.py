#!/usr/bin/env python3
"""
Load and preprocess the deepset/prompt-injections dataset
"""
import pandas as pd
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split

def load_prompt_injection_data():
    """
    Load the prompt injection dataset from Hugging Face
    Returns train and test datasets
    """
    print("Loading deepset/prompt-injections dataset...")

    # Load the dataset
    dataset = load_dataset("deepset/prompt-injections")

    # Convert to pandas for easier manipulation
    train_df = dataset['train'].to_pandas()
    test_df = dataset['test'].to_pandas()

    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    # Display sample data
    print("\nSample data:")
    print(train_df.head())
    print("\nColumn names:", train_df.columns.tolist())
    print("\nLabel distribution (train):")
    print(train_df['label'].value_counts())

    return train_df, test_df

def prepare_data_for_training(train_df, test_df, text_column='text', label_column='label'):
    """
    Prepare data for training by formatting it correctly
    """
    # Ensure we have the right columns
    train_data = train_df[[text_column, label_column]].copy()
    test_data = test_df[[text_column, label_column]].copy()

    # Rename columns to standard names
    train_data.columns = ['text', 'label']
    test_data.columns = ['text', 'label']

    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_data)
    test_dataset = Dataset.from_pandas(test_data)

    return train_dataset, test_dataset

if __name__ == "__main__":
    # Test the data loading
    train_df, test_df = load_prompt_injection_data()
    train_dataset, test_dataset = prepare_data_for_training(train_df, test_df)

    print("\n" + "="*50)
    print("Dataset prepared successfully!")
    print(f"Training examples: {len(train_dataset)}")
    print(f"Test examples: {len(test_dataset)}")
    print("="*50)

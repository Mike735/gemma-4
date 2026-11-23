#!/usr/bin/env python3
"""
Load and preprocess prompt-injection datasets.

Provides:
- load_prompt_injection_data / prepare_data_for_training: original deepset-only loader (used by train.py).
- load_combined_datasets: multi-source loader for continued fine-tuning (used by train_continued.py).
"""
import hashlib
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

def _normalize(df, text_col, label_col, source):
    """Standardize a dataset DataFrame to columns ['text', 'label', 'source']."""
    out = df[[text_col, label_col]].copy()
    out.columns = ['text', 'label']
    out['text'] = out['text'].astype(str)
    out['label'] = pd.to_numeric(out['label'], errors='coerce')
    out = out.dropna(subset=['text', 'label'])
    out = out[out['text'].str.len() > 0]
    out['label'] = out['label'].astype(int)
    out = out[out['label'].isin([0, 1])]
    out['source'] = source
    return out.reset_index(drop=True)


def _stratified_cap(df, n, seed=42):
    """Sample roughly n rows preserving class ratio. Pass through if df already smaller."""
    if len(df) <= n:
        return df.copy()
    parts = []
    for label_val, group in df.groupby('label'):
        share = max(1, int(round(n * len(group) / len(df))))
        share = min(share, len(group))
        parts.append(group.sample(n=share, random_state=seed))
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _safe_load(repo_id, split):
    """Load a HF dataset split into a pandas DataFrame, or return None on failure."""
    try:
        ds = load_dataset(repo_id, split=split)
        return ds.to_pandas()
    except Exception as e:
        print(f"  WARNING: failed to load {repo_id}[{split}]: {e}")
        return None


def load_combined_datasets(cap_large=20000, seed=42):
    """
    Load and merge multiple public prompt-injection datasets into a single
    binary classification corpus (label 0 = safe, 1 = injection).

    Returns:
        (train_df, test_df): pandas DataFrames with columns ['text', 'label'].

    Held-out test set is the concatenation of any official test splits
    (deepset + JasperLS); train rows whose text appears in test are dropped.
    """
    print("="*60)
    print("Loading combined prompt-injection datasets")
    print("="*60)

    train_parts = []
    test_parts = []

    # 1) deepset/prompt-injections — train + test (test held out)
    print("\n[1] deepset/prompt-injections")
    df = _safe_load("deepset/prompt-injections", "train")
    if df is not None:
        train_parts.append(_normalize(df, 'text', 'label', 'deepset'))
    df = _safe_load("deepset/prompt-injections", "test")
    if df is not None:
        test_parts.append(_normalize(df, 'text', 'label', 'deepset'))

    # 2) JasperLS/prompt-injections — train + test (test held out)
    print("\n[2] JasperLS/prompt-injections")
    df = _safe_load("JasperLS/prompt-injections", "train")
    if df is not None:
        train_parts.append(_normalize(df, 'text', 'label', 'jasperls'))
    df = _safe_load("JasperLS/prompt-injections", "test")
    if df is not None:
        test_parts.append(_normalize(df, 'text', 'label', 'jasperls'))

    # 3) imoxto/prompt_injection_cleaned_dataset-v2 — labels (plural), capped
    print("\n[3] imoxto/prompt_injection_cleaned_dataset-v2 (cap %d)" % cap_large)
    df = _safe_load("imoxto/prompt_injection_cleaned_dataset-v2", "train")
    if df is not None:
        norm = _normalize(df, 'text', 'labels', 'imoxto_v2')
        train_parts.append(_stratified_cap(norm, cap_large, seed=seed))

    # 4) imoxto/prompt_injection_hackaprompt_gpt35 — labels (plural), capped
    print("\n[4] imoxto/prompt_injection_hackaprompt_gpt35 (cap %d)" % cap_large)
    df = _safe_load("imoxto/prompt_injection_hackaprompt_gpt35", "train")
    if df is not None:
        norm = _normalize(df, 'text', 'labels', 'hackaprompt')
        train_parts.append(_stratified_cap(norm, cap_large, seed=seed))

    # 5) cgoosen/prompt_injection_password_or_secret — tiny, no cap
    print("\n[5] cgoosen/prompt_injection_password_or_secret")
    df = _safe_load("cgoosen/prompt_injection_password_or_secret", "train")
    if df is not None:
        train_parts.append(_normalize(df, 'text', 'label', 'cgoosen_password'))

    # 6) ivanleomk/prompt_injection_password — small, no cap
    print("\n[6] ivanleomk/prompt_injection_password")
    df = _safe_load("ivanleomk/prompt_injection_password", "train")
    if df is not None:
        train_parts.append(_normalize(df, 'text', 'label', 'ivanleomk_password'))

    if not train_parts:
        raise RuntimeError("No training datasets loaded — check network / HF access.")

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(
        columns=['text', 'label', 'source']
    )

    # De-dup train by text hash; remove any train rows whose text appears in held-out test.
    def _h(s):
        return hashlib.md5(s.encode('utf-8', errors='ignore')).hexdigest()

    train_df['_hash'] = train_df['text'].map(_h)
    train_df = train_df.drop_duplicates(subset='_hash').reset_index(drop=True)
    if len(test_df):
        test_hashes = set(test_df['text'].map(_h))
        before = len(train_df)
        train_df = train_df[~train_df['_hash'].isin(test_hashes)].reset_index(drop=True)
        leak = before - len(train_df)
        if leak:
            print(f"\nDropped {leak} train rows that overlapped with held-out test.")
    train_df = train_df.drop(columns=['_hash'])

    print("\n" + "="*60)
    print("Per-source training counts:")
    print(train_df['source'].value_counts().to_string())
    print("\nTrain class distribution:")
    print(train_df['label'].value_counts().to_string())
    if len(test_df):
        print("\nTest class distribution:")
        print(test_df['label'].value_counts().to_string())
    print("\nTotal train: %d   Total test: %d" % (len(train_df), len(test_df)))
    print("="*60)

    return train_df[['text', 'label']], test_df[['text', 'label']]


if __name__ == "__main__":
    # Test the data loading
    train_df, test_df = load_prompt_injection_data()
    train_dataset, test_dataset = prepare_data_for_training(train_df, test_df)

    print("\n" + "="*50)
    print("Dataset prepared successfully!")
    print(f"Training examples: {len(train_dataset)}")
    print(f"Test examples: {len(test_dataset)}")
    print("="*50)

    # Smoke-test the combined loader
    print("\n" + "="*50)
    print("Smoke-testing load_combined_datasets...")
    print("="*50)
    ctr, cte = load_combined_datasets()
    print(f"\nCombined train: {len(ctr)}  test: {len(cte)}")

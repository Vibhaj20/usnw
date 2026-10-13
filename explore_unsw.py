"""
UNSW-NB15 Initial Exploration
====================================================================
Run this BEFORE any preprocessing. Purpose: get real numbers on
class distribution, categorical values, and missing data -- and
specifically check for the known UNSW-NB15 label-consistency issue
(attack_cat spelling/casing sometimes differs between training-set
and testing-set files, e.g. "Backdoor" vs "Backdoors").

Author: Vibha J
====================================================================
"""

import pandas as pd

TRAIN_PATH = "./UNSW_NB15_training-set.csv"
TEST_PATH = "./UNSW_NB15_testing-set.csv"

print("="*70)
print("LOADING DATA")
print("="*70)
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
print(f"Training set: {train_df.shape}")
print(f"Testing set:  {test_df.shape}")

print("\n" + "="*70)
print("COLUMN DTYPES")
print("="*70)
print(train_df.dtypes)

print("\n" + "="*70)
print("MISSING VALUES (training set)")
print("="*70)
missing = train_df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")

print("\n" + "="*70)
print("ATTACK_CAT CLASS DISTRIBUTION -- TRAINING SET")
print("="*70)
print(train_df['attack_cat'].value_counts())

print("\n" + "="*70)
print("ATTACK_CAT CLASS DISTRIBUTION -- TESTING SET")
print("="*70)
print(test_df['attack_cat'].value_counts())

print("\n" + "="*70)
print("LABEL SPELLING CONSISTENCY CHECK (train vs test)")
print("="*70)
train_classes = set(train_df['attack_cat'].astype(str).str.strip().unique())
test_classes = set(test_df['attack_cat'].astype(str).str.strip().unique())
print(f"Classes in TRAIN only: {train_classes - test_classes}")
print(f"Classes in TEST only:  {test_classes - train_classes}")
print(f"Classes in BOTH:       {train_classes & test_classes}")

# Check for whitespace/casing issues specifically
print("\nRaw unique values in TRAIN attack_cat (checking for whitespace/casing):")
for val in sorted(train_df['attack_cat'].astype(str).unique()):
    print(f"  '{val}'  (len={len(val)})")

print("\nRaw unique values in TEST attack_cat (checking for whitespace/casing):")
for val in sorted(test_df['attack_cat'].astype(str).unique()):
    print(f"  '{val}'  (len={len(val)})")

print("\n" + "="*70)
print("BINARY LABEL DISTRIBUTION (label column: 0=normal, 1=attack)")
print("="*70)
print("Train:\n", train_df['label'].value_counts())
print("Test:\n", test_df['label'].value_counts())

print("\n" + "="*70)
print("CATEGORICAL COLUMN UNIQUE VALUE COUNTS")
print("="*70)
for col in ['proto', 'service', 'state']:
    print(f"\n{col}: {train_df[col].nunique()} unique values")
    print(train_df[col].value_counts().head(15))

print("\n" + "="*70)
print("DONE -- share this full output before preprocessing begins")
print("="*70)
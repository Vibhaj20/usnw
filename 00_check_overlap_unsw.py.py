"""
UNSW-NB15 Train/Test Overlap Check
====================================================================
Checks for the same leakage problem NSL-KDD had: exact duplicate rows
appearing in both the original training set and the test set. If
present, evaluating against the raw test set would inflate accuracy
the same way it did for NSL-KDD.

Checks the ORIGINAL preprocessed train data (before undersampling or
GAN augmentation) against the test set -- this is where genuine
dataset-level leakage would show up. Synthetic GAN rows added later
are not a leakage concern (a generated row matching a real test row
exactly is vanishingly unlikely).

Author: Vibha J
====================================================================
"""

import pandas as pd

TRAIN_PATH = "./unsw_preprocessed/X_train.csv"   # original, pre-undersampling, pre-GAN
TEST_PATH = "./unsw_preprocessed/X_test_clean.csv"

print("Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
print(f"Train: {train_df.shape}  Test: {test_df.shape}")

# Exact row-level duplicate check (all feature columns must match exactly)
print("\nChecking for exact duplicate rows between train and test...")
combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
dup_mask = combined.duplicated(keep=False)

n_total_dupes = dup_mask.sum()
n_test_dupes = dup_mask[len(train_df):].sum()

print(f"\nTotal rows involved in exact duplicates (train+test combined): {n_total_dupes}")
print(f"Test rows that are exact duplicates of a train row: {n_test_dupes}")
print(f"Percentage of test set affected: {n_test_dupes / len(test_df) * 100:.2f}%")

if n_test_dupes > 0:
    print("\nWARNING: Train/test overlap detected, similar to the NSL-KDD issue.")
    print("Recommend building a 'clean test' set (test rows with no exact match in")
    print("train) before reporting final accuracy numbers, same fix applied to NSL-KDD.")
else:
    print("\nNo exact duplicate rows found between train and test. Safe to evaluate")
    print("directly against the full test set.")
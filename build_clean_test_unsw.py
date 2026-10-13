"""
Build a "Clean" UNSW-NB15 Test Set
====================================================================
00_check_overlap_unsw.py found that 42.74% of the official UNSW-NB15
test set rows are EXACT duplicates of rows in the training set. Any
model evaluated against the raw test set gets credit for simply
memorizing those duplicated rows -- not for generalizing.

This script removes every test row that has an exact match in the
original (pre-undersampling, pre-GAN) training set, and writes out a
"clean" test set containing only genuinely unseen rows. This mirrors
the fix already applied to Keerthi's later NSL-KDD notebooks
(11_clean_test_evaluation, 13_clean_baseline_dt, etc.).

All 4 classifier scripts (02-05) should be re-run evaluating against
BOTH the full test set and this clean test set, so the report can show
the gap honestly rather than quoting only the inflated number.

Author: Vibha J
====================================================================
"""

import pandas as pd
import os

TRAIN_X_PATH = "./unsw_preprocessed/X_train.csv"          # original, pre-undersampling, pre-GAN
TEST_X_PATH = "./unsw_preprocessed/X_test.csv"
TEST_Y_MULTI_PATH = "./unsw_preprocessed/y_test_multiclass.csv"
TEST_Y_BINARY_PATH = "./unsw_preprocessed/y_test_binary.csv"

OUT_DIR = "./unsw_preprocessed"

print("=" * 60)
print("BUILDING CLEAN UNSW-NB15 TEST SET")
print("=" * 60)

print("\nLoading data...")
X_train = pd.read_csv(TRAIN_X_PATH)
X_test = pd.read_csv(TEST_X_PATH)
y_test_multi = pd.read_csv(TEST_Y_MULTI_PATH)
y_test_binary = pd.read_csv(TEST_Y_BINARY_PATH)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

assert len(X_test) == len(y_test_multi) == len(y_test_binary), (
    "X_test, y_test_multiclass, and y_test_binary must have the same "
    "number of rows and be in the same row order."
)

# Mark every test row that exactly matches a train row.
# Tag each frame before concatenating so duplicates are only flagged
# against train rows, not against other test rows.
train_tagged = X_train.copy()
train_tagged["_source"] = "train"

test_tagged = X_test.copy()
test_tagged["_source"] = "test"

combined = pd.concat([train_tagged, test_tagged], axis=0, ignore_index=True)

feature_cols = [c for c in combined.columns if c != "_source"]
dup_mask_any = combined.duplicated(subset=feature_cols, keep=False)

# Of the flagged rows, keep only the ones belonging to the test split
test_start = len(train_tagged)
test_is_dupe = dup_mask_any.values[test_start:]

n_dupes = test_is_dupe.sum()
print(f"\nTest rows that are exact duplicates of a train row: {n_dupes}")
print(f"Percentage of test set removed: {n_dupes / len(X_test) * 100:.2f}%")

keep_mask = ~test_is_dupe

X_test_clean = X_test.loc[keep_mask].reset_index(drop=True)
y_test_multi_clean = y_test_multi.loc[keep_mask].reset_index(drop=True)
y_test_binary_clean = y_test_binary.loc[keep_mask].reset_index(drop=True)

print(f"\nClean test set size: {len(X_test_clean)} (was {len(X_test)})")

print("\nClean test set class distribution (multiclass):")
print(y_test_multi_clean.iloc[:, 0].value_counts().sort_index())

# Save
X_test_clean.to_csv(os.path.join(OUT_DIR, "X_test_clean.csv"), index=False)
y_test_multi_clean.to_csv(os.path.join(OUT_DIR, "y_test_multiclass_clean.csv"), index=False)
y_test_binary_clean.to_csv(os.path.join(OUT_DIR, "y_test_binary_clean.csv"), index=False)

print("\nSaved:")
print(os.path.join(OUT_DIR, "X_test_clean.csv"))
print(os.path.join(OUT_DIR, "y_test_multiclass_clean.csv"))
print(os.path.join(OUT_DIR, "y_test_binary_clean.csv"))

print("\n" + "=" * 60)
print("DONE -- re-run the classifier scripts; they will now also")
print("evaluate against this clean test set automatically.")
print("=" * 60)

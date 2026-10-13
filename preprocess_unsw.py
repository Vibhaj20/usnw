"""
UNSW-NB15 Preprocessing Pipeline
====================================================================
Mirrors the NSL-KDD preprocessing convention established earlier in
this project, so the existing GAN scripts (vanilla/WCGAN/ACGAN) need
minimal changes to work on this dataset.

KEY DIFFERENCES FROM NSL-KDD HANDLED HERE:
1. `proto` has 133 unique values -- direct one-hot would create 133
   mostly-empty columns. Categories with fewer than MIN_CATEGORY_COUNT
   occurrences in the TRAINING set are grouped into a '<col>_other'
   bucket. The bucket set is determined from training data only, then
   applied identically to test data (no leakage).
2. `state` has 3 categories with exactly 1 occurrence each (PAR, URN,
   no) -- same bucketing logic absorbs these automatically.
3. `service` includes '-' as a legitimate category (no service
   identified) -- kept as-is, not treated as missing data.
4. 10 classes total (9 attacks + Normal), not 5 like NSL-KDD.

OUTPUTS (matching NSL-KDD file naming convention):
- gan_input_<ClassName>.csv  -- one per class, features only, no label
- X_train.csv, X_test.csv    -- full feature matrices
- y_train_multiclass.csv, y_test_multiclass.csv  -- encoded 0-9 labels
- y_train_binary.csv, y_test_binary.csv          -- 0=normal, 1=attack
- scaler.pkl, label_encoder.pkl

Author: Vibha J
====================================================================
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# ============================================================
# CONFIG
# ============================================================
TRAIN_PATH = "./UNSW_NB15_training-set.csv"
TEST_PATH = "./UNSW_NB15_testing-set.csv"
OUTPUT_DIR = "./unsw_preprocessed"

# TOP_K caps each categorical column's cardinality regardless of how the
# data is distributed (a count threshold doesn't bound the result -- if
# a column has a long "fairly common" tail, as UNSW-NB15's `proto` does,
# almost nothing gets filtered). Values chosen per column:
#   proto:   133 raw values, mostly obscure/rare protocol identifiers ->
#            cap hard at top 10 + 'other'
#   service: only 13 raw values already -> keep all of them (no real
#            dimensionality problem here)
#   state:   9 raw values, 4 of which have <15 occurrences -> top 5 + 'other'
TOP_K = {"proto": 10, "service": 13, "state": 5}
CATEGORICAL_COLS = ["proto", "service", "state"]
DROP_COLS = ["id"]  # row identifier, not predictive

os.makedirs(OUTPUT_DIR, exist_ok=True)


def bucket_rare_categories(train_series, test_series, top_k, other_label="other"):
    """Keep only the top_k most frequent categories (determined from TRAIN
    only), bucket everything else into 'other'. Applied identically to
    test -- guarantees a bounded column count regardless of how the raw
    category frequencies are distributed."""
    counts = train_series.value_counts()
    keep_categories = set(counts.head(top_k).index)
    dropped = set(counts.index) - keep_categories

    train_bucketed = train_series.apply(lambda x: x if x in keep_categories else other_label)
    test_bucketed = test_series.apply(lambda x: x if x in keep_categories else other_label)

    return train_bucketed, test_bucketed, keep_categories, dropped


def main():
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"Train: {train_df.shape}  Test: {test_df.shape}")

    train_df = train_df.drop(columns=DROP_COLS)
    test_df = test_df.drop(columns=DROP_COLS)

    # ---- Separate labels before any feature processing ----
    y_train_cat = train_df["attack_cat"].astype(str).str.strip()
    y_test_cat = test_df["attack_cat"].astype(str).str.strip()
    y_train_binary = train_df["label"].copy()
    y_test_binary = test_df["label"].copy()

    train_df = train_df.drop(columns=["attack_cat", "label"])
    test_df = test_df.drop(columns=["attack_cat", "label"])

    # ---- Bucket rare categories (fit on train, apply to both) ----
    print(f"\nBucketing categories (top-K per column, fit on train)...")
    for col in CATEGORICAL_COLS:
        train_bucketed, test_bucketed, kept, dropped = bucket_rare_categories(
            train_df[col], test_df[col], TOP_K[col]
        )
        train_df[col] = train_bucketed
        test_df[col] = test_bucketed
        print(f"  {col}: kept top {len(kept)} -> {sorted(kept)}")
        print(f"       bucketed {len(dropped)} others into 'other'"
              f"{' -> ' + str(sorted(dropped)) if len(dropped) <= 20 else ' (too many to list)'}")

    # ---- One-hot encode categorical columns (train defines column set) ----
    print("\nOne-hot encoding categorical columns...")
    train_onehot = pd.get_dummies(train_df[CATEGORICAL_COLS], prefix=CATEGORICAL_COLS)
    test_onehot = pd.get_dummies(test_df[CATEGORICAL_COLS], prefix=CATEGORICAL_COLS)
    # Align test columns to train's column set exactly (fills missing with 0, drops extras)
    test_onehot = test_onehot.reindex(columns=train_onehot.columns, fill_value=0)

    numeric_cols = [c for c in train_df.columns if c not in CATEGORICAL_COLS]
    train_numeric = train_df[numeric_cols].astype(float)
    test_numeric = test_df[numeric_cols].astype(float)

    # ---- Scale numeric features to [-1, 1] (matches NSL-KDD convention,
    # keeps existing GAN generator's Tanh output compatible) ----
    print("Scaling numeric features to [-1, 1]...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_numeric_scaled = pd.DataFrame(
        scaler.fit_transform(train_numeric), columns=numeric_cols, index=train_numeric.index
    )
    test_numeric_scaled = pd.DataFrame(
        scaler.transform(test_numeric), columns=numeric_cols, index=test_numeric.index
    )

    # ---- Combine: scaled numeric + one-hot categorical ----
    X_train = pd.concat([train_numeric_scaled.reset_index(drop=True),
                          train_onehot.reset_index(drop=True).astype(float)], axis=1)
    X_test = pd.concat([test_numeric_scaled.reset_index(drop=True),
                         test_onehot.reset_index(drop=True).astype(float)], axis=1)

    print(f"\nFinal feature count: {X_train.shape[1]} columns")
    print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")

    # ---- Encode multiclass labels (fit on train, apply to test) ----
    label_encoder = LabelEncoder()
    y_train_multiclass = pd.Series(label_encoder.fit_transform(y_train_cat), name="attack_category_encoded")
    y_test_multiclass = pd.Series(label_encoder.transform(y_test_cat), name="attack_category_encoded")

    print("\nLabel encoding map:")
    for i, cls in enumerate(label_encoder.classes_):
        print(f"  {i} = {cls}")

    # ---- Save main files ----
    X_train.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)
    y_train_multiclass.to_csv(os.path.join(OUTPUT_DIR, "y_train_multiclass.csv"), index=False)
    y_test_multiclass.to_csv(os.path.join(OUTPUT_DIR, "y_test_multiclass.csv"), index=False)
    y_train_binary.to_csv(os.path.join(OUTPUT_DIR, "y_train_binary.csv"), index=False)
    y_test_binary.to_csv(os.path.join(OUTPUT_DIR, "y_test_binary.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(OUTPUT_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)

    # ---- Save per-class gan_input files (features only, no label) ----
    print("\nSaving per-class gan_input files...")
    class_counts = {}
    for class_idx, class_name in enumerate(label_encoder.classes_):
        mask = (y_train_multiclass == class_idx).values
        class_df = X_train[mask].reset_index(drop=True)
        safe_name = class_name.replace(" ", "_")
        out_path = os.path.join(OUTPUT_DIR, f"gan_input_{safe_name}.csv")
        class_df.to_csv(out_path, index=False)
        class_counts[class_name] = len(class_df)
        print(f"  {class_name}: {len(class_df)} rows -> {out_path}")

    # ---- Summary ----
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"Total features: {X_train.shape[1]}")
    print(f"Train rows: {len(X_train)}  Test rows: {len(X_test)}")
    print("\nClass distribution (training set):")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls:20s} {count:>8,}")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
"""
KNN-Based Undersampling of Normal Class -- UNSW-NB15
====================================================================
Mirrors knn_undersample_normal.py from the NSL-KDD phase. Reduces
ONLY the Normal class via NearMiss-1 (memory-safe version -- NearMiss-2
caused a MemoryError on NSL-KDD at similar scale, learned that lesson
already). All 9 attack classes (Analysis, Backdoor, DoS, Exploits,
Fuzzers, Generic, Reconnaissance, Shellcode, Worms) are left completely
untouched here -- they get balanced via GAN oversampling separately.

Target: Normal 56,000 -> 40,000, matched to Generic (the largest
attack category), mirroring the NSL-KDD convention of undersampling
Normal to match the largest attack class.

Requires: imbalanced-learn (pip install imbalanced-learn)

Author: Vibha J
====================================================================
"""

import pandas as pd
import numpy as np
from imblearn.under_sampling import NearMiss

# ============================================================
# CONFIG -- EDIT THESE PATHS
# ============================================================
X_PATH = "./unsw_preprocessed/X_train.csv"
Y_PATH = "./unsw_preprocessed/y_train_multiclass.csv"
OUTPUT_X_PATH = "./unsw_preprocessed/X_train_undersampled.csv"
OUTPUT_Y_PATH = "./unsw_preprocessed/y_train_multiclass_undersampled.csv"

NORMAL_LABEL = 6          # from label encoding: 0=Analysis 1=Backdoor 2=DoS 3=Exploits
                           # 4=Fuzzers 5=Generic 6=Normal 7=Reconnaissance 8=Shellcode 9=Worms
TARGET_NORMAL_COUNT = 40000  # matched to Generic class size
NEARMISS_VERSION = 1      # memory-safe version (see NSL-KDD phase notes)
N_NEIGHBORS = 3


def main():
    print("Loading data...")
    X = pd.read_csv(X_PATH)
    y = pd.read_csv(Y_PATH)
    label_col = y.columns[0]
    y_series = y[label_col]

    print(f"\nOriginal class distribution:")
    print(y_series.value_counts().sort_index())
    print(f"Total: {len(y_series)}")

    normal_mask = y_series == NORMAL_LABEL
    attack_mask = ~normal_mask

    X_normal = X[normal_mask].reset_index(drop=True)
    y_normal = y_series[normal_mask].reset_index(drop=True)
    X_attacks = X[attack_mask].reset_index(drop=True)
    y_attacks = y_series[attack_mask].reset_index(drop=True)

    print(f"\nNormal class before undersampling: {len(X_normal)}")
    print(f"Attack classes (untouched): {len(X_attacks)}")

    if TARGET_NORMAL_COUNT >= len(X_normal):
        print(f"\nWARNING: TARGET_NORMAL_COUNT ({TARGET_NORMAL_COUNT}) >= current "
              f"Normal count ({len(X_normal)}). Nothing to undersample. Check your config.")
        return

    X_for_nearmiss = pd.concat([X_normal, X_attacks], axis=0, ignore_index=True)
    y_for_nearmiss = pd.Series(
        [NORMAL_LABEL] * len(X_normal) + [-1] * len(X_attacks)
    )

    sampling_strategy = {NORMAL_LABEL: TARGET_NORMAL_COUNT}

    print(f"\nRunning NearMiss-{NEARMISS_VERSION} undersampling on Normal class "
          f"(target: {TARGET_NORMAL_COUNT})...")
    nm = NearMiss(version=NEARMISS_VERSION, n_neighbors=N_NEIGHBORS,
                  sampling_strategy=sampling_strategy)
    X_resampled, y_resampled = nm.fit_resample(X_for_nearmiss, y_for_nearmiss)

    normal_result_mask = y_resampled == NORMAL_LABEL
    X_normal_undersampled = X_resampled[normal_result_mask].reset_index(drop=True)

    print(f"Normal class after undersampling: {len(X_normal_undersampled)}")

    X_final = pd.concat([X_normal_undersampled, X_attacks], axis=0, ignore_index=True)
    y_final = pd.concat([
        pd.Series([NORMAL_LABEL] * len(X_normal_undersampled)),
        y_attacks
    ], axis=0, ignore_index=True)

    shuffle_idx = np.random.RandomState(42).permutation(len(X_final))
    X_final = X_final.iloc[shuffle_idx].reset_index(drop=True)
    y_final = y_final.iloc[shuffle_idx].reset_index(drop=True)

    print(f"\nFinal class distribution after undersampling:")
    print(y_final.value_counts().sort_index())
    print(f"Total: {len(y_final)}")

    X_final.to_csv(OUTPUT_X_PATH, index=False)
    y_final.to_frame(name=label_col).to_csv(OUTPUT_Y_PATH, index=False)

    print(f"\nSaved -> {OUTPUT_X_PATH}")
    print(f"Saved -> {OUTPUT_Y_PATH}")
    print("\nSanity check: all 9 attack class counts should be UNCHANGED from original.")
    print("Compare the distribution above against the original printed at the top.")


if __name__ == "__main__":
    main()
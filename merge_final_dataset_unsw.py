"""
Final Merge -- UNSW-NB15: Undersampled Normal + Real Attacks + WCGAN Synthetic
====================================================================================
Assembles the final balanced training set for the classifier pipeline.
Unlike the NSL-KDD phase (WCGAN for most classes, ACGAN for U2R), this
dataset's comparison showed WCGAN winning on EVERY class, including
Worms (the most extreme minority class) -- so this script uses WCGAN
exclusively for all 9 attack classes. See the three-way comparison
discussion for why ACGAN underperformed here (9-way joint classification
task + 307x imbalance ratio outweighed its cross-class transfer benefit).

TARGET COUNT DESIGN (edit TARGET_COUNTS below if your guide wants a
different split -- this is a judgment call, same as U2R's cap in the
NSL-KDD phase):
  - Generic, Exploits, Fuzzers, DoS, Reconnaissance -> full target
    (40,000). Their real-to-target oversampling ratios are all under
    3x, safe territory.
  - Analysis, Backdoor, Shellcode -> moderate targets (not full
    40,000). Full balancing would mean 9-34x oversampling ratios;
    moderate targets keep this reasonable while still meaningfully
    boosting them.
  - Worms -> conservative target (5,000, NOT 40,000). Full balancing
    would be a ~307x oversampling ratio from only 130 real samples --
    far riskier than even U2R's 213x ratio in the NSL-KDD phase.

Author: Vibha J
====================================================================================
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ============================================================
# CONFIG
# ============================================================
UNDERSAMPLED_X_PATH = "./unsw_preprocessed/X_train_undersampled.csv"
UNDERSAMPLED_Y_PATH = "./unsw_preprocessed/y_train_multiclass_undersampled.csv"

WCGAN_DIR = "./unsw_wcgan_outputs"
OUTPUT_DIR = "./unsw_final_augmented"

LATENT_DIM = 32
SEED = 42

# Label encoding from preprocess_unsw.py's LabelEncoder (alphabetical):
# 0=Analysis 1=Backdoor 2=DoS 3=Exploits 4=Fuzzers 5=Generic
# 6=Normal 7=Reconnaissance 8=Shellcode 9=Worms
LABEL_MAP = {
    "Analysis": 0, "Backdoor": 1, "DoS": 2, "Exploits": 3, "Fuzzers": 4,
    "Generic": 5, "Reconnaissance": 7, "Shellcode": 8, "Worms": 9,
}
NORMAL_LABEL = 6

REAL_COUNTS = {
    "Analysis": 2000, "Backdoor": 1746, "DoS": 12264, "Exploits": 33393,
    "Fuzzers": 18184, "Generic": 40000, "Reconnaissance": 10491,
    "Shellcode": 1133, "Worms": 130,
}

# EDIT THIS if your guide wants different targets.
TARGET_COUNTS = {
    "Generic": 40000,         # already at target, 0 synthetic
    "Exploits": 40000,        # ratio ~0.2x
    "Fuzzers": 40000,         # ratio ~1.2x
    "DoS": 40000,             # ratio ~2.3x
    "Reconnaissance": 40000,  # ratio ~2.8x
    "Analysis": 20000,        # moderate cap, ratio ~9x (not full 40000 -> would be ~19x)
    "Backdoor": 20000,        # moderate cap, ratio ~10.5x (not full 40000 -> would be ~22x)
    "Shellcode": 15000,       # moderate cap, ratio ~12.2x (not full 40000 -> would be ~34x)
    "Worms": 5000,            # conservative cap, ratio ~37.5x (full 40000 would be ~307x)
}

torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# WCGAN GENERATOR (must match wcgan_baseline_unsw.py exactly to load weights)
# ============================================================
class WCGANGenerator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.BatchNorm1d(64), nn.ReLU(True),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Linear(128, output_dim), nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


# ============================================================
# ONE-HOT / BINARY COLUMN CLEANUP
# ============================================================
def clean_generated_columns(df):
    """Generator output is continuous even for columns that should be
    strictly 0/1 (one-hot dummies, binary flags). Snap them back."""
    df = df.copy()

    onehot_groups = {
        "proto": [c for c in df.columns if c.startswith("proto_")],
        "service": [c for c in df.columns if c.startswith("service_")],
        "state": [c for c in df.columns if c.startswith("state_")],
    }
    for group_name, cols in onehot_groups.items():
        if not cols:
            continue
        values = df[cols].values
        argmax_idx = np.argmax(values, axis=1)
        onehot = np.zeros_like(values)
        onehot[np.arange(len(values)), argmax_idx] = 1
        df[cols] = onehot

    binary_flags = ["is_ftp_login", "is_sm_ips_ports"]
    for col in binary_flags:
        if col in df.columns:
            df[col] = (df[col] > 0).astype(float)

    return df


def generate_synthetic(class_name, n_needed, feature_cols):
    if n_needed <= 0:
        return None

    model_path = os.path.join(WCGAN_DIR, f"generator_{class_name}.pt")
    output_dim = len(feature_cols)
    G = WCGANGenerator(LATENT_DIM, output_dim).to(device)
    G.load_state_dict(torch.load(model_path, map_location=device))
    G.eval()
    with torch.no_grad():
        z = torch.randn(n_needed, LATENT_DIM, device=device)
        synthetic = G(z).cpu().numpy()

    synth_df = pd.DataFrame(synthetic, columns=feature_cols)
    synth_df = clean_generated_columns(synth_df)
    print(f"  Generated {n_needed} synthetic {class_name} samples using WCGAN")
    return synth_df


# ============================================================
# MAIN
# ============================================================
def main():
    print("Loading undersampled Normal + real attack data...")
    X_base = pd.read_csv(UNDERSAMPLED_X_PATH)
    y_base = pd.read_csv(UNDERSAMPLED_Y_PATH)
    label_col = y_base.columns[0]
    feature_cols = X_base.columns.tolist()

    print(f"Base dataset: {len(X_base)} rows")
    print(f"Base class distribution:\n{y_base[label_col].value_counts().sort_index()}\n")

    all_X = [X_base]
    all_y = [y_base[label_col]]

    print("Generating synthetic samples per class (WCGAN only)...")
    for class_name, label_val in LABEL_MAP.items():
        target = TARGET_COUNTS[class_name]
        real_count = REAL_COUNTS[class_name]
        n_needed = max(0, target - real_count)
        ratio = n_needed / real_count if real_count > 0 else float("inf")

        print(f"\n{class_name}: real={real_count}  target={target}  "
              f"synthetic_needed={n_needed}  (oversampling ratio: {ratio:.1f}x)")
        if ratio > 15:
            print(f"  NOTE: high oversampling ratio for {class_name} -- document as a limitation.")

        synth_df = generate_synthetic(class_name, n_needed, feature_cols)
        if synth_df is not None:
            all_X.append(synth_df)
            all_y.append(pd.Series([label_val] * n_needed))

    X_final = pd.concat(all_X, axis=0, ignore_index=True)
    y_final = pd.concat(all_y, axis=0, ignore_index=True)

    shuffle_idx = np.random.RandomState(SEED).permutation(len(X_final))
    X_final = X_final.iloc[shuffle_idx].reset_index(drop=True)
    y_final = y_final.iloc[shuffle_idx].reset_index(drop=True)

    print(f"\n{'='*60}\nFINAL AUGMENTED DATASET (UNSW-NB15)\n{'='*60}")
    print(f"Total rows: {len(X_final)}")
    print("Final class distribution:")
    print(y_final.value_counts().sort_index())

    X_out_path = os.path.join(OUTPUT_DIR, "X_train_augmented.csv")
    y_out_path = os.path.join(OUTPUT_DIR, "y_train_multiclass_augmented.csv")
    X_final.to_csv(X_out_path, index=False)
    y_final.to_frame(name=label_col).to_csv(y_out_path, index=False)
    print(f"\nSaved -> {X_out_path}")
    print(f"Saved -> {y_out_path}")

    # ---- Summary table ----
    summary_rows = []
    for class_name, label_val in LABEL_MAP.items():
        real_count = REAL_COUNTS[class_name]
        target = TARGET_COUNTS[class_name]
        synthetic_added = max(0, target - real_count)
        summary_rows.append({
            "class": class_name, "real_samples": real_count,
            "synthetic_samples": synthetic_added, "architecture_used": "WCGAN",
            "final_total": real_count + synthetic_added,
        })
    normal_count = int((y_base[label_col] == NORMAL_LABEL).sum())
    summary_rows.append({
        "class": "Normal", "real_samples": normal_count, "synthetic_samples": 0,
        "architecture_used": "N/A (KNN undersampled)", "final_total": normal_count,
    })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "augmentation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nHandoff summary -> {summary_path}")
    print(summary_df.to_string(index=False))

    print("\nREMINDER: Do NOT touch X_test.csv / y_test_*.csv with any of this.")


if __name__ == "__main__":
    main()
import pandas as pd
import os


# ============================================================
# 1. MODEL NAMES AND CONFUSION MATRIX FILES
# ============================================================
# Two variants per model: "full" uses the raw official test set,
# "clean" uses the leakage-free test set built by
# build_clean_test_unsw.py (00_check_overlap_unsw.py found 42.74%
# train/test duplicate overlap in the official split -- the full-test
# numbers are inflated and should NOT be the ones quoted in the report).

MODEL_FILES_FULL = {
    "Decision Tree": "results/decision_tree_confusion_matrix_unsw.csv",
    "XGBoost": "results/xgboost_confusion_matrix_unsw.csv",
    "CNN": "results/cnn_confusion_matrix_unsw.csv",
    "AutoGluon": "results/autogluon_confusion_matrix_unsw.csv"
}

MODEL_FILES_CLEAN = {
    "Decision Tree": "results/decision_tree_confusion_matrix_unsw_clean.csv",
    "XGBoost": "results/xgboost_confusion_matrix_unsw_clean.csv",
    "CNN": "results/cnn_confusion_matrix_unsw_clean.csv",
    "AutoGluon": "results/autogluon_confusion_matrix_unsw_clean.csv"
}

# Label order matches preprocess_unsw.py's LabelEncoder (alphabetical):
# 0=Analysis 1=Backdoor 2=DoS 3=Exploits 4=Fuzzers 5=Generic
# 6=Normal 7=Reconnaissance 8=Shellcode 9=Worms
class_names = [
    "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers",
    "Generic", "Normal", "Reconnaissance", "Shellcode", "Worms"
]

# Minority-class deep dive: Worms (~130 train rows pre-GAN) and Shellcode
# are UNSW-NB15's smallest classes, same role R2L/U2R played for NSL-KDD.
WORMS_IDX = class_names.index("Worms")
SHELLCODE_IDX = class_names.index("Shellcode")

os.makedirs("results", exist_ok=True)


def compare_models(model_files, label):
    """Build a comparison table from a dict of {model_name: cm_csv_path}."""

    all_results = []

    for model_name, file_path in model_files.items():

        if not os.path.exists(file_path):
            print(f"\nSKIPPING {model_name} ({label}): {file_path} not found yet.")
            continue

        print("\n" + "=" * 60)
        print(f"{model_name.upper()} ({label})")
        print("=" * 60)

        cm_df = pd.read_csv(file_path, index_col=0)
        cm = cm_df.values

        print("\nConfusion Matrix:")
        print(cm_df)

        total = cm.sum()
        true_positives = cm.diagonal()
        false_negatives = cm.sum(axis=1) - true_positives
        false_positives = cm.sum(axis=0) - true_positives

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * (precision * recall) / (precision + recall)

        # Handle possible NaN (zero-support classes, zero-denominator cases)
        precision = pd.Series(precision).fillna(0).values
        recall = pd.Series(recall).fillna(0).values
        f1 = pd.Series(f1).fillna(0).values

        accuracy = true_positives.sum() / total

        macro_precision = precision.mean()
        macro_recall = recall.mean()
        macro_f1 = f1.mean()

        class_support = cm.sum(axis=1)

        weighted_precision = (precision * class_support).sum() / total
        weighted_recall = (recall * class_support).sum() / total
        weighted_f1 = (f1 * class_support).sum() / total

        print("\nPer-Class Metrics:")
        for i, class_name in enumerate(class_names):
            print(
                f"{class_name}: "
                f"Precision={precision[i]:.4f}, "
                f"Recall={recall[i]:.4f}, "
                f"F1={f1[i]:.4f}"
            )

        print("\nOverall Metrics:")
        print(f"Accuracy:          {accuracy:.4f}")
        print(f"Macro Precision:   {macro_precision:.4f}")
        print(f"Macro Recall:      {macro_recall:.4f}")
        print(f"Macro F1:          {macro_f1:.4f}")
        print(f"Weighted Precision:{weighted_precision:.4f}")
        print(f"Weighted Recall:   {weighted_recall:.4f}")
        print(f"Weighted F1:       {weighted_f1:.4f}")

        all_results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Macro Precision": macro_precision,
            "Macro Recall": macro_recall,
            "Macro F1": macro_f1,
            "Weighted Precision": weighted_precision,
            "Weighted Recall": weighted_recall,
            "Weighted F1": weighted_f1,
            "Worms Precision": precision[WORMS_IDX],
            "Worms Recall": recall[WORMS_IDX],
            "Worms F1": f1[WORMS_IDX],
            "Shellcode Precision": precision[SHELLCODE_IDX],
            "Shellcode Recall": recall[SHELLCODE_IDX],
            "Shellcode F1": f1[SHELLCODE_IDX]
        })

    return pd.DataFrame(all_results)


# ============================================================
# 2. FULL TEST SET COMPARISON (inflated by duplicate rows -- reference only)
# ============================================================

print("\n" + "#" * 80)
print("# FULL TEST SET (raw official split -- includes duplicate rows)")
print("#" * 80)

comparison_full = compare_models(MODEL_FILES_FULL, "full test")

if not comparison_full.empty:
    comparison_full = comparison_full.sort_values(by="Accuracy", ascending=False)
    comparison_full.to_csv("results/final_model_comparison_unsw_full.csv", index=False)

    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON -- FULL TEST SET (UNSW-NB15)")
    print("DO NOT quote these numbers as final results -- 42.74% of this test")
    print("set is exact duplicates of training rows (see 00_check_overlap_unsw.py)")
    print("=" * 80)
    print(comparison_full.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved to: results/final_model_comparison_unsw_full.csv")
else:
    print("\nNo full-test confusion matrices found yet -- run the 4 classifier scripts first.")

# ============================================================
# 3. CLEAN TEST SET COMPARISON (the numbers to actually report)
# ============================================================

print("\n\n" + "#" * 80)
print("# CLEAN TEST SET (duplicate rows removed -- these are the real numbers)")
print("#" * 80)

comparison_clean = compare_models(MODEL_FILES_CLEAN, "clean test")

if not comparison_clean.empty:
    comparison_clean = comparison_clean.sort_values(by="Accuracy", ascending=False)
    comparison_clean.to_csv("results/final_model_comparison_unsw_clean.csv", index=False)

    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON -- CLEAN TEST SET (UNSW-NB15)")
    print("This is the table to use in the report.")
    print("=" * 80)
    print(comparison_clean.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved to: results/final_model_comparison_unsw_clean.csv")
else:
    print("\nNo clean-test confusion matrices found yet.")
    print("Run build_clean_test_unsw.py, then re-run the 4 classifier scripts,")
    print("then re-run this comparison.")

print("\n" + "=" * 80)
print("COMPARISON COMPLETE")
print("=" * 80)

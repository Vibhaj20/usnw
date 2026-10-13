import pandas as pd
import os

from autogluon.tabular import TabularPredictor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. FILE PATHS

TRAIN_X = "unsw_final_augmented/X_train_augmented.csv"
TRAIN_Y = "unsw_final_augmented/y_train_multiclass_augmented.csv"

TEST_X = "unsw_preprocessed/X_test.csv"
TEST_Y = "unsw_preprocessed/y_test_multiclass.csv"

# Clean test set (no exact duplicates of training rows) -- built by
# build_clean_test_unsw.py after 00_check_overlap_unsw.py found 42.74%
# overlap between the official train/test split.
TEST_X_CLEAN = "unsw_preprocessed/X_test_clean.csv"
TEST_Y_CLEAN = "unsw_preprocessed/y_test_multiclass_clean.csv"

LABEL_COL = "label"
MODEL_PATH = "models/autogluon_model_unsw"

os.makedirs("results", exist_ok=True)

# 2. LOAD DATA

print("=" * 60)
print("AUTOGLUON - IDS CLASSIFIER (UNSW-NB15)")
print("=" * 60)

print("\nLoading training data...")
X_train = pd.read_csv(TRAIN_X)
y_train = pd.read_csv(TRAIN_Y)

print("Training features:", X_train.shape)
print("Training labels:", y_train.shape)

print("\nLoading test data...")
X_test = pd.read_csv(TEST_X)
y_test = pd.read_csv(TEST_Y)

print("Test features:", X_test.shape)
print("Test labels:", y_test.shape)

# 3. EXTRACT LABEL COLUMN / BUILD COMBINED FRAMES
# AutoGluon expects one dataframe with features + label column together

y_train = y_train.iloc[:, 0]
y_test = y_test.iloc[:, 0]

train_df = X_train.copy()
train_df[LABEL_COL] = y_train.values

test_df = X_test.copy()
test_df[LABEL_COL] = y_test.values

print("\nTraining class distribution:")
print(train_df[LABEL_COL].value_counts().sort_index())

print("\nTest class distribution:")
print(test_df[LABEL_COL].value_counts().sort_index())

# 4. TRAIN AUTOGLUON

print("\n" + "=" * 60)
print("TRAINING AUTOGLUON...")
print("=" * 60)

predictor = TabularPredictor(
    label=LABEL_COL,
    path=MODEL_PATH,
    problem_type="multiclass",
    eval_metric="accuracy"
).fit(
    train_df,
    presets="medium_quality",
    time_limit=600
)

print("\nAutoGluon training completed!")

# 5. PREDICTION

print("\nGenerating predictions...")
y_pred = predictor.predict(test_df.drop(columns=[LABEL_COL]))
print("Prediction completed!")

# 6. ACCURACY

accuracy = accuracy_score(test_df[LABEL_COL], y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)
print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

# 7. CLASSIFICATION REPORT

class_names = [
    "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers",
    "Generic", "Normal", "Reconnaissance", "Shellcode", "Worms"
]
all_labels = list(range(10))

print("\nClassification Report:")
print(
    classification_report(
        test_df[LABEL_COL], y_pred, labels=all_labels, target_names=class_names,
        digits=4, zero_division=0
    )
)

# 8. CONFUSION MATRIX

cm = confusion_matrix(test_df[LABEL_COL], y_pred, labels=all_labels)
print("\nConfusion Matrix:")
print(cm)

cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
cm_df.to_csv("results/autogluon_confusion_matrix_unsw.csv")

# 9. LEADERBOARD (which internal model AutoGluon picked/ranked)

leaderboard = predictor.leaderboard(test_df, silent=True)
leaderboard.to_csv("results/autogluon_leaderboard_unsw.csv", index=False)
print("\nLeaderboard saved to results/autogluon_leaderboard_unsw.csv")

# 10. EVALUATE AGAINST CLEAN TEST SET (no train/test duplicate rows)
# This is the number that belongs in the report. The full-test-set
# accuracy above is inflated by the 42.74% of test rows that are exact
# duplicates of training rows (see 00_check_overlap_unsw.py).

accuracy_clean = None

if os.path.exists(TEST_X_CLEAN) and os.path.exists(TEST_Y_CLEAN):
    print("\n" + "=" * 60)
    print("EVALUATING ON CLEAN TEST SET (duplicates removed)")
    print("=" * 60)

    X_test_clean = pd.read_csv(TEST_X_CLEAN)
    y_test_clean_col = pd.read_csv(TEST_Y_CLEAN).iloc[:, 0]

    test_df_clean = X_test_clean.copy()
    test_df_clean[LABEL_COL] = y_test_clean_col.values

    print("Clean test features:", X_test_clean.shape)

    y_pred_clean = predictor.predict(test_df_clean.drop(columns=[LABEL_COL]))
    accuracy_clean = accuracy_score(test_df_clean[LABEL_COL], y_pred_clean)

    print(f"\nClean-test Accuracy: {accuracy_clean:.4f}")
    print(f"Clean-test Accuracy: {accuracy_clean * 100:.2f}%")
    print(f"(Full-test Accuracy was: {accuracy * 100:.2f}% -- inflated by duplicate rows)")

    print("\nClean-Test Classification Report:")
    print(
        classification_report(
            test_df_clean[LABEL_COL], y_pred_clean, labels=all_labels,
            target_names=class_names, digits=4, zero_division=0
        )
    )

    cm_clean = confusion_matrix(test_df_clean[LABEL_COL], y_pred_clean, labels=all_labels)
    cm_clean_df = pd.DataFrame(cm_clean, index=class_names, columns=class_names)
    cm_clean_df.to_csv("results/autogluon_confusion_matrix_unsw_clean.csv")
    print("\nClean confusion matrix saved to:")
    print("results/autogluon_confusion_matrix_unsw_clean.csv")
else:
    print("\nNOTE: unsw_preprocessed/X_test_clean.csv not found.")
    print("Run build_clean_test_unsw.py first to get a leakage-free accuracy number.")

# 11. SAVE RESULTS

results = {
    "Model": ["AutoGluon"],
    "Accuracy_full_test": [accuracy],
    "Accuracy_clean_test": [accuracy_clean],
}
pd.DataFrame(results).to_csv("results/autogluon_results_unsw.csv", index=False)

print("\nResults saved to:")
print("results/autogluon_results_unsw.csv")
print("results/autogluon_confusion_matrix_unsw.csv")
print("Model saved to:", MODEL_PATH)

print("\n" + "=" * 60)
print("AUTOGLUON COMPLETE")
print("=" * 60)

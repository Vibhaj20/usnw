import pandas as pd
import os
import joblib

from xgboost import XGBClassifier
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

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 2. LOAD DATA

print("=" * 60)
print("XGBOOST - IDS CLASSIFIER (UNSW-NB15)")
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

# 3. EXTRACT LABEL COLUMN

y_train = y_train.iloc[:, 0]
y_test = y_test.iloc[:, 0]

print("\nTraining class distribution:")
print(y_train.value_counts().sort_index())

print("\nTest class distribution:")
print(y_test.value_counts().sort_index())

# 4. CREATE XGBOOST MODEL
# num_class=10 -- UNSW-NB15 has 10 classes (9 attacks + Normal), not 5

print("\n" + "=" * 60)
print("CREATING XGBOOST MODEL...")
print("=" * 60)

model = XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=10,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

# 5. TRAIN MODEL

print("\nTraining XGBoost...")
model.fit(X_train, y_train)
print("\nXGBoost training completed!")

# 6. PREDICTION

print("\nGenerating predictions...")
y_pred = model.predict(X_test)
print("Prediction completed!")

# 7. ACCURACY

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)
print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

# 8. CLASSIFICATION REPORT

class_names = [
    "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers",
    "Generic", "Normal", "Reconnaissance", "Shellcode", "Worms"
]
all_labels = list(range(10))

print("\nClassification Report:")
print(
    classification_report(
        y_test, y_pred, labels=all_labels, target_names=class_names,
        digits=4, zero_division=0
    )
)

# 9. CONFUSION MATRIX

cm = confusion_matrix(y_test, y_pred, labels=all_labels)
print("\nConfusion Matrix:")
print(cm)

# 10. SAVE CONFUSION MATRIX

cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
cm_df.to_csv("results/xgboost_confusion_matrix_unsw.csv")

# 11. SAVE MODEL

model_path = "models/xgboost_model_unsw.pkl"
joblib.dump(model, model_path)
print("\nModel saved to:", model_path)

# 12. EVALUATE AGAINST CLEAN TEST SET (no train/test duplicate rows)
# This is the number that belongs in the report. The full-test-set
# accuracy above is inflated by the 42.74% of test rows that are exact
# duplicates of training rows (see 00_check_overlap_unsw.py).

accuracy_clean = None

if os.path.exists(TEST_X_CLEAN) and os.path.exists(TEST_Y_CLEAN):
    print("\n" + "=" * 60)
    print("EVALUATING ON CLEAN TEST SET (duplicates removed)")
    print("=" * 60)

    X_test_clean = pd.read_csv(TEST_X_CLEAN)
    y_test_clean = pd.read_csv(TEST_Y_CLEAN).iloc[:, 0]

    print("Clean test features:", X_test_clean.shape)

    y_pred_clean = model.predict(X_test_clean)
    accuracy_clean = accuracy_score(y_test_clean, y_pred_clean)

    print(f"\nClean-test Accuracy: {accuracy_clean:.4f}")
    print(f"Clean-test Accuracy: {accuracy_clean * 100:.2f}%")
    print(f"(Full-test Accuracy was: {accuracy * 100:.2f}% -- inflated by duplicate rows)")

    print("\nClean-Test Classification Report:")
    print(
        classification_report(
            y_test_clean, y_pred_clean, labels=all_labels,
            target_names=class_names, digits=4, zero_division=0
        )
    )

    cm_clean = confusion_matrix(y_test_clean, y_pred_clean, labels=all_labels)
    cm_clean_df = pd.DataFrame(cm_clean, index=class_names, columns=class_names)
    cm_clean_df.to_csv("results/xgboost_confusion_matrix_unsw_clean.csv")
    print("\nClean confusion matrix saved to:")
    print("results/xgboost_confusion_matrix_unsw_clean.csv")
else:
    print("\nNOTE: unsw_preprocessed/X_test_clean.csv not found.")
    print("Run build_clean_test_unsw.py first to get a leakage-free accuracy number.")

# 13. SAVE RESULTS

results = {
    "Model": ["XGBoost"],
    "Accuracy_full_test": [accuracy],
    "Accuracy_clean_test": [accuracy_clean],
}
pd.DataFrame(results).to_csv("results/xgboost_results_unsw.csv", index=False)

print("\nResults saved to:")
print("results/xgboost_results_unsw.csv")
print("results/xgboost_confusion_matrix_unsw.csv")

print("\n" + "=" * 60)
print("XGBOOST COMPLETE")
print("=" * 60)

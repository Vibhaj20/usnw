import pandas as pd
import numpy as np
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
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

NUM_CLASSES = 10

# 2. LOAD DATA

print("=" * 60)
print("CNN - IDS CLASSIFIER (UNSW-NB15)")
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

# 4. RESHAPE FOR CNN (samples, features, 1)
# UNSW-NB15 has 69 features after preprocessing, not NSL-KDD's original count

n_features = X_train.shape[1]
print(f"\nNumber of features: {n_features}")

X_train_cnn = X_train.values.reshape(-1, n_features, 1)
X_test_cnn = X_test.values.reshape(-1, n_features, 1)

y_train_cat = to_categorical(y_train, num_classes=NUM_CLASSES)
y_test_cat = to_categorical(y_test, num_classes=NUM_CLASSES)

print("X_train_cnn shape:", X_train_cnn.shape)
print("X_test_cnn shape:", X_test_cnn.shape)

# 5. BUILD CNN MODEL

print("\n" + "=" * 60)
print("BUILDING CNN MODEL...")
print("=" * 60)

model = Sequential([
    Conv1D(64, kernel_size=3, activation="relu", input_shape=(n_features, 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(128, kernel_size=3, activation="relu"),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 6. TRAIN MODEL

print("\nTraining CNN...")
history = model.fit(
    X_train_cnn, y_train_cat,
    validation_split=0.1,
    epochs=20,
    batch_size=256,
    verbose=1
)
print("\nCNN training completed!")

# 7. PREDICTION

print("\nGenerating predictions...")
y_pred_probs = model.predict(X_test_cnn)
y_pred = np.argmax(y_pred_probs, axis=1)
print("Prediction completed!")

# 8. ACCURACY

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)
print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

# 9. CLASSIFICATION REPORT

class_names = [
    "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers",
    "Generic", "Normal", "Reconnaissance", "Shellcode", "Worms"
]
all_labels = list(range(NUM_CLASSES))

print("\nClassification Report:")
print(
    classification_report(
        y_test, y_pred, labels=all_labels, target_names=class_names,
        digits=4, zero_division=0
    )
)

# 10. CONFUSION MATRIX

cm = confusion_matrix(y_test, y_pred, labels=all_labels)
print("\nConfusion Matrix:")
print(cm)

cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
cm_df.to_csv("results/cnn_confusion_matrix_unsw.csv")

# 11. SAVE MODEL

model_path = "models/cnn_model_unsw.keras"
model.save(model_path)
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

    X_test_clean_cnn = X_test_clean.values.reshape(-1, n_features, 1)

    print("Clean test features:", X_test_clean.shape)

    y_pred_clean_probs = model.predict(X_test_clean_cnn)
    y_pred_clean = np.argmax(y_pred_clean_probs, axis=1)
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
    cm_clean_df.to_csv("results/cnn_confusion_matrix_unsw_clean.csv")
    print("\nClean confusion matrix saved to:")
    print("results/cnn_confusion_matrix_unsw_clean.csv")
else:
    print("\nNOTE: unsw_preprocessed/X_test_clean.csv not found.")
    print("Run build_clean_test_unsw.py first to get a leakage-free accuracy number.")

# 13. SAVE RESULTS

results = {
    "Model": ["CNN"],
    "Accuracy_full_test": [accuracy],
    "Accuracy_clean_test": [accuracy_clean],
}
pd.DataFrame(results).to_csv("results/cnn_results_unsw.csv", index=False)

print("\nResults saved to:")
print("results/cnn_results_unsw.csv")
print("results/cnn_confusion_matrix_unsw.csv")

print("\n" + "=" * 60)
print("CNN COMPLETE")
print("=" * 60)

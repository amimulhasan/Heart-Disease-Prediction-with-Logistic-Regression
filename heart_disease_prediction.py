"""
Heart Disease Prediction using Logistic Regression
===================================================
Full pipeline: load → EDA → train → evaluate → plots → predictive system
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


def load_dataset():
    for path in [
        "heart_disease_data.csv",
        "data.csv",
        "/content/heart_disease_data.csv",
        "/content/data.csv",
    ]:
        if os.path.exists(path):
            print(f"Loading: {path}")
            return pd.read_csv(path)
    raise FileNotFoundError(
        "Dataset not found. Place heart_disease_data.csv in the project folder."
    )


def main():
    print("=" * 60)
    print("  Heart Disease Prediction | Logistic Regression")
    print("=" * 60)

    # 1. Load
    print("\n[1/7] Loading dataset...")
    heart_data = load_dataset()
    print(f"Shape: {heart_data.shape}")
    print("\nFirst 5 rows:\n", heart_data.head())
    print("\nLast 5 rows:\n", heart_data.tail())
    print("\nInfo:")
    print(heart_data.info())
    print("\nMissing values:\n", heart_data.isnull().sum())
    print("\nStatistical summary:\n", heart_data.describe())

    # 2. Target distribution
    print("\n[2/7] Target distribution...")
    print(heart_data["target"].value_counts())
    print("1 → Defective Heart | 0 → Healthy Heart")

    plt.figure(figsize=(6, 4))
    sns.countplot(x="target", data=heart_data, palette=["#2ecc71", "#e74c3c"])
    plt.title("Heart Disease Target Distribution")
    plt.xlabel("Target (0=Healthy, 1=Disease)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("target_distribution.png", dpi=150)
    print("Saved: target_distribution.png")
    plt.close()

    # 3. EDA plots
    print("\n[3/7] Generating EDA plots...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        heart_data.corr(),
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        square=True,
        cbar=True,
        annot_kws={"size": 7},
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=150)
    print("Saved: correlation_heatmap.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(data=heart_data, x="age", hue="target", kde=True, palette=["#2ecc71", "#e74c3c"])
    plt.title("Age Distribution by Heart Disease Status")
    plt.tight_layout()
    plt.savefig("age_by_target.png", dpi=150)
    print("Saved: age_by_target.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.countplot(x="cp", hue="target", data=heart_data, palette=["#2ecc71", "#e74c3c"])
    plt.title("Chest Pain Type vs Heart Disease")
    plt.xlabel("Chest Pain Type (cp)")
    plt.tight_layout()
    plt.savefig("cp_vs_target.png", dpi=150)
    print("Saved: cp_vs_target.png")
    plt.close()

    # 4. Split
    print("\n[4/7] Train / Test split (80/20, stratified)...")
    X = heart_data.drop(columns="target", axis=1)
    Y = heart_data["target"]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=2
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # 5. Train
    print("\n[5/7] Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000, random_state=2)
    model.fit(X_train, Y_train)
    print("Model trained successfully!")

    # 6. Evaluate
    print("\n[6/7] Evaluating model...")
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(Y_train, train_pred)
    test_acc = accuracy_score(Y_test, test_pred)

    print(f"\nAccuracy on Training data : {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Accuracy on Test data     : {test_acc:.4f} ({test_acc*100:.2f}%)")

    print("\nClassification Report (Test Set):")
    print(
        classification_report(
            Y_test,
            test_pred,
            target_names=["Healthy (0)", "Disease (1)"],
        )
    )

    cm = confusion_matrix(Y_test, test_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Healthy", "Disease"],
        yticklabels=["Healthy", "Disease"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Saved: confusion_matrix.png")
    plt.close()

    # 7. Predictive system
    print("\n[7/7] Predictive system demo...")
    input_data = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2)
    input_array = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    print(f"Input sample: {input_data}")
    print(f"Prediction: {prediction[0]}")
    if prediction[0] == 0:
        print("The Person does not have a Heart Disease")
    else:
        print("The Person has Heart Disease")

    print("\n" + "=" * 60)
    print("Done! Generated plots:")
    print("  - target_distribution.png")
    print("  - correlation_heatmap.png")
    print("  - age_by_target.png")
    print("  - cp_vs_target.png")
    print("  - confusion_matrix.png")
    print("=" * 60)


if __name__ == "__main__":
    main()

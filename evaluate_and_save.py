import os
os.makedirs("models", exist_ok=True)

import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from preprocessing import preprocess_data
from data_loader import load_data


def evaluate_and_save_model(X, y, scaler):

    # -------------------------
    # Train-test split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -------------------------
    # Handle class imbalance
    # -------------------------
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # -------------------------
    # Models
    # -------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True)
    }

    best_model = None
    best_f1 = 0

    model_names = []
    f1_scores = []

    print("\n📊 Model Evaluation Results\n")

    # -------------------------
    # Training loop
    # -------------------------
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"{name}")
        print(f" Accuracy : {acc:.4f}")
        print(f" Precision: {prec:.4f}")
        print(f" Recall   : {rec:.4f}")
        print(f" F1 Score : {f1:.4f}\n")

        model_names.append(name)
        f1_scores.append(f1)

        # Select best model
        if f1 > best_f1:
            best_f1 = f1
            best_model = model

    # -------------------------
    # Show best model
    # -------------------------
    print(f"🏆 Best Model Selected: {best_model.__class__.__name__}")

    # -------------------------
    # Confusion Matrix (best model)
    # -------------------------
    y_pred_best = best_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred_best)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix (Best Model)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # -------------------------
    # Model Comparison Graph
    # -------------------------
    plt.figure(figsize=(6, 4))
    plt.bar(model_names, f1_scores)
    plt.title("Model Comparison (F1 Score)")
    plt.xlabel("Models")
    plt.ylabel("F1 Score")
    plt.xticks(rotation=30)
    plt.show()

    # -------------------------
    # Save model and scaler
    # -------------------------
    with open("models/model_and_scaler.pkl", "wb") as file:
        pickle.dump({
            "model": best_model,
            "scaler": scaler
        }, file)

    print("✅ Model and scaler saved successfully")


if __name__ == "__main__":
    df = load_data("C:\\Users\\arund\\Desktop\\Stroke_and_Cardiac\\health_data.csv")
    X, y, scaler = preprocess_data(df)
    evaluate_and_save_model(X, y, scaler)
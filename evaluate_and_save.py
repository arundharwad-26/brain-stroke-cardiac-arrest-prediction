import os
os.makedirs("../models", exist_ok=True)
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from preprocessing import preprocess_data
from data_loader import load_data

def evaluate_and_save_model(X,y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=0.2, random_state=42
    )
    models={
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "SVM": SVC()
    }
    best_model = None
    best_f1 = 0

    print("\n model evaluation results\n")

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc=accuracy_score(y_test,y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        # prec=precision_score(y_test,y_pred)
        # rec=recall_score(y_test, y_pred)
        # f1=f1_score(y_test,y_pred)

        print(f"{name}")
        print(f" Accuracy : {acc:.4f}")
        print(f" Precision: {prec:.4f}")
        print(f" Recall   : {rec:.4f}")
        print(f" F1 Score : {f1:.4f}\n")

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
    # Save best model
    with open("../models/best_model.pkl", "wb") as file:
        pickle.dump(best_model, file)
    print("✅ Best model saved as best_model.pkl")

if __name__ == "__main__":
    df = load_data("C:\\Users\\arund\\Desktop\\Stroke_and_Cardiac\\health_data.csv")
    X, y = preprocess_data(df)
    evaluate_and_save_model(X, y)
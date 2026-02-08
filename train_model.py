# src/train_model.py

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from preprocessing import preprocess_data
from data_loader import load_data

def train_models(X,y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=0.2, random_state=42
    )
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "SVM":SVC()
    }
    print("\n Model training results\n")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {acc:.4f}")
    return models

if __name__ == "__main__":
    df = load_data("C:\\Users\\arund\\Desktop\\Stroke_and_Cardiac\\health_data.csv")
    X,y = preprocess_data(df)
    train_models(X,y)
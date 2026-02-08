# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_data(df):
    data = df.copy()

    # -------------------------
    # Handle missing values
    # -------------------------
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = data[column].fillna(data[column].mode()[0])
        else:
            data[column] = data[column].fillna(data[column].mean())

    # -------------------------
    # Encode categorical columns
    # -------------------------
    label_encoder = LabelEncoder()
    categorical_cols = data.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        data[col] = label_encoder.fit_transform(data[col])

    # -------------------------
    # Drop non-useful column
    # -------------------------
    if 'id' in data.columns:
        data = data.drop('id', axis=1)

    # -------------------------
    # Split features and target
    # -------------------------
    X = data.drop('stroke', axis=1)
    y = data['stroke']

    # -------------------------
    # Final NaN safety
    # -------------------------
    X = X.fillna(0)

    # -------------------------
    # Feature scaling
    # -------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("✅ Data preprocessing completed")
    print(f"📊 Processed feature shape: {X_scaled.shape}")

    return X_scaled, y


if __name__ == "__main__":
    from data_loader import load_data

    df = load_data("C:\\Users\\arund\\Desktop\\Stroke_and_Cardiac\\health_data.csv")
    X, y = preprocess_data(df)

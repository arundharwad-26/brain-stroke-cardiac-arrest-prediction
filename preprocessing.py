# src/preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_data(df):
    data = df.copy()

    # Handle missing values
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = data[column].fillna(data[column].mode()[0])
        else:
            data[column] = data[column].fillna(data[column].mean())

    # Encode categorical columns
    categorical_cols = data.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    # Drop ID column
    if 'id' in data.columns:
        data = data.drop('id', axis=1)

    # Split features and target
    X = data.drop('stroke', axis=1)
    y = data['stroke']

    # Final NaN safety
    X = X.fillna(0)

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("✅ Data preprocessing completed")
    print(f"📊 Processed feature shape: {X_scaled.shape}")

    return X_scaled, y, scaler
# predict.py
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

from preprocessing import preprocess_data
from data_loader import load_data


def load_model():
    with open("models/best_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model


def predict_risk(input_data):
    model = load_model()

    # Convert input to array
    input_array = np.array(input_data).reshape(1, -1)

    # IMPORTANT: scale input the same way as training
    scaler = StandardScaler()
    input_array = scaler.fit_transform(input_array)

    prediction = model.predict(input_array)[0]

    if prediction == 0:
        return "LOW RISK"
    else:
        return "HIGH RISK"


if __name__ == "__main__":
    print("\n🩺 Stroke & Cardiac Risk Prediction System\n")

    print("Enter patient details:")

    gender = int(input("Gender (0 = Female, 1 = Male): "))
    age = float(input("Age: "))
    hypertension = int(input("Hypertension (0 = No, 1 = Yes): "))
    heart_disease = int(input("Heart Disease (0 = No, 1 = Yes): "))
    ever_married = int(input("Ever Married (0 = No, 1 = Yes): "))
    work_type = int(input("Work Type (0–4): "))
    residence_type = int(input("Residence Type (0 = Rural, 1 = Urban): "))
    avg_glucose = float(input("Average Glucose Level: "))
    bmi = float(input("BMI: "))
    smoking_status = int(input("Smoking Status (0–3): "))

    user_input = [
        gender,
        age,
        hypertension,
        heart_disease,
        ever_married,
        work_type,
        residence_type,
        avg_glucose,
        bmi,
        smoking_status
    ]

    result = predict_risk(user_input)

    print("\n🧠 Prediction Result:")
    print(f"👉 Patient is at **{result}**")

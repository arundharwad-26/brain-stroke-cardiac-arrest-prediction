# predict.py
import pickle
import numpy as np


def load_model():
    with open("models/model_and_scaler.pkl", "rb") as file:
        data = pickle.load(file)
    return data["model"], data["scaler"]


def predict_risk(input_data):
    model, scaler = load_model()

    input_array = np.array(input_data).reshape(1, -1)
    input_array = scaler.transform(input_array)

    probability = model.predict_proba(input_array)[0][1]

    if probability < 0.30:
        return f"LOW RISK (Probability: {probability:.2f})"
    elif probability < 0.60:
        return f"MEDIUM RISK (Probability: {probability:.2f})"
    else:
        return f"HIGH RISK (Probability: {probability:.2f})"


# -------------------------
# Safe input handling
# -------------------------
def safe_int(prompt):
    try:
        return int(input(prompt))
    except:
        print("Invalid input! Please enter a number.")
        return safe_int(prompt)


def safe_float(prompt):
    try:
        return float(input(prompt))
    except:
        print("Invalid input! Please enter a valid number.")
        return safe_float(prompt)


if __name__ == "__main__":
    print("\n==============================")
    print("🩺 Stroke & Cardiac Risk Prediction System")
    print("==============================\n")

    gender = safe_int("Gender (0 = Female, 1 = Male): ")
    age = safe_float("Age: ")
    hypertension = safe_int("Hypertension (0 = No, 1 = Yes): ")
    heart_disease = safe_int("Heart Disease (0 = No, 1 = Yes): ")
    ever_married = safe_int("Ever Married (0 = No, 1 = Yes): ")
    work_type = safe_int("Work Type (0–4): ")
    residence_type = safe_int("Residence Type (0 = Rural, 1 = Urban): ")
    avg_glucose = safe_float("Average Glucose Level: ")
    bmi = safe_float("BMI: ")
    smoking_status = safe_int("Smoking Status (0–3): ")

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
    print(f"👉 Risk Level: {result}")
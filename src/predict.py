import joblib
import pandas as pd

MODEL_PATH = "models/churn_model.pkl"


def predict_churn(customer_data: dict):
    model = joblib.load(MODEL_PATH)
    input_df = pd.DataFrame([customer_data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return prediction, probability


if __name__ == "__main__":
    sample_customer = {
        "gender": "Female",
        "age": 35,
        "tenure_months": 12,
        "monthly_charges": 85.0,
        "total_charges": 1020.0,
        "contract_type": "Month-to-month",
        "payment_method": "Electronic Check",
        "internet_service": "Fiber optic",
        "support_calls": 6,
        "streaming_service": "Yes",
        "senior_citizen": 0,
        "paperless_billing": "Yes",
    }

    prediction, probability = predict_churn(sample_customer)

    print("Churn prediction:", "Yes" if prediction == 1 else "No")
    print(f"Churn probability: {probability:.2%}")

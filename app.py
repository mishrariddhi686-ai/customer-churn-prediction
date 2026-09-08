import os
import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = "models/churn_model.pkl"

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


st.title("Customer Churn Prediction")
st.write(
    "Enter customer information to predict whether the customer is likely to churn."
)

model = load_model()

if model is None:
    st.error(
        "Model not found. First run: python src/data_preprocessing.py "
        "then: python src/train_model.py"
    )
    st.stop()

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    tenure_months = st.number_input(
        "Tenure (months)", min_value=1, max_value=120, value=12
    )
    monthly_charges = st.number_input(
        "Monthly Charges", min_value=0.0, value=80.0, step=1.0
    )

with col2:
    contract_type = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"],
    )
    payment_method = st.selectbox(
        "Payment Method",
        ["Credit Card", "Bank Transfer", "Electronic Check"],
    )
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"],
    )
    support_calls = st.number_input(
        "Support Calls", min_value=0, max_value=30, value=3
    )

with col3:
    streaming_service = st.selectbox("Streaming Service", ["Yes", "No"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

total_charges = monthly_charges * tenure_months

st.info(f"Calculated Total Charges: {total_charges:.2f}")

if st.button("Predict Churn", type="primary"):
    customer = pd.DataFrame(
        [{
            "gender": gender,
            "age": age,
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract_type,
            "payment_method": payment_method,
            "internet_service": internet_service,
            "support_calls": support_calls,
            "streaming_service": streaming_service,
            "senior_citizen": senior_citizen,
            "paperless_billing": paperless_billing,
        }]
    )

    prediction = model.predict(customer)[0]
    probability = model.predict_proba(customer)[0][1]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.warning("High risk: Customer is likely to churn.")
    else:
        st.success("Low risk: Customer is likely to stay.")

    st.metric("Churn Probability", f"{probability:.1%}")

    if probability >= 0.60:
        st.write(
            "The model estimates a relatively high churn risk. "
            "Consider proactive customer-retention actions."
        )
    elif probability >= 0.40:
        st.write(
            "The customer has a moderate churn risk. "
            "Monitoring and engagement may be useful."
        )
    else:
        st.write(
            "The customer has a relatively low predicted churn risk."
        )

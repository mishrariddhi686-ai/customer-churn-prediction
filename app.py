from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
DATA_PATH = BASE_DIR / "data" / "customer_churn.csv"

st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

    .stApp {
        background: #f5f7fb;
        color: #172033;
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stSidebar"] {
        background: #111827;
    }
    [data-testid="stSidebar"] * {
        color: #d7deeb;
    }
    .brand {
        padding: 1.25rem 0 2rem;
    }
    .brand h1 {
        color: #ffffff;
        font-size: 1.35rem;
        margin: 0.5rem 0 0.25rem;
    }
    .brand p {
        color: #8d99ae;
        font-size: 0.82rem;
        margin: 0;
    }
    .eyebrow {
        color: #667085;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    .hero h1 {
        color: #172033;
        font-size: 2.15rem;
        margin: 0;
    }
    .hero p {
        color: #667085;
        margin-top: 0.35rem;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e8ecf3;
        border-radius: 14px;
        padding: 1.15rem 1.25rem;
        box-shadow: 0 5px 18px rgba(23, 32, 51, 0.04);
    }
    [data-testid="stMetricLabel"] {
        color: #667085;
    }
    [data-testid="stMetricValue"] {
        color: #172033;
    }
    .section-title {
        color: #172033;
        font-size: 1.1rem;
        font-weight: 700;
        margin: 1.75rem 0 0.75rem;
    }
    div.stButton > button {
        border-radius: 9px;
        font-weight: 600;
        min-height: 2.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


def render_overview(data):
    total_customers = len(data)
    churned_customers = int((data["churn"] == "Yes").sum())
    churn_rate = churned_customers / total_customers if total_customers else 0
    average_charges = data["monthly_charges"].mean()
    average_tenure = data["tenure_months"].mean()

    st.markdown('<div class="section-title">Business snapshot</div>', unsafe_allow_html=True)
    metrics = st.columns(4)
    metrics[0].metric("Total customers", f"{total_customers:,}", "Active records")
    metrics[1].metric("Churn rate", f"{churn_rate:.1%}", f"{churned_customers:,} customers at risk")
    metrics[2].metric("Avg. monthly charges", f"${average_charges:,.2f}", "Across all customers")
    metrics[3].metric("Avg. tenure", f"{average_tenure:.1f} mo", "Customer lifetime")

    st.markdown('<div class="section-title">Customer analytics</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown("**Churn by contract type**")
        contract_summary = (
            data.groupby("contract_type")["churn"]
            .apply(lambda values: (values == "Yes").mean())
            .sort_values(ascending=False)
            .rename("Churn rate")
        )
        st.bar_chart(contract_summary, color="#5b6ee1", use_container_width=True)
    with right:
        st.markdown("**Customer distribution by tenure**")
        tenure_bins = pd.cut(
            data["tenure_months"],
            bins=[0, 12, 24, 48, 120],
            labels=["0–12 months", "13–24 months", "25–48 months", "49+ months"],
        )
        tenure_summary = tenure_bins.value_counts().sort_index().rename("Customers")
        st.bar_chart(tenure_summary, color="#2bb673", use_container_width=True)

    st.markdown('<div class="section-title">Customer statistics</div>', unsafe_allow_html=True)
    stats_left, stats_right = st.columns(2)
    with stats_left:
        st.markdown("**Churn by internet service**")
        internet_summary = (
            data.groupby("internet_service")["churn"]
            .apply(lambda values: (values == "Yes").mean())
            .sort_values(ascending=False)
            .mul(100)
            .round(1)
            .rename("Churn rate (%)")
        )
        st.dataframe(internet_summary, use_container_width=True)
    with stats_right:
        st.markdown("**Customer profile**")
        profile = pd.DataFrame(
            {
                "Metric": ["Median age", "Median monthly charges", "Customers with streaming", "Paperless billing"],
                "Value": [
                    f"{data['age'].median():.0f} years",
                    f"${data['monthly_charges'].median():,.2f}",
                    f"{(data['streaming_service'] == 'Yes').mean():.1%}",
                    f"{(data['paperless_billing'] == 'Yes').mean():.1%}",
                ],
            }
        )
        st.dataframe(profile, hide_index=True, use_container_width=True)


def render_prediction(model):
    st.markdown('<div class="section-title">Predict individual customer risk</div>', unsafe_allow_html=True)
    st.caption("Adjust the customer profile below to generate a churn risk score.")

    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        tenure_months = st.number_input("Tenure (months)", min_value=1, max_value=120, value=12)
        monthly_charges = st.number_input("Monthly charges", min_value=0.0, value=80.0, step=1.0)
    with col2:
        contract_type = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox(
            "Payment method", ["Credit Card", "Bank Transfer", "Electronic Check"]
        )
        internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        support_calls = st.number_input("Support calls", min_value=0, max_value=30, value=3)
    with col3:
        streaming_service = st.selectbox("Streaming service", ["Yes", "No"])
        senior_citizen = st.selectbox("Senior citizen", [0, 1])
        paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])

    total_charges = monthly_charges * tenure_months
    st.info(f"Estimated total charges: ${total_charges:,.2f}")

    if st.button("Calculate churn risk", type="primary", use_container_width=True):
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

        st.markdown('<div class="section-title">Prediction result</div>', unsafe_allow_html=True)
        result_left, result_right = st.columns([1, 2])
        with result_left:
            st.metric("Churn probability", f"{probability:.1%}")
        with result_right:
            if prediction == 1:
                st.warning("High risk — this customer is likely to churn.")
            else:
                st.success("Low risk — this customer is likely to stay.")
            st.progress(float(probability), text="Model confidence")


model = load_model()
data = load_data()

with st.sidebar:
    st.markdown(
        '<div class="brand"><div style="font-size: 2rem;">📈</div>'
        "<h1>Churn Intelligence</h1><p>Customer retention command center</p></div>",
        unsafe_allow_html=True,
    )
    page = st.radio("Navigate", ["Overview", "Risk prediction"], label_visibility="collapsed")
    st.divider()
    st.caption("Model status")
    if model is not None:
        st.success("Ready for predictions")
    else:
        st.error("Model not found")

if model is None:
    st.error(
        "Model not found. Run `python src\\data_preprocessing.py` and "
        "`python src\\train_model.py` from the project directory."
    )
    st.stop()

if data is None:
    st.error("Customer dataset not found at data/customer_churn.csv.")
    st.stop()

st.markdown(
    '<div class="hero"><div class="eyebrow">Customer analytics platform</div>'
    "<h1>Good morning, let's reduce churn.</h1>"
    "<p>Monitor customer health, discover retention opportunities, and predict risk.</p></div>",
    unsafe_allow_html=True,
)

if page == "Overview":
    render_overview(data)
else:
    render_prediction(model)

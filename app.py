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
    .stApp { background: #f5f7fb; color: #172033; font-family: 'DM Sans', sans-serif; }
    [data-testid="stSidebar"] { background: #111827; }
    [data-testid="stSidebar"] * { color: #d7deeb; }
    .brand { padding: 1.25rem 0 2rem; }
    .brand h1 { color: #fff; font-size: 1.35rem; margin: .5rem 0 .25rem; }
    .brand p { color: #8d99ae; font-size: .82rem; margin: 0; }
    .eyebrow { color: #667085; font-size: .78rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; margin-bottom: .35rem; }
    .hero h1 { color: #172033; font-size: 2.15rem; margin: 0; }
    .hero p { color: #667085; margin-top: .35rem; }
    [data-testid="stMetric"] { background: #fff; border: 1px solid #e8ecf3; border-radius: 14px; padding: 1.15rem 1.25rem; box-shadow: 0 5px 18px rgba(23, 32, 51, .04); }
    [data-testid="stMetricLabel"] { color: #667085; }
    [data-testid="stMetricValue"] { color: #172033; }
    .section-title { color: #172033; font-size: 1.1rem; font-weight: 700; margin: 1.75rem 0 .75rem; }
    div.stButton > button { border-radius: 9px; font-weight: 600; min-height: 2.8rem; }
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


@st.cache_data
def score_customers(_model, data):
    features = data.drop(columns=["customer_id", "churn"])
    scored = data.copy()
    scored["churn_probability"] = _model.predict_proba(features)[:, 1]
    scored["risk"] = scored["churn_probability"].apply(risk_level)
    return scored


def risk_level(probability):
    if probability >= 0.60:
        return "High"
    if probability >= 0.35:
        return "Medium"
    return "Low"


def retention_recommendations(customer, risk):
    recommendations = []
    if risk == "High":
        recommendations.append("Prioritize a personal retention call within 48 hours.")
    elif risk == "Medium":
        recommendations.append("Add the customer to a proactive engagement campaign.")
    else:
        recommendations.append("Maintain regular product engagement and satisfaction checks.")
    if customer["contract_type"] == "Month-to-month":
        recommendations.append("Offer a loyalty incentive for a longer-term contract.")
    if customer["support_calls"] >= 5:
        recommendations.append("Escalate support history for service-recovery follow-up.")
    if customer["monthly_charges"] >= 85:
        recommendations.append("Review plan value and offer a personalized pricing option.")
    if not recommendations:
        recommendations.append("Continue monitoring customer health signals.")
    return recommendations


def render_kpis(data):
    total = len(data)
    churned = int((data["churn"] == "Yes").sum())
    churn_rate = churned / total if total else 0
    metrics = st.columns(5)
    metrics[0].metric("Total customers", f"{total:,}")
    metrics[1].metric("Churn rate", f"{churn_rate:.1%}", f"{churned:,} churned")
    metrics[2].metric("Retention rate", f"{1 - churn_rate:.1%}", f"{total - churned:,} retained")
    metrics[3].metric("Avg. monthly charges", f"${data['monthly_charges'].mean():,.2f}")
    metrics[4].metric("Avg. tenure", f"{data['tenure_months'].mean():.1f} mo")


def render_overview(data):
    st.markdown('<div class="section-title">Business snapshot</div>', unsafe_allow_html=True)
    render_kpis(data)
    st.markdown('<div class="section-title">Customer analytics</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown("**Churn by contract type**")
        contract = data.groupby("contract_type")["churn"].apply(lambda v: (v == "Yes").mean()).sort_values(ascending=False)
        st.bar_chart(contract, color="#5b6ee1", use_container_width=True)
    with right:
        st.markdown("**Customer distribution by tenure**")
        bins = pd.cut(data["tenure_months"], [0, 12, 24, 48, 120], labels=["0–12 months", "13–24 months", "25–48 months", "49+ months"])
        st.bar_chart(bins.value_counts().sort_index().rename("Customers"), color="#2bb673", use_container_width=True)


def render_analytics(data):
    st.markdown('<div class="section-title">Churn analytics</div>', unsafe_allow_html=True)
    render_kpis(data)
    st.caption("Use the controls to compare retention performance across customer segments.")
    dimension = st.selectbox("Analyze churn by", ["contract_type", "payment_method", "internet_service", "gender", "streaming_service"])
    grouped = data.groupby(dimension).agg(
        Customers=("customer_id", "count"),
        Churned=("churn", lambda v: (v == "Yes").sum()),
        AverageCharges=("monthly_charges", "mean"),
    )
    grouped["Churn rate"] = grouped["Churned"] / grouped["Customers"]
    grouped["Retention rate"] = 1 - grouped["Churn rate"]
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown(f"**Churn rate by {dimension.replace('_', ' ')}**")
        st.bar_chart(grouped["Churn rate"].sort_values(ascending=False), color="#ef6a6a", use_container_width=True)
    with chart_right:
        st.markdown(f"**Customer volume by {dimension.replace('_', ' ')}**")
        st.bar_chart(grouped["Customers"].sort_values(ascending=False), color="#5b6ee1", use_container_width=True)
    display = grouped.reset_index()
    display["Churn rate"] = display["Churn rate"].map(lambda value: f"{value:.1%}")
    display["Retention rate"] = display["Retention rate"].map(lambda value: f"{value:.1%}")
    display["AverageCharges"] = display["AverageCharges"].map(lambda value: f"${value:,.2f}")
    st.dataframe(display, hide_index=True, use_container_width=True)

    st.markdown('<div class="section-title">Churn drivers</div>', unsafe_allow_html=True)
    driver_left, driver_right = st.columns(2)
    with driver_left:
        st.markdown("**Churn rate by support calls**")
        support = data.assign(call_band=pd.cut(data["support_calls"], [-1, 2, 5, 30], labels=["0–2", "3–5", "6+"]))
        support_rate = support.groupby("call_band", observed=False)["churn"].apply(lambda v: (v == "Yes").mean())
        st.line_chart(support_rate, color="#f0a44b", use_container_width=True)
    with driver_right:
        st.markdown("**Monthly charges and churn**")
        charge_view = data.assign(charge_band=pd.qcut(data["monthly_charges"], 4, duplicates="drop"))
        charge_rate = charge_view.groupby("charge_band", observed=False)["churn"].apply(lambda v: (v == "Yes").mean())
        st.area_chart(charge_rate, color="#8b78d9", use_container_width=True)


def render_customer_intelligence(scored):
    st.markdown('<div class="section-title">Customer intelligence</div>', unsafe_allow_html=True)
    st.caption("Search records, filter risk segments, and open any customer for an individual analysis.")
    filters = st.columns([1.5, 1, 1, 1])
    with filters[0]:
        search = st.text_input("Search customer ID", placeholder="e.g. CUST0042")
    with filters[1]:
        risk_filter = st.multiselect("Risk level", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    with filters[2]:
        contract_filter = st.multiselect("Contract", sorted(scored["contract_type"].unique()), default=sorted(scored["contract_type"].unique()))
    with filters[3]:
        service_filter = st.multiselect("Internet service", sorted(scored["internet_service"].unique()), default=sorted(scored["internet_service"].unique()))

    filtered = scored[
        scored["risk"].isin(risk_filter)
        & scored["contract_type"].isin(contract_filter)
        & scored["internet_service"].isin(service_filter)
    ]
    if search:
        filtered = filtered[filtered["customer_id"].str.contains(search.strip(), case=False, na=False)]
    st.write(f"Showing **{len(filtered):,}** of {len(scored):,} customers")
    table = filtered[["customer_id", "contract_type", "monthly_charges", "tenure_months", "churn_probability", "risk", "churn"]].copy()
    table["churn_probability"] = table["churn_probability"].map(lambda value: f"{value:.1%}")
    st.dataframe(table, hide_index=True, use_container_width=True)

    if filtered.empty:
        st.info("No customers match the selected filters.")
        return
    selected_id = st.selectbox("Analyze customer", filtered["customer_id"].tolist())
    customer = filtered.loc[filtered["customer_id"] == selected_id].iloc[0]
    left, right = st.columns([1, 2])
    with left:
        st.metric("Churn probability", f"{customer['churn_probability']:.1%}")
        st.metric("Risk classification", customer["risk"])
    with right:
        st.markdown(f"**{selected_id} retention plan**")
        for recommendation in retention_recommendations(customer, customer["risk"]):
            st.markdown(f"- {recommendation}")


def render_prediction(model):
    st.markdown('<div class="section-title">Advanced ML prediction</div>', unsafe_allow_html=True)
    st.caption("Generate a probability score, risk classification, and targeted retention recommendations.")
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        tenure_months = st.number_input("Tenure (months)", min_value=1, max_value=120, value=12)
        monthly_charges = st.number_input("Monthly charges", min_value=0.0, value=80.0, step=1.0)
    with col2:
        contract_type = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox("Payment method", ["Credit Card", "Bank Transfer", "Electronic Check"])
        internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        support_calls = st.number_input("Support calls", min_value=0, max_value=30, value=3)
    with col3:
        streaming_service = st.selectbox("Streaming service", ["Yes", "No"])
        senior_citizen = st.selectbox("Senior citizen", [0, 1])
        paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
    total_charges = monthly_charges * tenure_months
    st.info(f"Estimated total charges: ${total_charges:,.2f}")
    if st.button("Calculate churn risk", type="primary", use_container_width=True):
        customer = pd.DataFrame([{
            "gender": gender, "age": age, "tenure_months": tenure_months,
            "monthly_charges": monthly_charges, "total_charges": total_charges,
            "contract_type": contract_type, "payment_method": payment_method,
            "internet_service": internet_service, "support_calls": support_calls,
            "streaming_service": streaming_service, "senior_citizen": senior_citizen,
            "paperless_billing": paperless_billing,
        }])
        probability = float(model.predict_proba(customer)[0][1])
        risk = risk_level(probability)
        st.markdown('<div class="section-title">Prediction result</div>', unsafe_allow_html=True)
        result_left, result_right = st.columns([1, 2])
        with result_left:
            st.metric("Churn probability", f"{probability:.1%}")
            st.metric("Risk classification", risk)
        with result_right:
            if risk == "High":
                st.error("High risk — immediate retention action recommended.")
            elif risk == "Medium":
                st.warning("Medium risk — proactive engagement recommended.")
            else:
                st.success("Low risk — customer is likely to stay.")
            st.progress(probability, text="Model probability")
            for recommendation in retention_recommendations(customer.iloc[0], risk):
                st.markdown(f"- {recommendation}")


model = load_model()
data = load_data()
if model is None:
    st.error("Model not found. Run the training scripts from the project directory.")
    st.stop()
if data is None:
    st.error("Customer dataset not found at data/customer_churn.csv.")
    st.stop()

scored_data = score_customers(model, data)
with st.sidebar:
    st.markdown('<div class="brand"><div style="font-size: 2rem;">📈</div><h1>Churn Intelligence</h1><p>Customer retention command center</p></div>', unsafe_allow_html=True)
    page = st.radio("Navigate", ["Overview", "Churn analytics", "Customer intelligence", "Risk prediction"], label_visibility="collapsed")
    st.divider()
    st.caption("Model status")
    st.success("Ready for predictions")

st.markdown('<div class="hero"><div class="eyebrow">Customer analytics platform</div><h1>Good morning, let\'s reduce churn.</h1><p>Monitor retention, understand churn drivers, and take action on customer risk.</p></div>', unsafe_allow_html=True)
if page == "Overview":
    render_overview(data)
elif page == "Churn analytics":
    render_analytics(data)
elif page == "Customer intelligence":
    render_customer_intelligence(scored_data)
else:
    render_prediction(model)

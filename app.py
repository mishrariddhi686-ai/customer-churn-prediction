
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

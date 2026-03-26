import os
import joblib
import pandas as pd
import streamlit as st

st.title("Customer Churn Prediction")
st.write("Fill the customer details and click predict.")

MODEL_PATH = "C:/Users/bisht/Downloads/customer_churn_prediction/model/churn_model_v1.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

model_package = joblib.load(MODEL_PATH)
pipeline = model_package["pipeline"] if isinstance(model_package, dict) and "pipeline" in model_package else model_package
threshold = model_package.get("threshold", 0.5) if isinstance(model_package, dict) else 0.5

tenure_months = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
num_support_calls = st.number_input("Number of Support Calls", min_value=0, max_value=50, value=1)
service_usage_gb = st.number_input("Service Usage (GB)", min_value=0.0, value=20.0)
num_services = st.number_input("Number of Services", min_value=0, max_value=20, value=3)

paperless_billing = st.radio("Paperless Billing", ["No", "Yes"])
senior_citizen = st.radio("Senior Citizen", ["No", "Yes"])
has_partner = st.radio("Has Partner", ["No", "Yes"])
has_dependents = st.radio("Has Dependents", ["No", "Yes"])

contract_type = st.selectbox("Contract Type", ["month-to-month", "one year", "two year"])
payment_method = st.selectbox("Payment Method", ["auto", "manual"])

mapping = {"No": 0, "Yes": 1}

input_df = pd.DataFrame([{
    "tenure_months": tenure_months,
    "monthly_charges": monthly_charges,
    "num_support_calls": num_support_calls,
    "service_usage_gb": service_usage_gb,
    "num_services": num_services,
    "paperless_billing": mapping[paperless_billing],
    "senior_citizen": mapping[senior_citizen],
    "has_partner": mapping[has_partner],
    "has_dependents": mapping[has_dependents],
    "contract_type": contract_type,
    "payment_method": payment_method,
}])

if st.button("Predict"):
    prob = pipeline.predict_proba(input_df)[0][1]
    pred = int(prob >= threshold)

    if pred == 1:
        st.error(f"Customer is likely to churn. Probability: {prob:.2%}")
    else:
        st.success(f"Customer is likely to stay. Probability: {1 - prob:.2%}")
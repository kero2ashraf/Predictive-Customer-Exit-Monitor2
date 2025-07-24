import streamlit as st
import pandas as pd
import joblib

# --- Load model and preprocessor ---
model_path        = r'C:\Users\kirol\OneDrive - Arab Open University - AOU\Desktop\streamlit_apps\logistic_regression_churn_model.pkl'
preprocessor_path = r'C:\Users\kirol\OneDrive - Arab Open University - AOU\Desktop\streamlit_apps\preprocessor.pkl'
model             = joblib.load(model_path)
preprocessor      = joblib.load(preprocessor_path)

# --- Streamlit UI ---
st.title("🧠 Predictive Customer Exit Monitor")
st.set_page_config(page_title="Predictive Customer Exit Monitor")
st.write("Enter customer details to predict whether they are likely to churn.")

# --- Reset button: clears ALL widget state ---
if st.button("🔄 Reset Inputs"):
    st.session_state.clear()
    st.rerun() # now that session_state is empty, rerun picks up defaults

# --- Inputs in specified order with placeholders + keys ---
gender = st.selectbox(
    "Gender",
    ["Select your gender", "Female", "Male"],
    index=0,
    key="gender"
)
senior = st.selectbox(
    "Senior Citizen",
    ["Select senior‑citizen status", 0, 1],
    index=0,
    key="senior"
)
partner = st.selectbox(
    "Partner",
    ["Select partner status", "Yes", "No"],
    index=0,
    key="partner"
)
dependents = st.selectbox(
    "Dependents",
    ["Select dependents status", "Yes", "No"],
    index=0,
    key="dependents"
)

tenure = st.number_input(
    "Tenure (months)",
    value=0,
    step=1,
    key="tenure"
)

phone = st.selectbox("Phone Service", ["Select Phone Service", "Yes", "No"], index=0, key="phone")
multiline = st.selectbox("Multiple Lines", ["Select Multiple Lines", "Yes", "No"], index=0, key="multiline")
internet = st.selectbox("Internet Service", ["Select Internet Service", "DSL", "Fiber optic", "No"], index=0, key="internet")
online_sec = st.selectbox("Online Security", ["Select Online Security", "Yes", "No"], index=0, key="online_sec")
online_backup = st.selectbox("Online Backup", ["Select Online Backup", "Yes", "No"], index=0, key="online_backup")
device_protect = st.selectbox("Device Protection", ["Select Device Protection", "Yes", "No"], index=0, key="device_protect")
tech_support = st.selectbox("Tech Support", ["Select Tech Support", "Yes", "No"], index=0, key="tech_support")
stream_tv = st.selectbox("Streaming TV", ["Select Streaming TV", "Yes", "No"], index=0, key="stream_tv")
stream_movies = st.selectbox("Streaming Movies", ["Select Streaming Movies", "Yes", "No"], index=0, key="stream_movies")

contract = st.selectbox(
    "Contract",
    ["Select contract term", "Month-to-month", "One year", "Two year"],
    index=0,
    key="contract"
)
paperless = st.selectbox(
    "Paperless Billing",
    ["Select paperless billing", "Yes", "No"],
    index=0,
    key="paperless"
)
payment_method = st.selectbox(
    "Payment Method",
    [
        "Select payment method",
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
    index=0,
    key="payment_method"
)

monthly_charges_str = st.text_input(
    "Monthly Charges ($)",
    value="",
    placeholder="e.g. 45.67",
    key="monthly_charges_str"
)
total_charges_str = st.text_input(
    "Total Charges ($)",
    value="",
    placeholder="e.g. 1234.56",
    key="total_charges_str"
)

# --- Validation of placeholders ---
placeholders = [
    gender, senior, partner, dependents, phone, multiline, internet,
    online_sec, online_backup, device_protect, tech_support, stream_tv,
    stream_movies, contract, paperless, payment_method
]
if any(isinstance(v, str) and v.startswith("Select") for v in placeholders):
    st.error("🚨 Please fill out all dropdowns (don’t leave any “Select …” fields).")
    st.stop()

# Validate numeric strings
try:
    monthly_charges = float(monthly_charges_str)
    total_charges   = float(total_charges_str)
except ValueError:
    st.error("🚨 Please enter valid numbers for Monthly and Total Charges.")
    st.stop()

# --- Predict Button & Logic ---
if st.button("Predict Churn"):
    input_df = pd.DataFrame([{
        'gender':          gender,
        'SeniorCitizen':   senior,
        'Partner':         partner,
        'Dependents':      dependents,
        'tenure':          tenure,
        'PhoneService':    phone,
        'MultipleLines':   multiline,
        'InternetService': internet,
        'OnlineSecurity':  online_sec,
        'OnlineBackup':    online_backup,
        'DeviceProtection':device_protect,
        'TechSupport':     tech_support,
        'StreamingTV':     stream_tv,
        'StreamingMovies': stream_movies,
        'Contract':        contract,
        'PaperlessBilling':paperless,
        'PaymentMethod':   payment_method,
        'MonthlyCharges':  monthly_charges,
        'TotalCharges':    total_charges
    }])

    X_processed = preprocessor.transform(input_df)
    pred_label  = model.predict(X_processed)[0]

    st.markdown("---")
    if pred_label == 1:
        st.error("🚨 **High Risk**: This customer is likely to churn.")
    else:
        st.success("✅ **Low Risk**: This customer is likely to stay.")
    st.markdown("---")

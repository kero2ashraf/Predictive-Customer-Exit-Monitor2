import streamlit as st
import pandas as pd
import joblib
import os

# --- Load model and preprocessor from RELATIVE PATH ---
MODEL_FILENAME = r"C:\streamlit_apps\logistic_regression_churn_model.pkl"
PREPROCESSOR_FILENAME = r"C:\streamlit_apps\preprocessor.pkl"

model = joblib.load(open(MODEL_FILENAME, 'rb'))
preprocessor = joblib.load(open(PREPROCESSOR_FILENAME, 'rb'))

# --- Streamlit UI ---
st.set_page_config(page_title="Predictive Customer Exit Monitor")
st.title("🧠 Predictive Customer Exit Monitor")
st.write("Enter customer details to predict whether they are likely to churn.")

# --- Reset button ---
if st.button("🔄 Reset Inputs"):
    st.session_state.clear()
    st.rerun()

# --- Inputs ---
gender = st.selectbox("Gender", ["Select", "Female", "Male"], index=0)
senior = st.selectbox("Senior Citizen", ["Select", 0, 1], index=0)
partner = st.selectbox("Partner", ["Select", "Yes", "No"], index=0)
dependents = st.selectbox("Dependents", ["Select", "Yes", "No"], index=0)
tenure = st.number_input("Tenure (months)", value=0, step=1)
phone = st.selectbox("Phone Service", ["Select", "Yes", "No"], index=0)
multiline = st.selectbox("Multiple Lines", ["Select", "Yes", "No"], index=0)
internet = st.selectbox("Internet Service", ["Select", "DSL", "Fiber optic", "No"], index=0)
online_sec = st.selectbox("Online Security", ["Select", "Yes", "No"], index=0)
online_backup = st.selectbox("Online Backup", ["Select", "Yes", "No"], index=0)
device_protect = st.selectbox("Device Protection", ["Select", "Yes", "No"], index=0)
tech_support = st.selectbox("Tech Support", ["Select", "Yes", "No"], index=0)
stream_tv = st.selectbox("Streaming TV", ["Select", "Yes", "No"], index=0)
stream_movies = st.selectbox("Streaming Movies", ["Select", "Yes", "No"], index=0)
contract = st.selectbox("Contract", ["Select", "Month-to-month", "One year", "Two year"], index=0)
paperless = st.selectbox("Paperless Billing", ["Select", "Yes", "No"], index=0)
payment_method = st.selectbox("Payment Method", [
    "Select", "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
], index=0)

monthly_charges_str = st.text_input("Monthly Charges ($)", placeholder="e.g. 45.67")
total_charges_str = st.text_input("Total Charges ($)", placeholder="e.g. 1234.56")

# --- Validations ---
dropdowns = [
    gender, senior, partner, dependents, phone, multiline, internet,
    online_sec, online_backup, device_protect, tech_support, stream_tv,
    stream_movies, contract, paperless, payment_method
]

if any(option == "Select" for option in dropdowns):
    st.warning("🚨 Please select an option for all dropdowns.")
    st.stop()

try:
    monthly_charges = float(monthly_charges_str)
    total_charges = float(total_charges_str)
except ValueError:
    st.error("🚨 Please enter valid numbers for Monthly and Total Charges.")
    st.stop()

# --- Predict ---
if st.button("Predict Churn"):
    input_df = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone,
        'MultipleLines': multiline,
        'InternetService': internet,
        'OnlineSecurity': online_sec,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protect,
        'TechSupport': tech_support,
        'StreamingTV': stream_tv,
        'StreamingMovies': stream_movies,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }])

    # Preprocess and Predict
    X_processed = preprocessor.transform(input_df)
    prediction = model.predict(X_processed)[0]

    # Show result
    st.markdown("---")
    if prediction == 1:
        st.error("🚨 High Risk: This customer is likely to churn.")
    else:
        st.success("✅ Low Risk: This customer is likely to stay.")
    st.markdown("---")

"""
app.py
------
Streamlit dashboard for the Loan Approval Prediction project.
Designed to be simple enough to demo live in front of a non-technical
audience: fill in the form on the left, click Predict, see the result.

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Loan Approval Predictor | Nirmaan AI Training",
    page_icon="🏦",
    layout="centered",
)

# ---------------------------------------------------------------------
# LOAD MODEL + SUPPORTING FILES (cached so it only loads once)
# ---------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("loan_model.pkl")
    encoders = joblib.load("label_encoders.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    meta = joblib.load("model_meta.pkl")
    return model, encoders, feature_cols, meta

model, encoders, feature_cols, meta = load_artifacts()

# ---------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------
st.title("🏦 Loan Approval Predictor")
st.caption(
    f"Built by Nirmaan students · Model: **{meta['model_name']}** "
    f"· Test Accuracy: **{meta['accuracy']*100:.1f}%**"
)
st.markdown(
    "Fill in an applicant's details below and get an **instant prediction** "
    "on whether their loan would likely be approved."
)
st.divider()

# ---------------------------------------------------------------------
# INPUT FORM
# ---------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

with col2:
    applicant_income = st.number_input("Applicant Monthly Income (₹)", min_value=0, value=5000, step=500)
    coapplicant_income = st.number_input("Co-applicant Monthly Income (₹)", min_value=0, value=0, step=500)
    loan_amount = st.number_input("Loan Amount (in thousands ₹)", min_value=0, value=120, step=10)
    loan_term = st.selectbox("Loan Term (days)", [360, 180, 120, 84, 60, 36, 12])
    credit_history = st.selectbox("Has Good Credit History?", ["Yes", "No"])

st.divider()

# ---------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------
if st.button("🔍 Predict Loan Approval", use_container_width=True, type="primary"):

    # Build a single-row dataframe matching training-time preprocessing
    raw_input = {
        "Gender": gender,
        "Married": married,
        "Dependents": 3 if dependents == "3+" else int(dependents),
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": 1 if credit_history == "Yes" else 0,
        "Property_Area": property_area,
    }

    row = pd.DataFrame([raw_input])

    # Apply the SAME label encoders used during training
    for col in ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]:
        row[col] = row[col].astype(str)  # match the string encoding used in train.py
        row[col] = encoders[col].transform(row[col])

    # Ensure column order matches training
    row = row[feature_cols]

    prediction = model.predict(row)[0]
    probability = model.predict_proba(row)[0][1]  # probability of "Approved"

    st.subheader("Result")
    if prediction == 1:
        st.success(f"✅ Loan likely **APPROVED** (confidence: {probability*100:.1f}%)")
    else:
        st.error(f"❌ Loan likely **REJECTED** (confidence: {(1-probability)*100:.1f}%)")

    st.progress(float(probability))
    st.caption("This is a prediction from a trained ML model, not a bank decision.")

st.divider()
st.markdown(
    "<small>Project by Redington Nirmaan AI/Data Science trainees · "
    "Trained on the Kaggle Loan Prediction dataset</small>",
    unsafe_allow_html=True,
)

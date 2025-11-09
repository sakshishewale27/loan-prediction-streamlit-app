import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ===============================
# Load model and preprocessors
# ===============================
model = joblib.load("loan_model.pkl")
preprocessors = joblib.load("preprocessors.pkl")

scaler = preprocessors["scaler"]
label_encoders = preprocessors["label_encoders"]
feature_names = preprocessors["feature_names"]

# ===============================
# Feature Engineering Function
# (same as in Colab)
# ===============================
def engineer_features(df):
    df = df.copy()

    # Financial ratios
    df["Monthly_Income"] = df["Income"] / 12
    df["EMI_to_Income_Ratio"] = (df["Loan_Amount"] / df["Loan_Tenure_Months"]) / df["Monthly_Income"]
    df["Loan_to_Income_Ratio"] = df["Loan_Amount"] / df["Income"]

    # Credit behavior
    df["Credit_Utilization"] = df["Existing_Loans"] * 50000 / df["Income"]
    df["Avg_Loan_per_Account"] = df["Existing_Loans"] / (df["Num_Bank_Accounts"] + 1)

    # Stability indicators
    df["Job_Stability_Score"] = df["Employment_Years"] / (df["Age"] - 18 + 1)
    df["Financial_Maturity"] = df["Bank_Account_Age_Years"] * df["Credit_Score"] / 1000

    # Risk flag
    df["High_Risk_Flag"] = (
        (df["Previous_Defaults"] > 0)
        | (df["Payment_Delay_Days"] > 30)
        | (df["Debt_to_Income_Ratio"] > 0.5)
    ).astype(int)

    # Age groups
    df["Age_Group"] = pd.cut(df["Age"], bins=[0, 30, 45, 60, 100],
                             labels=["Young", "Middle", "Senior", "Retired"])

    # Income groups
    df["Income_Group"] = pd.cut(df["Income"], bins=[0, 30000, 60000, 100000, np.inf],
                                labels=["Low", "Medium", "High", "Very_High"])

    return df

# ===============================
# Streamlit App UI
# ===============================
st.set_page_config(page_title="Loan Approval Predictor", page_icon="💰", layout="wide")
st.title("💰 Loan Approval Prediction App")
st.markdown("Predict whether a loan will be approved based on applicant details.")

st.subheader("Enter Applicant Details:")

col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", 21, 70, 35)
    Income = st.number_input("Annual Income (₹)", 10000, 2000000, 60000)
    Employment_Years = st.number_input("Years of Employment", 0, 40, 5)
    Job_Type = st.selectbox("Job Type", ["Salaried", "Self-Employed", "Business"])
    Credit_Score = st.number_input("Credit Score", 300, 850, 720)
    Existing_Loans = st.number_input("Existing Loans", 0, 10, 1)
    Loan_Amount = st.number_input("Loan Amount (₹)", 5000, 1000000, 50000)

with col2:
    Loan_Tenure_Months = st.selectbox("Loan Tenure (Months)", [12, 24, 36, 48, 60, 84, 120, 180, 240, 360])
    Debt_to_Income_Ratio = st.slider("Debt to Income Ratio", 0.0, 1.0, 0.3)
    Previous_Defaults = st.selectbox("Previous Defaults", [0, 1, 2, 3])
    Payment_Delay_Days = st.number_input("Payment Delay Days", 0, 100, 5)
    Bank_Account_Age_Years = st.number_input("Bank Account Age (Years)", 1, 40, 10)
    Num_Bank_Accounts = st.number_input("Number of Bank Accounts", 1, 10, 2)

# ===============================
# Prediction
# ===============================
if st.button("🔍 Predict Loan Approval"):
    # Step 1: Create DataFrame
    input_data = {
        "Age": Age,
        "Income": Income,
        "Employment_Years": Employment_Years,
        "Job_Type": Job_Type,
        "Credit_Score": Credit_Score,
        "Existing_Loans": Existing_Loans,
        "Loan_Amount": Loan_Amount,
        "Loan_Tenure_Months": Loan_Tenure_Months,
        "Debt_to_Income_Ratio": Debt_to_Income_Ratio,
        "Previous_Defaults": Previous_Defaults,
        "Payment_Delay_Days": Payment_Delay_Days,
        "Bank_Account_Age_Years": Bank_Account_Age_Years,
        "Num_Bank_Accounts": Num_Bank_Accounts,
    }

    df = pd.DataFrame([input_data])

    # Step 2: Feature Engineering (same as training)
    df = engineer_features(df)

    # Step 3: Label Encoding
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))

    # Step 4: Ensure all trained features exist
    missing_cols = set(feature_names) - set(df.columns)
    for col in missing_cols:
        df[col] = 0  # fill missing columns if any

    df = df[feature_names]  # reorder columns

    # Step 5: Scaling
    df_scaled = scaler.transform(df)

    # Step 6: Prediction
    pred = model.predict(df_scaled)[0]
    proba = model.predict_proba(df_scaled)[0][1]

    # Step 7: Display
    st.subheader("📊 Prediction Result:")
    if pred == 1:
        st.success(f"✅ Loan Approved! (Probability: {proba*100:.2f}%)")
    else:
        st.error(f"❌ Loan Rejected! (Approval Probability: {proba*100:.2f}%)")

    st.caption("Model trained with engineered financial, behavioral, and credit-based features.")

import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("🏦 Customer Churn Prediction App")

st.write("Enter customer details to predict churn")

# =========================
# USER INPUTS
# =========================

credit_score = st.number_input("Credit Score", 300, 900, 600)
age = st.slider("Age", 18, 100, 30)
tenure = st.slider("Tenure (Years)", 0, 10, 3)
balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)
num_products = st.slider("Number of Products", 1, 4, 1)

has_card = st.selectbox("Has Credit Card", [0, 1])
is_active = st.selectbox("Is Active Member", [0, 1])

salary = st.number_input("Estimated Salary", 10000.0, 200000.0, 50000.0)

geography = st.selectbox("Geography", ["Germany", "Spain", "France"])
gender = st.selectbox("Gender", ["Male", "Female"])

# =========================
# FEATURE ENGINEERING (same as training)
# =========================

balance_per_product = balance / (num_products + 1)
activity_score = is_active * num_products

# =========================
# ENCODING (manual)
# =========================

geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0

gender_male = 1 if gender == "Male" else 0

# =========================
# FINAL INPUT ARRAY
# =========================

features = np.array([[credit_score, age, tenure, balance,
                      num_products, has_card, is_active,
                      salary, balance_per_product,
                      activity_score,
                      geo_germany, geo_spain,
                      gender_male]])

# =========================
# PREDICTION
# =========================

if st.button("Predict"):
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ Customer is likely to CHURN\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Customer is likely to STAY\nProbability: {probability:.2f}")
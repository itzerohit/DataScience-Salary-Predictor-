import streamlit as st
import numpy as np
import pickle

# Load model and scaler
with open("gridsvr.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# --- Page Config ---
st.set_page_config(page_title="💼 DS Salary Predictor", layout="centered")

# --- Header ---
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>💼 Data Science Salary Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>🔍 Enter job details and get an estimated salary prediction based on your role and company details.</p>", unsafe_allow_html=True)
st.write("")

# --- Inputs with Descriptions ---
st.subheader("📌 Job & Company Details")

col1, col2 = st.columns(2)

with col1:
    experience_level = st.selectbox("🧠 Experience Level", options=[
        ("Entry-level (0)", 0),
        ("Mid-level (1)", 1),
        ("Senior (2)", 2),
        ("Executive (3)", 3)
    ], help="Choose your level of experience.")
    exp_val = experience_level[1]

    company_location = st.number_input("🌍 Company Location Code", min_value=0, step=1, help="This is label encoded. Eg: US=0, IN=1...")

    employee_residence = st.number_input("🏠 Employee Residence Code", min_value=0, step=1, help="Your working/residing country code.")

with col2:
    employment_type = st.selectbox("📃 Employment Type", options=[
        ("Full-Time (0)", 0),
        ("Part-Time (1)", 1),
        ("Contract (2)", 2),
        ("Freelance (3)", 3)
    ], help="Nature of employment contract.")
    emp_type_val = employment_type[1]

    company_size = st.selectbox("🏢 Company Size", options=[
        ("Small (0)", 0),
        ("Medium (1)", 1),
        ("Large (2)", 2)
    ], help="Size category of the company.")
    company_size_val = company_size[1]

# --- Prediction ---
st.markdown("---")
st.subheader("🎯 Salary Estimation")

input_data = np.array([[exp_val, emp_type_val, company_location, company_size_val, employee_residence]])
scaled_input = scaler.transform(input_data)

if st.button("💰 Predict Salary"):
    prediction = model.predict(scaled_input)[0]
    st.success(f"💲 Estimated Salary: **${prediction:,.2f}**")
    st.balloons()
else:
    st.info("Click the button to generate your salary prediction.")

# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>✨ Powered by Support Vector Regression | Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)

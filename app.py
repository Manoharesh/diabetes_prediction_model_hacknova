import streamlit as st
import pandas as pd
import joblib
from datetime import date, timedelta
import numpy as np
import plotly.graph_objects as go

# Load the trained model
model = joblib.load('diabetes_dataset_model.pkl')

# Page Configuration
st.set_page_config(page_title="Pediatric Diabetes Calculator", layout="centered")

# Custom Styling
st.markdown("""
    <style>
    h1, h2, h3, h4, h5, h6 {
        display: inline;
    }
    .css-1v0mbdj a {
        display: none !important;
    }
    [data-testid="stSidebar"] {
        background-color: #1e1e2f;
        padding: 20px;
        border-right: 2px solid #4CAF50;
    }
    [data-testid="stSidebar"] h2 {
        color: #ffffff;
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 18px;
        color: #e0e0e0;
        padding: 8px 12px;
        border-radius: 8px;
        display: flex;
        align-items: center;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"] {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background-color: #2e2e3f;
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
page = st.sidebar.radio("Navigation", ["Home", "About"])

# Original Gauge (No Glowing Needle)
def create_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Diabetes Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red" if probability > 0.6 else "orange" if probability > 0.3 else "green"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 100], 'color': "salmon"}
            ]
        }
    ))
    fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
    return fig

# ----------------------------
# Home Page
# ----------------------------
if page == "Home":
    # Title
    st.markdown(
        """
        <h1 style='
            text-align: center;
            color: #4CAF50;
            font-size: 45px;
            font-weight: bold;
            margin-top: -20px;
            margin-bottom: 10px;
        '>
            Pediatric Diabetes Risk Calculator
        </h1>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<p style='text-align:center;'>This tool calculates the risk of diabetes in children based on key health metrics.</p>", unsafe_allow_html=True)
    st.caption("**Note:** This calculator is designed **only for children** and **not intended for adults.**")

    # Form Section
    with st.form("risk_form"):
        st.subheader("Enter Patient Details:")

        today = date.today()
        max_dob = today - timedelta(days=5 * 365)   # Max age = 5 years
        min_dob = today - timedelta(days=18 * 365)  # Min age = 18 years

        dob = st.date_input("Date of Birth", min_value=min_dob, max_value=max_dob)
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

        height = st.number_input("Height (cm)", min_value=30, max_value=200, value=120, step=1)
        weight = st.number_input("Weight (kg)", min_value=5, max_value=150, value=30, step=1)
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, value=90, step=1)
        insulin = st.number_input("Insulin Level (μU/mL)", min_value=0, value=20, step=1)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=50, max_value=200, value=90, step=1)

        sex = st.selectbox("Sex", options=["Male", "Female"])
        activity = st.selectbox("Activity Level", options=["Sedentary", "Moderate", "Active"])
        family_history = st.selectbox("Family History of Diabetes", options=["Yes", "No"])

        submitted = st.form_submit_button("🔍 Predict Risk")

    # Prediction Section
    if submitted:
        bmi = round(weight / ((height / 100) ** 2), 2)

        sex_encoded = 1 if sex == "Male" else 0
        activity_map = {"Sedentary": 0, "Moderate": 1, "Active": 2}
        activity_encoded = activity_map[activity]
        family_history_encoded = 1 if family_history == "Yes" else 0

        input_data = pd.DataFrame([[
            age, sex_encoded, bmi, glucose, insulin,
            blood_pressure, activity_encoded, family_history_encoded
        ]], columns=[
            "Age", "Sex", "BMI", "Glucose", "Insulin",
            "BloodPressure", "PhysicalActivityLevel", "FamilyHistory"
        ])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("📊 Prediction Result:")
        st.write(f"**Age:** {age} years")
        st.write(f"**BMI:** {bmi}")
        st.success("⚠️ At Risk of Diabetes" if prediction == 1 else "✅ Not at Risk")

        # Speedometer Gauge
        st.plotly_chart(create_gauge(probability), use_container_width=True)

        st.markdown("---")
        st.subheader("💡 Health Suggestions")
        if bmi > 22:
            st.warning("BMI is high for age. Consider weight management.")
        if glucose > 110:
            st.warning("Glucose is elevated. Consider dietary monitoring.")
        if insulin > 25:
            st.warning("Insulin is elevated. Consult a pediatrician.")
        if blood_pressure > 120:
            st.warning("High blood pressure. Needs monitoring.")
        if family_history_encoded:
            st.info("Family history increases risk. Take preventive measures.")

        # ----------------------------
        # Single Metric What-If Analysis
        # ----------------------------
        if prediction == 1:
            # Create a copy of the data
            what_if_data = input_data.copy()

            # Store changes for BMI, Glucose, Insulin
            changes = {}

            if bmi > 22:
                what_if_data["BMI"] = 20  # Healthy BMI
                new_prob_bmi = model.predict_proba(what_if_data)[0][1]
                changes["BMI"] = probability - new_prob_bmi
                what_if_data["BMI"] = bmi  # Reset

            if glucose > 110:
                what_if_data["Glucose"] = 100  # Healthy glucose
                new_prob_glucose = model.predict_proba(what_if_data)[0][1]
                changes["Glucose"] = probability - new_prob_glucose
                what_if_data["Glucose"] = glucose  # Reset

            if insulin > 25:
                what_if_data["Insulin"] = 20  # Healthy insulin
                new_prob_insulin = model.predict_proba(what_if_data)[0][1]
                changes["Insulin"] = probability - new_prob_insulin
                what_if_data["Insulin"] = insulin  # Reset

            if changes:
                # Pick the factor with the maximum drop in risk
                best_factor = max(changes, key=changes.get)
                drop = changes[best_factor] * 100
                st.info(f"⚡ If **{best_factor}** was regulated to healthy levels, the risk would drop by **{drop:.1f}%**.")

# ----------------------------
# About Page
# ----------------------------
elif page == "About":
    st.markdown("<h2 style='text-align:center; color:#2196F3;'>About This Project</h2>", unsafe_allow_html=True)
    st.write("""
    **Pediatric Diabetes Risk Calculator** is a predictive tool built using machine learning
    to assess the risk of diabetes in children based on health metrics like BMI, glucose, insulin, and blood pressure.

    **Key Features:**
    - Uses a trained ML model for accurate predictions.
    - Provides probability-based risk analysis.
    - Offers actionable health suggestions.

    **Disclaimer:** This tool is intended for educational and informational purposes only.  
    Consult a healthcare professional for medical advice.

    **Developed by:** Health Hackers    
    **For:** HackNova 2025
    """)
    st.markdown("---")

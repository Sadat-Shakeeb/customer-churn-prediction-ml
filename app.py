# =========================
# Imports & Configuration
# =========================
import os
import warnings

import streamlit as st
import joblib
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

# Ignore sklearn version mismatch warning (safe for demo)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# -------------------------
# Streamlit Page Settings
# -------------------------
st.set_page_config(
    page_title="ChurnShield AI",
    page_icon="🛡️",
    layout="wide"
)

# -------------------------
# Custom Styling
# -------------------------
st.markdown(
    """
    <style>
        .main { background-color: #f5f7f9; }
        .stButton>button {
            width: 100%;
            height: 3em;
            border-radius: 6px;
            background-color: #007bff;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Model & Scaler Loading
# =========================
BASE_PATH = os.path.dirname(__file__)


@st.cache_resource
def load_model_assets():
    """Load trained model and scaler from disk."""
    try:
        model = joblib.load(os.path.join(BASE_PATH, "churn_model.pkl"))
        scaler = joblib.load(os.path.join(BASE_PATH, "scaler.pkl"))
        return model, scaler
    except Exception:
        return None, None


model, scaler = load_model_assets()

# =========================
# App Header
# =========================
st.title("🛡️ ChurnShield: Telco Retention AI")
st.info(
    "This application analyzes customer behavior to predict churn risk "
    "and support proactive retention strategies."
)

# =========================
# UI: Input Form
# =========================
if model and scaler:

    st.subheader("📋 Customer Profile")
    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.number_input("Tenure (Months)", 0, 72, 12)
        senior_citizen = st.selectbox("Senior Citizen?", ["No", "Yes"])

    with col2:
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, value=70.0)
        contract = st.selectbox(
            "Contract Type", ["Month-to-month", "One year", "Two year"]
        )

    with col3:
        avg_charges = st.number_input("Avg Lifetime Charges ($)", 0.0, value=60.0)
        internet_service = st.selectbox(
            "Internet Service", ["Fiber optic", "DSL", "No"]
        )

    st.divider()
    st.subheader("🛠️ Services & Billing")
    col4, col5, col6 = st.columns(3)

    with col4:
        multiple_lines = st.selectbox(
            "Multiple Lines?", ["No", "Yes", "No phone service"]
        )
        paperless_billing = st.selectbox(
            "Paperless Billing?", ["Yes", "No"]
        )

    with col5:
        streaming_tv = st.selectbox(
            "Streaming TV?", ["No", "Yes", "No internet service"]
        )
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    with col6:
        streaming_movies = st.selectbox(
            "Streaming Movies?", ["No", "Yes", "No internet service"]
        )

    # =========================
    # Prediction Logic
    # =========================
    if st.button("🚀 Analyze Churn Risk"):

        # Manual feature encoding (must match training pipeline)
        input_data = {
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "AvgCharges": avg_charges,

            "InternetService_Fiber optic": 1 if internet_service == "Fiber optic" else 0,
            "InternetService_No": 1 if internet_service == "No" else 0,

            "Contract_One year": 1 if contract == "One year" else 0,
            "Contract_Two year": 1 if contract == "Two year" else 0,

            "PaperlessBilling_Yes": 1 if paperless_billing == "Yes" else 0,

            "PaymentMethod_Credit card (automatic)": 1 if payment_method == "Credit card (automatic)" else 0,
            "PaymentMethod_Electronic check": 1 if payment_method == "Electronic check" else 0,
            "PaymentMethod_Mailed check": 1 if payment_method == "Mailed check" else 0,

            "StreamingTV_Yes": 1 if streaming_tv == "Yes" else 0,
            "StreamingTV_No internet service": 1 if streaming_tv == "No internet service" else 0,

            "StreamingMovies_Yes": 1 if streaming_movies == "Yes" else 0,
            "StreamingMovies_No internet service": 1 if streaming_movies == "No internet service" else 0,

            "MultipleLines_Yes": 1 if multiple_lines == "Yes" else 0,
            "MultipleLines_No phone service": 1 if multiple_lines == "No phone service" else 0,
        }

        # Align features with training schema
        input_df = pd.DataFrame([input_data])
        expected_columns = scaler.feature_names_in_

        for col in expected_columns:
            if col not in input_df:
                input_df[col] = 0

        input_df = input_df[expected_columns]

        # Scale + Predict
        scaled_input = scaler.transform(input_df)
        churn_probability = model.predict_proba(scaled_input)[0][1]

        # =========================
        # Results & Interpretation
        # =========================
        st.subheader("📊 Churn Risk Analysis")
        st.progress(churn_probability)

        if churn_probability > 0.60:
            st.error(f"🚨 **High Churn Risk:** {churn_probability:.1%}")
            st.write("**Recommended Action:** Immediate retention offer or concierge outreach.")
        elif churn_probability > 0.30:
            st.warning(f"⚠️ **Medium Churn Risk:** {churn_probability:.1%}")
            st.write("**Recommended Action:** Encourage long-term contract or bundled services.")
        else:
            st.success(f"✅ **Low Churn Risk:** {churn_probability:.1%}")
            st.write("**Recommended Action:** Maintain standard engagement strategy.")

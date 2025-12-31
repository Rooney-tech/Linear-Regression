# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Set page config
st.set_page_config(page_title="Healthcare Billing Predictor", layout="wide")

# Title
st.title("🏥 Healthcare Billing Amount Predictor")

# --- Model Loading (with portable path fallback) ---
@st.cache_resource
def load_model():
    # Try relative path first (for GitHub/Streamlit Cloud)
    model_path = Path(__file__).parent / "Linear_regression.joblib"
    if not model_path.exists():
        # Fallback to your local absolute path (only works on your machine)
        model_path = Path(r"C:\Users\DELL\OneDrive\Python Programming\Machine Learning\Libraries\Machine Learning\Libraries\Streamlit\Linear Regression\Linear-Regression\Linear regression\Notebooks\Linear_regression.joblib")
    if not model_path.exists():
        st.error("❌ Model 'Linear_regression.joblib' not found in expected locations.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# --- Tab UI: Single vs Batch Prediction ---
tab1, tab2 = st.tabs(["🔹 Single Patient", "Batch Upload (CSV)"])

# ======================
# TAB 1: Single Prediction
# ======================
with tab1:
    st.subheader("Enter Patient Details")
    
    with st.form("single_patient_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=45)
            medical_condition = st.selectbox(
                "Medical Condition",
                [
                    "Cancer", "Diabetes", "Heart Disease", "Hypertension", "Asthma",
                    "Arthritis", "Stroke", "COPD", "Migraine", "Obesity",
                    "Anxiety", "Depression"
                ],
                index=0
            )
        
        with col2:
            admission_type = st.selectbox(
                "Admission Type",
                ["Emergency", "Urgent", "Elective", "Outpatient"],
                index=0
            )
        
        submitted = st.form_submit_button(" Predict Billing Amount")
    
    if submitted:
        # Create input DataFrame (must match training schema)
        input_df = pd.DataFrame([{
            "Age": age,
            "Medical Condition": medical_condition,
            "Admission Type": admission_type
        }])
        
        try:
            log_pred = model.predict(input_df)
            pred_amount = np.expm1(log_pred[0])  # scalar
            
            st.success(f"###  Predicted Billing Amount: **${pred_amount:,.2f}**")
            
            # Optional: show input
            with st.expander(" Submitted Details"):
                st.write(input_df.iloc[0].to_dict())
                
        except Exception as e:
            st.error("Prediction failed. Check input or model compatibility.")
            st.exception(e)

# ======================
# TAB 2: Batch Upload (Your Original Logic)
# ======================
with tab2:
    st.markdown("""
    Upload a CSV file with columns:  
    `Age`, `Medical Condition`, `Admission Type`  
    *(Include `Billing Amount` to see prediction errors)*
    """)
    
    uploaded_file = st.file_uploader("📤 Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            new_data = pd.read_csv(uploaded_file)
            st.subheader("Preview")
            st.dataframe(new_data.head(), use_container_width=True)

            required_cols = {'Age', 'Medical Condition', 'Admission Type'}
            missing_cols = required_cols - set(new_data.columns)
            if missing_cols:
                st.error(f"Missing columns: {missing_cols}")
                st.stop()

            log_preds = model.predict(new_data)
            predictions = np.expm1(log_preds)

            results = new_data.copy()
            results['Predicted Billing Amount'] = predictions

            if 'Billing Amount' in new_data.columns:
                results['Absolute Error'] = np.abs(results['Billing Amount'] - predictions)
                mae = results['Absolute Error'].mean()
                st.success(f"✅ Batch prediction complete! MAE: ${mae:,.2f}")

            st.subheader("Results")
            st.dataframe(results, use_container_width=True)

            csv = results.to_csv(index=False)
            st.download_button(
                "📥 Download Predictions",
                csv,
                "billing_predictions.csv",
                "text/csv"
            )

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)
    else:
        st.info("👆 Upload a CSV file to predict for multiple patients.")
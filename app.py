import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "models" / "house_price_model.joblib"
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")
st.title("🏠 House Price Prediction")
st.caption("Prediction using the supplied housing dataset.")

with st.form("prediction_form"):
    area = st.number_input("Area", min_value=100, max_value=10000, value=2000, step=50)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
    floors = st.number_input("Floors", min_value=1, max_value=10, value=2, step=1)
    year_built = st.number_input("Year Built", min_value=1800, max_value=2030, value=2000, step=1)
    location = st.selectbox("Location", ["Downtown", "Suburban", "Urban", "Rural"])
    condition = st.selectbox("Condition", ["Excellent", "Good", "Fair", "Poor"])
    garage = st.selectbox("Garage", ["Yes", "No"])
    submitted = st.form_submit_button("Predict Price")

if submitted:
    row = pd.DataFrame([{
        "Area": area, "Bedrooms": bedrooms, "Bathrooms": bathrooms,
        "Floors": floors, "YearBuilt": year_built, "Location": location,
        "Condition": condition, "Garage": garage
    }])
    prediction = float(model.predict(row)[0])
    st.success(f"Estimated price: ${prediction:,.0f}")

st.divider()
st.warning(
    "Model validation found very weak predictive signal in the supplied data. "
    "Treat the estimate as a demonstration, not a real-world valuation."
)

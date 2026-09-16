from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "house_price_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered",
)

st.title("🏠 House Price Predictor")
st.caption("Estimate a house price using the features supplied in the training dataset.")

if not MODEL_PATH.exists():
    st.error("The trained model has not been generated yet.")
    st.code("python src/train.py", language="powershell")
    st.stop()

model = joblib.load(MODEL_PATH)

st.info(
    "This is a machine-learning demonstration built from a synthetic/randomly generated "
    "dataset. The estimate should not be treated as a real property valuation."
)

with st.expander("📏 What does Area mean?", expanded=True):
    st.markdown(
        "**Area is the floor area of the house measured in square feet (sq ft).** "
        "For example, enter **2,000** for a house with 2,000 square feet of floor space. "
        "The supplied dataset contains values from **500 to 5,000 sq ft**."
    )
    st.caption(
        "Enter the total house floor area represented by the dataset—not the number of bedrooms "
        "and not the land size unless your source specifically defines it that way."
    )

with st.expander("ℹ️ How to use this form"):
    st.markdown(
        "1. Enter the house's known characteristics.\n"
        "2. Keep values within the dataset's ranges where possible.\n"
        "3. Click **Predict House Price**.\n"
        "4. The result is an estimated price in **US dollars**, matching the supplied dataset."
    )

st.subheader("House details")

with st.form("prediction_form"):
    area = st.number_input(
        "Area (square feet)",
        min_value=500,
        max_value=5000,
        value=2000,
        step=50,
        help="House floor area in square feet. Dataset range: 500–5,000 sq ft.",
    )

    col1, col2 = st.columns(2)
    with col1:
        bedrooms = st.number_input(
            "Bedrooms",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
            help="Number of bedrooms. Dataset range: 1–5.",
        )
        floors = st.number_input(
            "Floors",
            min_value=1,
            max_value=3,
            value=2,
            step=1,
            help="Number of floors. Dataset range: 1–3.",
        )
        location = st.selectbox(
            "Location",
            ["Downtown", "Suburban", "Urban", "Rural"],
            help="Location category used by the dataset.",
        )

    with col2:
        bathrooms = st.number_input(
            "Bathrooms",
            min_value=1,
            max_value=4,
            value=2,
            step=1,
            help="Number of bathrooms. Dataset range: 1–4.",
        )
        year_built = st.number_input(
            "Year Built",
            min_value=1900,
            max_value=2023,
            value=2000,
            step=1,
            help="Year the house was built. Dataset range: 1900–2023.",
        )
        garage = st.selectbox(
            "Garage",
            ["Yes", "No"],
            help="Whether the property has a garage.",
        )

    condition = st.selectbox(
        "House Condition",
        ["Excellent", "Good", "Fair", "Poor"],
        help="Current condition category used by the dataset.",
    )

    submitted = st.form_submit_button("🔮 Predict House Price", use_container_width=True)

if submitted:
    row = pd.DataFrame(
        [
            {
                "Area": area,
                "Bedrooms": bedrooms,
                "Bathrooms": bathrooms,
                "Floors": floors,
                "YearBuilt": year_built,
                "Location": location,
                "Condition": condition,
                "Garage": garage,
            }
        ]
    )

    prediction = float(model.predict(row)[0])

    st.success(f"Estimated house price: **${prediction:,.0f}**")

    st.subheader("Your input")
    display_row = row.rename(
        columns={
            "Area": "Area (sq ft)",
            "YearBuilt": "Year Built",
        }
    ).T
    display_row.columns = ["Value"]
    st.dataframe(display_row, use_container_width=True)

    if METRICS_PATH.exists():
        metrics = pd.read_json(METRICS_PATH)
        # Keep the app's main result simple; detailed metrics remain in the project files.
        _ = metrics

    st.warning(
        "Validation of this supplied dataset found very weak relationships between the input "
        "features and price. The current trained model selected a baseline because it performed "
        "better than the tested ML models on the validation split."
    )

st.divider()
st.caption(
    "Dataset: 2,000 synthetic records. Area is measured in square feet. "
    "Price values are represented in US dollars in the supplied dataset documentation."
)

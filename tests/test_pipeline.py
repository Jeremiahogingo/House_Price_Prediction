from pathlib import Path
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]

def test_dataset_schema():
    df = pd.read_csv(ROOT / "data" / "houses.csv")
    assert df.shape == (2000, 10)
    assert list(df.columns) == ["Id","Area","Bedrooms","Bathrooms","Floors","YearBuilt","Location","Condition","Garage","Price"]
    assert df.isna().sum().sum() == 0

def test_model_predicts_positive_number():
    model = joblib.load(ROOT / "models" / "house_price_model.joblib")
    row = pd.DataFrame([{
        "Area": 2000, "Bedrooms": 3, "Bathrooms": 2, "Floors": 2,
        "YearBuilt": 2000, "Location": "Suburban", "Condition": "Good", "Garage": "Yes"
    }])
    prediction = float(model.predict(row)[0])
    assert prediction > 0

from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "houses.csv"
MODEL = ROOT / "models" / "house_price_model.joblib"
TRAINING_LOG = ROOT / "training_output.txt"

EXPECTED_COLUMNS = [
    "Id",
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Floors",
    "YearBuilt",
    "Location",
    "Condition",
    "Garage",
    "Price",
]


def test_dataset_schema():
    df = pd.read_csv(DATA)
    assert df.shape == (2000, 10)
    assert list(df.columns) == EXPECTED_COLUMNS
    assert df.isna().sum().sum() == 0


def test_area_is_square_feet_range_from_documentation():
    df = pd.read_csv(DATA)
    assert df["Area"].between(500, 5000).all()


def test_required_model_files_exist():
    assert MODEL.exists(), "Run `python src/train.py` before running the tests."
    assert TRAINING_LOG.exists(), "training_output.txt should be created by train.py."


def test_model_predicts_positive_number():
    model = joblib.load(MODEL)
    row = pd.DataFrame(
        [
            {
                "Area": 2000,
                "Bedrooms": 3,
                "Bathrooms": 2,
                "Floors": 2,
                "YearBuilt": 2000,
                "Location": "Suburban",
                "Condition": "Good",
                "Garage": "Yes",
            }
        ]
    )
    prediction = float(model.predict(row)[0])
    assert prediction > 0

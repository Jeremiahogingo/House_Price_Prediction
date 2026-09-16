import argparse
from pathlib import Path
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL = joblib.load(ROOT / "models" / "house_price_model.joblib")

parser = argparse.ArgumentParser()
parser.add_argument("--area", type=float, required=True)
parser.add_argument("--bedrooms", type=int, required=True)
parser.add_argument("--bathrooms", type=int, required=True)
parser.add_argument("--floors", type=int, required=True)
parser.add_argument("--year-built", type=int, required=True)
parser.add_argument("--location", choices=["Downtown","Suburban","Urban","Rural"], required=True)
parser.add_argument("--condition", choices=["Excellent","Good","Fair","Poor"], required=True)
parser.add_argument("--garage", choices=["Yes","No"], required=True)
args = parser.parse_args()

row = pd.DataFrame([{
    "Area": args.area, "Bedrooms": args.bedrooms, "Bathrooms": args.bathrooms,
    "Floors": args.floors, "YearBuilt": args.year_built,
    "Location": args.location, "Condition": args.condition, "Garage": args.garage
}])
print(f"Predicted price: ${MODEL.predict(row)[0]:,.2f}")

# House Price Prediction Model

## 1. Objective
Predict `Price` from the supplied housing features.

## 2. Dataset
The supplied file contains **2,000 rows and 10 columns**: `Id`, `Area`, `Bedrooms`, `Bathrooms`, `Floors`, `YearBuilt`, `Location`, `Condition`, `Garage`, and `Price`. The prediction target is `Price`. fileciteturn1file0L11-L20

`Id` is excluded from modeling because it is an identifier, not a meaningful house characteristic.

## 3. Pipeline
- Load and validate data
- Split into 80% train / 20% test
- Impute numeric/categorical values
- Standardize numeric features
- One-hot encode categorical features
- Benchmark Ridge, Random Forest, Gradient Boosting, and a mean baseline
- Select the lowest-MAE model
- Retrain the selected pipeline on all labeled data
- Save `models/house_price_model.joblib`
- Serve predictions through Streamlit

## 4. Validation result
The holdout test was run with `random_state=42`.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| ridge | $243,254 | $279,854 | -0.0067 |
| random_forest | $251,299 | $290,829 | -0.0872 |
| gradient_boosting | $243,961 | $281,701 | -0.0200 |
| mean_baseline | $242,480 | $279,024 | -0.0007 |

### Important finding
The **mean baseline** achieved the lowest MAE ($242,480). The ML models do not beat this baseline on the supplied data, and R² values are around zero or negative.

This means the dataset, as supplied, contains very little usable relationship between the available house attributes and `Price`. This should be reported honestly rather than presenting a high-looking accuracy number.

## 5. Run locally

### Windows PowerShell
```powershell
cd house_price_prediction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/train.py
pytest -q
streamlit run app.py
```

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

### Example CLI prediction
```powershell
python src/predict.py --area 2000 --bedrooms 3 --bathrooms 2 --floors 2 --year-built 2000 --location Suburban --condition Good --garage Yes
```

## 6. What to improve for a real estate project
The current features are not enough for a reliable real-world valuation. Add variables such as:
- city/neighborhood or GPS-derived location
- land size
- usable floor area / square footage with consistent units
- property type
- distance to CBD, schools, hospitals and transport
- number of parking spaces
- amenities
- furnishing
- security
- sale/listing date
- local market price index
- comparable nearby transactions

The next dataset should also be checked for synthetic/random target generation before model tuning.

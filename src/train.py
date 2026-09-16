from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "houses.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = ["Area","Bedrooms","Bathrooms","Floors","YearBuilt","Location","Condition","Garage"]
TARGET = "Price"

df = pd.read_csv(DATA)
X, y = df[FEATURES], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

num = ["Area","Bedrooms","Bathrooms","Floors","YearBuilt"]
cat = ["Location","Condition","Garage"]
pre = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
])

models = {
    "ridge": Ridge(alpha=10),
    "random_forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=2, max_features=0.8, random_state=42, n_jobs=-1),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=2, loss="huber", random_state=42),
    "mean_baseline": DummyRegressor(strategy="mean"),
}

results = {}
trained = {}
for name, estimator in models.items():
    pipe = Pipeline([("preprocessor", pre), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    results[name] = {
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R2": r2_score(y_test, pred),
    }
    trained[name] = pipe

selected = min(results, key=lambda n: results[n]["MAE"])
final_model = trained[selected].fit(X, y)
joblib.dump(final_model, MODEL_DIR / "house_price_model.joblib")

metrics = {
    "rows": len(df), "selected_model": selected,
    "results": {k: {m: float(v) for m, v in r.items()} for k, r in results.items()},
}
(MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))

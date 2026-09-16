from pathlib import Path
import json
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "houses.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "house_price_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
TRAINING_OUTPUT = ROOT / "training_output.txt"
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = [
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Floors",
    "YearBuilt",
    "Location",
    "Condition",
    "Garage",
]
TARGET = "Price"
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


def money(value: float) -> str:
    return f"${value:,.2f}"


def build_preprocessor() -> ColumnTransformer:
    numeric = ["Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt"]
    categorical = ["Location", "Condition", "Garage"]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )


def main() -> None:
    log_lines: list[str] = []

    def log(message: str = "") -> None:
        print(message)
        log_lines.append(message)

    started = datetime.now()
    log("HOUSE PRICE PREDICTION - TRAINING OUTPUT")
    log("=" * 62)
    log(f"Training started: {started:%Y-%m-%d %H:%M:%S}")
    log(f"Dataset: {DATA.relative_to(ROOT)}")

    try:
        if not DATA.exists():
            raise FileNotFoundError(f"Dataset not found: {DATA}")

        df = pd.read_csv(DATA)

        # Dataset validation before training.
        if list(df.columns) != EXPECTED_COLUMNS:
            raise ValueError(
                "Unexpected dataset columns. "
                f"Expected {EXPECTED_COLUMNS}, got {list(df.columns)}"
            )
        if df.empty:
            raise ValueError("The dataset is empty.")
        missing = int(df.isna().sum().sum())
        if missing:
            raise ValueError(f"Dataset contains {missing} missing values.")

        log(f"Rows: {len(df):,}")
        log(f"Columns: {len(df.columns)}")
        log(f"Features used: {', '.join(FEATURES)}")
        log(f"Target: {TARGET}")
        log(f"Missing values: {missing}")
        log("")
        log("Feature ranges in supplied dataset:")
        log(f"  Area: {df['Area'].min():,.0f} - {df['Area'].max():,.0f} sq ft")
        log(f"  Bedrooms: {df['Bedrooms'].min():.0f} - {df['Bedrooms'].max():.0f}")
        log(f"  Bathrooms: {df['Bathrooms'].min():.0f} - {df['Bathrooms'].max():.0f}")
        log(f"  Floors: {df['Floors'].min():.0f} - {df['Floors'].max():.0f}")
        log(f"  Year Built: {df['YearBuilt'].min():.0f} - {df['YearBuilt'].max():.0f}")
        log(f"  Price: {money(df[TARGET].min())} - {money(df[TARGET].max())}")
        log("")
        log("Price summary:")
        log(f"  Mean:   {money(df[TARGET].mean())}")
        log(f"  Median: {money(df[TARGET].median())}")
        log("")

        X, y = df[FEATURES], df[TARGET]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )
        log(f"Training rows: {len(X_train):,}")
        log(f"Validation rows: {len(X_test):,}")
        log("")

        models = {
            "ridge": Ridge(alpha=10),
            "random_forest": RandomForestRegressor(
                n_estimators=400,
                min_samples_leaf=2,
                max_features=0.8,
                random_state=42,
                n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=2,
                loss="huber",
                random_state=42,
            ),
            "mean_baseline": DummyRegressor(strategy="mean"),
        }

        results: dict[str, dict[str, float]] = {}
        trained: dict[str, Pipeline] = {}

        for name, estimator in models.items():
            pipe = Pipeline(
                [
                    ("preprocessor", build_preprocessor()),
                    ("model", estimator),
                ]
            )
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            results[name] = {
                "MAE": float(mean_absolute_error(y_test, pred)),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
                "R2": float(r2_score(y_test, pred)),
            }
            trained[name] = pipe

        selected = min(results, key=lambda name: results[name]["MAE"])
        final_model = trained[selected]
        final_model.fit(X, y)
        joblib.dump(final_model, MODEL_PATH)

        metrics = {
            "rows": int(len(df)),
            "selected_model": selected,
            "training_started": started.isoformat(timespec="seconds"),
            "results": results,
        }
        METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        log("Validation results (lower MAE/RMSE is better; R² closer to 1 is better):")
        log(f"{'Model':<20} {'MAE':>14} {'RMSE':>14} {'R²':>10}")
        log("-" * 62)
        for name, scores in results.items():
            log(
                f"{name:<20} {money(scores['MAE']):>14} "
                f"{money(scores['RMSE']):>14} {scores['R2']:>10.4f}"
            )

        log("")
        log(f"Selected model: {selected}")
        log(f"Model file generated: {MODEL_PATH}")
        log(f"Metrics file generated: {METRICS_PATH}")
        if selected == "mean_baseline":
            log(
                "NOTE: The mean baseline performed best on the supplied dataset. "
                "This indicates very weak predictive signal in the generated data."
            )

        finished = datetime.now()
        log(f"Training finished: {finished:%Y-%m-%d %H:%M:%S}")
        log("TRAINING COMPLETED SUCCESSFULLY")

    except Exception as exc:
        log("")
        log("TRAINING FAILED")
        log(f"Error: {type(exc).__name__}: {exc}")
        TRAINING_OUTPUT.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        raise

    TRAINING_OUTPUT.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"Training log saved to: {TRAINING_OUTPUT}")


if __name__ == "__main__":
    main()

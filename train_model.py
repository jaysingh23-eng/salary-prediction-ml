"""
train_model.py
--------------
Trains multiple regression models, compares them, and saves the best one.
Models trained:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - XGBoost Regressor (if xgboost is installed)
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Local imports
sys.path.append(os.path.dirname(__file__))
from preprocessing import load_data, prepare_data, build_preprocessor

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "salary_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)


def build_models() -> dict:
    """
    Return a dictionary of model name → scikit-learn estimator.
    Each model is wrapped in a Pipeline that first runs the preprocessor.
    """
    preprocessor = build_preprocessor()

    models = {
        "Linear Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", LinearRegression()),
            ]
        ),
        "Decision Tree": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("regressor", DecisionTreeRegressor(max_depth=10, random_state=42)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
                    ),
                ),
            ]
        ),
    }

    # Add XGBoost only if it is installed
    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "regressor",
                    XGBRegressor(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=6,
                        random_state=42,
                        verbosity=0,
                    ),
                ),
            ]
        )
        print("[INFO] XGBoost found and added ✓")
    except ImportError:
        print("[INFO] XGBoost not installed – skipping")

    return models


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    """Compute regression metrics for a single trained model."""
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    return {
        "Model": name,
        "MAE":   round(mae, 2),
        "MSE":   round(mse, 2),
        "RMSE":  round(rmse, 2),
        "R²":    round(r2, 4),
    }


def train_and_compare():
    """
    Main training function:
      1. Load and split data
      2. Train all models
      3. Compare metrics
      4. Save the best model
    Returns the results DataFrame and the best trained Pipeline.
    """
    print("\n" + "=" * 60)
    print("   SALARY PREDICTION – MODEL TRAINING PIPELINE")
    print("=" * 60 + "\n")

    # ── 1. Load data ──────────────────────────────────────────────
    data_path = os.path.join(BASE_DIR, "data", "job_salary_prediction_dataset.csv")
    df = load_data(data_path)

    # ── 2. Prepare split ──────────────────────────────────────────
    X_train, X_test, y_train, y_test = prepare_data(df)

    # ── 3. Train all models ───────────────────────────────────────
    models   = build_models()
    results  = []
    trained  = {}

    for name, pipeline in models.items():
        print(f"[TRAINING] {name} ...", end=" ", flush=True)
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(name, pipeline, X_test, y_test)
        results.append(metrics)
        trained[name] = pipeline
        print(f"done  →  R²={metrics['R²']}  RMSE={metrics['RMSE']:,.0f}")

    # ── 4. Compare ────────────────────────────────────────────────
    results_df = pd.DataFrame(results).sort_values("R²", ascending=False)
    print("\n" + "─" * 60)
    print("MODEL COMPARISON (sorted by R² score)")
    print("─" * 60)
    print(results_df.to_string(index=False))
    print("─" * 60)

    # ── 5. Pick best model ────────────────────────────────────────
    best_name  = results_df.iloc[0]["Model"]
    best_model = trained[best_name]
    print(f"\n[BEST MODEL] {best_name}")

    # ── 6. Save best model ────────────────────────────────────────
    joblib.dump(best_model, MODEL_PATH)
    print(f"[SAVED]  Best model → {MODEL_PATH}\n")

    return results_df, best_model, best_name


if __name__ == "__main__":
    train_and_compare()

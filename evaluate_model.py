"""
evaluate_model.py
-----------------
Loads the saved best model and prints a detailed evaluation report,
including per-model metric tables and residual plots.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(__file__))
from preprocessing import load_data, prepare_data
from train_model import train_and_compare

BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(BASE_DIR, "models", "salary_model.pkl")
IMAGE_DIR  = os.path.join(BASE_DIR, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)


def plot_actual_vs_predicted(y_test, y_pred, model_name: str) -> None:
    """Scatter plot of actual vs predicted salaries."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.3, color="#4C72B0", edgecolors="none", s=15)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")
    ax.set_xlabel("Actual Salary ($)", fontsize=12)
    ax.set_ylabel("Predicted Salary ($)", fontsize=12)
    ax.set_title(f"Actual vs Predicted – {model_name}", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(IMAGE_DIR, "actual_vs_predicted.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[SAVED] {path}")


def plot_residuals(y_test, y_pred, model_name: str) -> None:
    """Residual distribution plot (errors)."""
    residuals = y_test.values - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs predicted
    axes[0].scatter(y_pred, residuals, alpha=0.3, color="#DD8452", s=15)
    axes[0].axhline(0, color="red", linewidth=2, linestyle="--")
    axes[0].set_xlabel("Predicted Salary ($)", fontsize=12)
    axes[0].set_ylabel("Residual ($)", fontsize=12)
    axes[0].set_title(f"Residuals vs Predicted – {model_name}", fontsize=13, fontweight="bold")

    # Distribution of residuals
    axes[1].hist(residuals, bins=50, color="#55A868", edgecolor="white", alpha=0.85)
    axes[1].axvline(0, color="red", linewidth=2, linestyle="--")
    axes[1].set_xlabel("Residual ($)", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("Residual Distribution", fontsize=13, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(IMAGE_DIR, "residuals.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[SAVED] {path}")


def plot_model_comparison(results_df: pd.DataFrame) -> None:
    """Bar chart comparing R² and RMSE across all models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # R² comparison
    colors = sns.color_palette("muted", len(results_df))
    axes[0].barh(results_df["Model"], results_df["R²"], color=colors)
    axes[0].set_xlabel("R² Score", fontsize=12)
    axes[0].set_title("Model Comparison – R² Score", fontsize=13, fontweight="bold")
    axes[0].set_xlim(0, 1.05)
    for i, v in enumerate(results_df["R²"]):
        axes[0].text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=10)

    # RMSE comparison
    axes[1].barh(results_df["Model"], results_df["RMSE"], color=colors)
    axes[1].set_xlabel("RMSE ($)", fontsize=12)
    axes[1].set_title("Model Comparison – RMSE", fontsize=13, fontweight="bold")
    for i, v in enumerate(results_df["RMSE"]):
        axes[1].text(v + 100, i, f"${v:,.0f}", va="center", fontsize=10)

    plt.tight_layout()
    path = os.path.join(IMAGE_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[SAVED] {path}")


def run_evaluation():
    """Full evaluation pipeline."""
    print("\n" + "=" * 60)
    print("   MODEL EVALUATION REPORT")
    print("=" * 60 + "\n")

    data_path = os.path.join(BASE_DIR, "data", "job_salary_prediction_dataset.csv")
    df = load_data(data_path)
    _, X_test, _, y_test = prepare_data(df)

    if not os.path.exists(MODEL_PATH):
        print("[ERROR] Model not found. Run train_model.py first.")
        return

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)

    # Summary metrics
    mae  = np.mean(np.abs(y_test.values - y_pred))
    rmse = np.sqrt(np.mean((y_test.values - y_pred) ** 2))
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_pred)

    print(f"  MAE  : ${mae:>12,.2f}")
    print(f"  RMSE : ${rmse:>12,.2f}")
    print(f"  R²   :  {r2:>12.4f}")
    print()

    model_name = "Best Model"
    plot_actual_vs_predicted(y_test, y_pred, model_name)
    plot_residuals(y_test, y_pred, model_name)

    print("\n[DONE] Evaluation complete. Charts saved to images/")


if __name__ == "__main__":
    run_evaluation()

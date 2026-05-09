"""
preprocessing.py
----------------
Handles all data loading, cleaning, encoding, and splitting tasks.
This module is used by both the training pipeline and the Streamlit app.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
import os

# ─── Column definitions ───────────────────────────────────────────────────────
CATEGORICAL_FEATURES = [
    "job_title",
    "education_level",
    "industry",
    "company_size",
    "location",
    "remote_work",
]
NUMERICAL_FEATURES = ["experience_years", "skills_count", "certifications"]
TARGET_COLUMN = "salary"

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "job_salary_prediction_dataset.csv"
)


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the dataset from CSV.
    Returns a cleaned DataFrame.
    """
    df = pd.read_csv(path)
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def check_missing_values(df: pd.DataFrame) -> None:
    """Print a summary of missing values in the dataset."""
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("[INFO] No missing values found ✓")
    else:
        print("[WARNING] Missing values detected:")
        print(missing[missing > 0])


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values:
      - Numerical columns → median
      - Categorical columns → mode (most frequent)
    """
    df = df.copy()
    for col in NUMERICAL_FEATURES:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    for col in CATEGORICAL_FEATURES:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer that:
      - OneHotEncodes categorical features
      - StandardScales numerical features
    Returns the unfitted preprocessor (will be fit inside a Pipeline).
    """
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Return human-readable feature names after transformation."""
    num_names = NUMERICAL_FEATURES
    cat_names = (
        preprocessor.named_transformers_["cat"]
        .get_feature_names_out(CATEGORICAL_FEATURES)
        .tolist()
    )
    return num_names + cat_names


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Full preprocessing pipeline:
      1. Handle missing values
      2. Split into X (features) and y (target)
      3. Train/test split
    Returns X_train, X_test, y_train, y_test
    """
    df = handle_missing_values(df)

    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"[INFO] Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def save_preprocessor(preprocessor, path: str = "../models/preprocessor.pkl") -> None:
    """Save the fitted preprocessor to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(preprocessor, path)
    print(f"[INFO] Preprocessor saved → {path}")


def load_preprocessor(path: str = "../models/preprocessor.pkl"):
    """Load a previously saved preprocessor from disk."""
    return joblib.load(path)

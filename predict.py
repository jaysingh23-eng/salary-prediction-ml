"""
predict.py
----------
Provides a clean predict() function used by both main.py and app.py.
Loads the saved model and returns a salary prediction for new input.
"""

import os
import sys
import joblib
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from preprocessing import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(BASE_DIR, "models", "salary_model.pkl")


def load_model(path: str = MODEL_PATH):
    """Load the trained model pipeline from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at '{path}'. "
            "Please run 'python main.py' or 'python src/train_model.py' first."
        )
    return joblib.load(path)


def predict_salary(
    job_title: str,
    experience_years: int,
    education_level: str,
    skills_count: int,
    industry: str,
    company_size: str,
    location: str,
    remote_work: str,
    certifications: int,
    model=None,
) -> float:
    """
    Predict the salary for a single candidate profile.

    Parameters
    ----------
    job_title        : One of the job titles in the dataset
    experience_years : 0–20
    education_level  : 'High School', 'Diploma', 'Bachelor', 'Master', 'PhD'
    skills_count     : 1–19
    industry         : Industry name
    company_size     : 'Startup', 'Small', 'Medium', 'Large', 'Enterprise'
    location         : Country / region
    remote_work      : 'Yes', 'No', 'Hybrid'
    certifications   : 0–5
    model            : (optional) pre-loaded model; if None it will be loaded

    Returns
    -------
    float  : Predicted annual salary in USD
    """
    if model is None:
        model = load_model()

    # Build a single-row DataFrame matching training column order
    input_data = pd.DataFrame(
        {
            "experience_years": [experience_years],
            "skills_count":     [skills_count],
            "certifications":   [certifications],
            "job_title":        [job_title],
            "education_level":  [education_level],
            "industry":         [industry],
            "company_size":     [company_size],
            "location":         [location],
            "remote_work":      [remote_work],
        }
    )

    predicted = model.predict(input_data)[0]
    return round(float(predicted), 2)


if __name__ == "__main__":
    # Quick smoke test
    salary = predict_salary(
        job_title="Data Scientist",
        experience_years=5,
        education_level="Master",
        skills_count=8,
        industry="Technology",
        company_size="Large",
        location="USA",
        remote_work="Yes",
        certifications=2,
    )
    print(f"Predicted Salary: ${salary:,.2f}")

"""
main.py
-------
Project entry point.
Run this file once to:
  1. Train all models and save the best one
  2. Generate all EDA charts
  3. Run evaluation and produce comparison charts
  4. Print a sample prediction

Usage
-----
    python main.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from train_model import train_and_compare
from evaluate_model import run_evaluation
from predict import predict_salary

# ─── Optional: generate EDA plots ─────────────────────────────────────────────
# If you want to regenerate EDA charts (requires running once after data is ready)
EDA_SCRIPT = os.path.join(os.path.dirname(__file__), "notebooks", "eda_script.py")


def run_eda():
    """Run EDA analysis and save plots."""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")          # Non-interactive backend for saving
    import matplotlib.pyplot as plt
    import seaborn as sns

    DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "job_salary_prediction_dataset.csv")
    IMAGE_DIR = os.path.join(os.path.dirname(__file__), "images")
    os.makedirs(IMAGE_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Salary Distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["salary"], kde=True, color="#4C72B0", bins=60, ax=ax)
    ax.set_title("Salary Distribution", fontsize=15, fontweight="bold")
    ax.set_xlabel("Annual Salary (USD)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "salary_distribution.png"), dpi=150)
    plt.close()
    print("[EDA] salary_distribution.png saved")

    # 2. Experience vs Salary
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=df, x="experience_years", y="salary", alpha=0.3, ax=ax, color="#55A868", s=10)
    sns.regplot(data=df, x="experience_years", y="salary", scatter=False, color="red", ax=ax)
    ax.set_title("Experience Years vs Salary", fontsize=15, fontweight="bold")
    ax.set_xlabel("Experience (Years)", fontsize=12)
    ax.set_ylabel("Salary (USD)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "experience_vs_salary.png"), dpi=150)
    plt.close()
    print("[EDA] experience_vs_salary.png saved")

    # 3. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    num_df = df[["experience_years", "skills_count", "certifications", "salary"]]
    sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax, linewidths=0.5)
    ax.set_title("Correlation Heatmap", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()
    print("[EDA] correlation_heatmap.png saved")

    # 4. Top Job Titles by Average Salary
    top_jobs = df.groupby("job_title")["salary"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=top_jobs.values, y=top_jobs.index, palette="muted", ax=ax)
    ax.set_title("Average Salary by Job Title", fontsize=15, fontweight="bold")
    ax.set_xlabel("Average Salary (USD)", fontsize=12)
    ax.set_ylabel("")
    for i, v in enumerate(top_jobs.values):
        ax.text(v + 500, i, f"${v:,.0f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "top_job_titles_salary.png"), dpi=150)
    plt.close()
    print("[EDA] top_job_titles_salary.png saved")

    # 5. Industry Salary Comparison
    industry_salary = df.groupby("industry")["salary"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=industry_salary.values, y=industry_salary.index, palette="Set2", ax=ax)
    ax.set_title("Average Salary by Industry", fontsize=15, fontweight="bold")
    ax.set_xlabel("Average Salary (USD)", fontsize=12)
    ax.set_ylabel("")
    for i, v in enumerate(industry_salary.values):
        ax.text(v + 200, i, f"${v:,.0f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "industry_salary_comparison.png"), dpi=150)
    plt.close()
    print("[EDA] industry_salary_comparison.png saved")

    # 6. Remote Work Salary Analysis
    fig, ax = plt.subplots(figsize=(8, 5))
    order = ["Yes", "Hybrid", "No"]
    sns.boxplot(data=df, x="remote_work", y="salary", order=order, palette="pastel", ax=ax)
    ax.set_title("Remote Work vs Salary", fontsize=15, fontweight="bold")
    ax.set_xlabel("Remote Work", fontsize=12)
    ax.set_ylabel("Salary (USD)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "remote_work_salary.png"), dpi=150)
    plt.close()
    print("[EDA] remote_work_salary.png saved")

    # 7. Education Level vs Salary
    edu_order = ["High School", "Diploma", "Bachelor", "Master", "PhD"]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="education_level", y="salary", order=edu_order, palette="Blues", ax=ax)
    ax.set_title("Education Level vs Salary", fontsize=15, fontweight="bold")
    ax.set_xlabel("Education Level", fontsize=12)
    ax.set_ylabel("Salary (USD)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "education_vs_salary.png"), dpi=150)
    plt.close()
    print("[EDA] education_vs_salary.png saved")

    print(f"\n[EDA] All plots saved to {IMAGE_DIR}/\n")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("   SALARY PREDICTION PROJECT – FULL PIPELINE")
    print("=" * 60)

    # Step 1: EDA
    print("\n[STEP 1] Running Exploratory Data Analysis...")
    run_eda()

    # Step 2: Train
    print("\n[STEP 2] Training Models...")
    results_df, best_model, best_name = train_and_compare()

    # Step 3: Evaluate
    print("\n[STEP 3] Evaluating Best Model...")
    run_evaluation()

    # Step 4: Sample prediction
    print("\n[STEP 4] Sample Prediction")
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
    print(f"  Profile  : Data Scientist | 5 yrs | Master's | Tech | USA")
    print(f"  Predicted: ${salary:,.2f} per year")

    print("\n" + "=" * 60)
    print("  ALL STEPS COMPLETE ✓")
    print(f"  Best model : {best_name}")
    print("  Model saved: models/salary_model.pkl")
    print("  Charts saved: images/")
    print("  To launch the web app: streamlit run app.py")
    print("=" * 60 + "\n")

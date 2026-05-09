"""
app.py
------
Streamlit web application for the Salary Prediction project.

Run with:
    streamlit run app.py
"""

import os
import sys
import joblib
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from predict import predict_salary, load_model

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="💼 Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Main background */
        .main { background-color: #f0f4f8; }
        /* Sidebar style */
        section[data-testid="stSidebar"] {
            background: linear-gradient(160deg, #1e3a5f 0%, #2d6a9f 100%);
            color: white;
        }
        section[data-testid="stSidebar"] * { color: white !important; }
        section[data-testid="stSidebar"] .stSelectbox > div > div { background: rgba(255,255,255,0.15); }
        section[data-testid="stSidebar"] .stSlider > div { background: transparent; }
        /* Prediction card */
        .pred-card {
            background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
            border-radius: 16px;
            padding: 36px 24px;
            text-align: center;
            color: white;
            box-shadow: 0 8px 32px rgba(30,58,95,0.25);
            margin-top: 16px;
        }
        .pred-card h1 { font-size: 3rem; margin: 8px 0; }
        .pred-card p  { font-size: 1.1rem; opacity: 0.9; }
        /* Metric cards row */
        .metric-card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        }
        /* Hide Streamlit default footer */
        footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Header ───────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_title:
    st.markdown("# 💼 AI-Powered Salary Predictor")
    st.markdown(
        "Enter your professional profile in the sidebar to get an instant salary estimate "
        "powered by a Machine Learning model trained on 250,000 job records."
    )

st.markdown("---")

# ─── Sidebar Inputs ───────────────────────────────────────────────────────────
st.sidebar.markdown("## 🎯 Your Professional Profile")
st.sidebar.markdown("Fill in the fields below and hit **Predict**.")
st.sidebar.markdown("---")

job_title = st.sidebar.selectbox(
    "💼 Job Title",
    options=[
        "AI Engineer", "Backend Developer", "Business Analyst",
        "Cloud Engineer", "Cybersecurity Analyst", "Data Analyst",
        "Data Scientist", "DevOps Engineer", "Frontend Developer",
        "Machine Learning Engineer", "Product Manager", "Software Engineer",
    ],
)

experience_years = st.sidebar.slider(
    "📅 Years of Experience", min_value=0, max_value=20, value=5, step=1
)

education_level = st.sidebar.selectbox(
    "🎓 Education Level",
    options=["High School", "Diploma", "Bachelor", "Master", "PhD"],
    index=2,
)

skills_count = st.sidebar.slider(
    "🛠️ Number of Skills", min_value=1, max_value=19, value=8, step=1
)

industry = st.sidebar.selectbox(
    "🏭 Industry",
    options=[
        "Consulting", "Education", "Finance", "Government",
        "Healthcare", "Manufacturing", "Media", "Retail",
        "Technology", "Telecom",
    ],
    index=8,
)

company_size = st.sidebar.selectbox(
    "🏢 Company Size",
    options=["Startup", "Small", "Medium", "Large", "Enterprise"],
    index=3,
)

location = st.sidebar.selectbox(
    "🌍 Location",
    options=[
        "Australia", "Canada", "Germany", "India",
        "Netherlands", "Remote", "Singapore",
        "Sweden", "UK", "USA",
    ],
    index=9,
)

remote_work = st.sidebar.selectbox(
    "🏠 Remote Work",
    options=["Yes", "No", "Hybrid"],
    index=0,
)

certifications = st.sidebar.slider(
    "📜 Number of Certifications", min_value=0, max_value=5, value=2, step=1
)

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔮 Predict My Salary", use_container_width=True)

# ─── Prediction ───────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "salary_model.pkl")

# Main area – two columns
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.subheader("📋 Your Profile Summary")

    profile_data = {
        "Field": [
            "Job Title", "Experience", "Education",
            "Skills", "Industry", "Company Size",
            "Location", "Remote Work", "Certifications",
        ],
        "Value": [
            job_title, f"{experience_years} years", education_level,
            str(skills_count), industry, company_size,
            location, remote_work, str(certifications),
        ],
    }
    st.table(pd.DataFrame(profile_data))

with right_col:
    st.subheader("💰 Salary Prediction")

    if predict_btn:
        if not os.path.exists(MODEL_PATH):
            st.error(
                "⚠️ Model not found. Please run `python main.py` first to train and save the model."
            )
        else:
            with st.spinner("🤖 Predicting..."):
                try:
                    model = load_model(MODEL_PATH)
                    predicted = predict_salary(
                        job_title=job_title,
                        experience_years=experience_years,
                        education_level=education_level,
                        skills_count=skills_count,
                        industry=industry,
                        company_size=company_size,
                        location=location,
                        remote_work=remote_work,
                        certifications=certifications,
                        model=model,
                    )

                    # Salary range ± 10%
                    low  = predicted * 0.90
                    high = predicted * 1.10

                    st.markdown(
                        f"""
                        <div class="pred-card">
                            <p>🎯 Estimated Annual Salary</p>
                            <h1>${predicted:,.0f}</h1>
                            <p>Salary Range: <strong>${low:,.0f} – ${high:,.0f}</strong></p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Breakdown metrics
                    st.markdown("#### 📊 Breakdown")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Monthly", f"${predicted/12:,.0f}")
                    m2.metric("Weekly",  f"${predicted/52:,.0f}")
                    m3.metric("Hourly",  f"${predicted/2080:,.1f}")

                    st.success("✅ Prediction successful!")

                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
    else:
        st.info(
            "👈 Fill in your profile in the sidebar and click **Predict My Salary** to see your estimate."
        )
        # Placeholder graphic
        st.image(
            "https://img.icons8.com/clouds/400/salary.png",
            width=220,
        )

# ─── EDA Images Section ───────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📈 Dataset Insights")

IMAGE_DIR = os.path.join(os.path.dirname(__file__), "images")

eda_images = {
    "Salary Distribution":        "salary_distribution.png",
    "Experience vs Salary":       "experience_vs_salary.png",
    "Top Job Titles by Salary":   "top_job_titles_salary.png",
    "Industry Salary Comparison": "industry_salary_comparison.png",
    "Remote Work vs Salary":      "remote_work_salary.png",
    "Education vs Salary":        "education_vs_salary.png",
    "Correlation Heatmap":        "correlation_heatmap.png",
    "Actual vs Predicted":        "actual_vs_predicted.png",
    "Model Comparison":           "model_comparison.png",
}

# Display in a 2-column grid
keys = list(eda_images.keys())
for i in range(0, len(keys), 2):
    cols = st.columns(2)
    for j, col in enumerate(cols):
        if i + j < len(keys):
            title = keys[i + j]
            img_path = os.path.join(IMAGE_DIR, eda_images[title])
            if os.path.exists(img_path):
                col.image(img_path, caption=title, use_container_width=True)
            else:
                col.info(f"📊 {title} (Run main.py to generate)")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center>Built with ❤️ using Python · Scikit-learn · Streamlit | "
    "Trained on 250,000 job records</center>",
    unsafe_allow_html=True,
)

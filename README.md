# 💼 AI-Powered Salary Predictor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-189fdd?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)

**Predict tech salaries with 97.9% accuracy using Machine Learning**

[🚀 Live Demo](#-deployment) · [📊 Model Performance](#-model-performance) · [⚙️ Installation](#-installation)

</div>

---

## 📌 Project Overview

This end-to-end Machine Learning project predicts annual tech salaries based on a professional's profile — job title, experience, education, skills, and more. Trained on **250,000 real job records**, the best model (XGBoost) achieves an **R² of 0.979** and an average prediction error of only **~$4,300**.

The project includes a production-ready **Streamlit web app** where anyone can input their profile and receive an instant salary estimate.

---

## ✨ Features

- ✅ **4 ML models** trained and compared automatically
- ✅ **97.9% accuracy** (R²) with XGBoost
- ✅ **250,000 records** — robust and generalizable model
- ✅ **7 EDA charts** generated automatically
- ✅ **Streamlit web app** with a beautiful UI
- ✅ **Clean, beginner-friendly code** with comments
- ✅ **Sklearn Pipelines** for reproducible preprocessing
- ✅ **Model saved with Joblib** for reuse

---

## 📂 Project Structure

```
salary_prediction_project/
│
├── data/
│   └── job_salary_prediction_dataset.csv   # 250k-row dataset
│
├── images/                                 # Auto-generated EDA & evaluation plots
│   ├── salary_distribution.png
│   ├── experience_vs_salary.png
│   ├── correlation_heatmap.png
│   ├── top_job_titles_salary.png
│   ├── industry_salary_comparison.png
│   ├── remote_work_salary.png
│   ├── education_vs_salary.png
│   ├── actual_vs_predicted.png
│   ├── residuals.png
│   └── model_comparison.png
│
├── models/
│   └── salary_model.pkl                    # Best trained model (XGBoost)
│
├── notebooks/
│   └── eda.ipynb                           # Jupyter notebook for EDA
│
├── src/
│   ├── preprocessing.py                    # Data loading, cleaning, encoding
│   ├── train_model.py                      # Model training + comparison
│   ├── evaluate_model.py                   # Evaluation metrics + plots
│   └── predict.py                          # Prediction function
│
├── app.py                                  # Streamlit web application
├── main.py                                 # ← RUN THIS FIRST
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

| Feature            | Description                                        |
|--------------------|----------------------------------------------------|
| `job_title`        | Role (e.g., Data Scientist, AI Engineer)          |
| `experience_years` | Years of work experience (0–20)                    |
| `education_level`  | High School / Diploma / Bachelor / Master / PhD   |
| `skills_count`     | Number of technical skills (1–19)                  |
| `industry`         | Sector (Technology, Finance, Healthcare, etc.)     |
| `company_size`     | Startup / Small / Medium / Large / Enterprise      |
| `location`         | Country (USA, India, UK, Canada, etc.)             |
| `remote_work`      | Yes / No / Hybrid                                  |
| `certifications`   | Number of professional certifications (0–5)        |
| `salary` *(target)*| Annual salary in USD                               |

- **Rows**: 250,000
- **Missing values**: None
- **Salary range**: $31,867 – $333,046

---

## 🤖 Model Performance

All 4 models were trained and evaluated on a 20% held-out test set (50,000 rows).

| Model             |   MAE ($) |     MSE |   RMSE ($) |  R² Score |
|-------------------|----------:|--------:|-----------:|----------:|
| **XGBoost** ⭐    | **4,296** | **29M** |  **5,386** | **0.9791** |
| Linear Regression |     5,436 |    51M  |      7,126 |    0.9635 |
| Random Forest     |    10,184 |   170M  |     13,020 |    0.8780 |
| Decision Tree     |    12,907 |   269M  |     16,407 |    0.8063 |

> **XGBoost** was automatically selected as the best model and saved to `models/salary_model.pkl`.

---

## 🛠️ Technologies Used

| Library        | Purpose                          |
|----------------|----------------------------------|
| `pandas`       | Data loading & manipulation      |
| `numpy`        | Numerical operations             |
| `matplotlib`   | Base plotting                    |
| `seaborn`      | Statistical visualizations       |
| `scikit-learn` | Preprocessing, models, metrics   |
| `xgboost`      | Best-performing regression model |
| `joblib`       | Model serialization              |
| `streamlit`    | Interactive web application      |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/salary-prediction-ml.git
cd salary-prediction-ml
```

### 2. Create a virtual environment (recommended)

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the full pipeline

```bash
python main.py
```

This will:
- Generate all 7 EDA charts → `images/`
- Train all 4 ML models
- Save the best model → `models/salary_model.pkl`
- Print a sample prediction

### 5. Launch the Streamlit app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 📸 Screenshots

### 🌐 Web Application
> Fill in your profile → click **Predict My Salary** → instant result

### 📈 EDA Charts

| Chart | Description |
|---|---|
| `salary_distribution.png` | Histogram with KDE of all salaries |
| `experience_vs_salary.png` | Scatter + regression line |
| `top_job_titles_salary.png` | Horizontal bar chart of avg salary |
| `industry_salary_comparison.png` | Industry-wise averages |
| `remote_work_salary.png` | Box plot by remote policy |
| `education_vs_salary.png` | Box plot by education level |
| `correlation_heatmap.png` | Pearson correlation of numerical features |

---

## 🚀 Deployment

### Deploy to Streamlit Community Cloud (Free)

1. Push your project to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select repo → select `app.py` → **Deploy**
5. Your app is live at `https://your-app.streamlit.app`

> **Important**: Make sure `models/salary_model.pkl` is committed to GitHub, OR add a `@st.cache_resource` model-training step at app startup.

---

## 🔮 Future Improvements

- [ ] Add more features (gender, company rating, job posting date)
- [ ] Hyperparameter tuning with GridSearchCV / Optuna
- [ ] SHAP values for model explainability
- [ ] Add a "compare yourself to market" feature
- [ ] Deploy as a REST API (FastAPI + Docker)
- [ ] Add CI/CD with GitHub Actions

---

## 👤 Author

**Your Name**
- 🌐 Portfolio: [yourportfolio.com](https://yourportfolio.com)
- 💼 LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
- 🐙 GitHub: [@yourusername](https://github.com/yourusername)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
⭐ If you found this project useful, please give it a star on GitHub!
</div>

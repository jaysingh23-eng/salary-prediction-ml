# 💼 AI-Powered Salary Predictor

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?logo=streamlit)]()
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)]()
[![XGBoost](https://img.shields.io/badge/XGBoost-Best_Model-green)]()
[![Live App](https://img.shields.io/badge/Streamlit-Live_App-red?logo=streamlit)](https://YOUR-STREAMLIT-APP.streamlit.app)

---
## 📌 Project Overview

This end-to-end Machine Learning project predicts annual tech salaries based on a professional's profile — job title, experience, education, skills, and more. Trained on **250,000 real job records**, the best model (XGBoost) achieves an **R² of 0.979** and an average prediction error of only **~$4,300**.

The project includes a production-ready **Streamlit web app** where anyone can input their profile and receive an instant salary estimate.

---

---

Predict tech salaries using Machine Learning based on:

- Experience
- Education
- Skills
- Industry
- Company Size
- Remote Work
- Certifications

Built with:

- Python
- Scikit-learn
- XGBoost
- Streamlit
- Pandas

---

# 📌 Project Overview

This end-to-end Machine Learning project predicts annual tech salaries using multiple professional attributes such as job title, experience, education, industry, and more.

The project includes:

✅ Data preprocessing pipeline  
✅ Exploratory Data Analysis (EDA)  
✅ Multiple ML model training & comparison  
✅ XGBoost best model selection  
✅ Streamlit web application  
✅ Visualization charts  
✅ Model serialization using Joblib  

---

# ✨ Features

- ✅ Predict salaries instantly
- ✅ Streamlit interactive UI
- ✅ Multiple ML models compared
- ✅ XGBoost best-performing model
- ✅ Automated preprocessing pipeline
- ✅ Professional EDA visualizations
- ✅ Beginner-friendly clean code

---

# 📂 Project Structure

```text
salary_prediction_project/
│
├── data/
├── images/
├── models/
├── notebooks/
├── src/
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📊 Dataset Information

| Feature | Description |
|---|---|
| job_title | Role name |
| experience_years | Years of experience |
| education_level | Education qualification |
| skills_count | Number of technical skills |
| industry | Industry sector |
| company_size | Startup / Small / Large |
| location | Country |
| remote_work | Remote / Hybrid / Onsite |
| certifications | Professional certifications |
| salary | Target variable |

---

# 🤖 Model Performance

| Model | R² Score |
|---|---|
| XGBoost ⭐ | 0.979 |
| Linear Regression | 0.963 |
| Random Forest | 0.878 |
| Decision Tree | 0.806 |

Best model automatically saved as:

```text
models/salary_model.pkl
```

---

# 📸 Application Screenshots

## 🏠 Home Page

![Home Page](images/app_home.png)

---

## 📈 Prediction Result

![Prediction Result](images/prediction_result.png)

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Streamlit
- Joblib

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/jaysingh23-eng/salary-prediction-project.git
cd salary-prediction-project
```

## 2️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

## 3️⃣ Train Model

```bash
python main.py
```

## 4️⃣ Run Streamlit App

```bash
streamlit run app.py
```

---

# 🚀 Deployment

This project is deployed using Streamlit Community Cloud.

### Deploy Your Own

1. Push project to GitHub
2. Open Streamlit Cloud
3. Connect GitHub repository
4. Select `app.py`
5. Click Deploy

---

# 🔮 Future Improvements

- Add SHAP explainability
- Add salary trend forecasting
- Add resume-based prediction
- Add FastAPI backend
- Add Power BI dashboard integration
- Add authentication system

---

# 👨‍💻 Author

## Jay Singh


- 🌐 GitHub: https://github.com/jaysingh23-eng
- 💼 LinkedIn: https://www.linkedin.com/in/jay-singh-206356349/

---

# ⭐ Support

If you found this project useful, give it a ⭐ on GitHub.

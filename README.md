# 💼 Adult Income Prediction using Machine Learning

An end-to-end Machine Learning project to predict whether an individual earns **more than 50K annually** based on demographic and employment-related features.

---

## 🚀 Project Overview
This project uses the **Adult Income dataset** to build a binary classification model that predicts:

- `<=50K`
- `>50K`

The project includes:
- data cleaning
- feature engineering
- preprocessing pipelines
- model selection
- threshold tuning
- deployment using Streamlit

---

## 🛠 Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

---

## 🧹 Data Preprocessing
- handled missing values
- dropped `fnlwgt`
- feature engineering:
  - `age_group`
  - `hours_group`
- simplified `native-country` into:
  - `US`
  - `Other`

---

## 🤖 Final Model
**Logistic Regression**

### Hyperparameters
- `l1_ratio = 1`
- `C = 0.1`
- `solver = liblinear`
- `class_weight = balanced`

### Decision Threshold
- `0.7`

---

## 📊 Model Performance
- **Cross Validation F1:** `0.6856`
- **ROC-AUC:** `0.909`

---

## 🌐 Streamlit App
The model is deployed as an interactive Streamlit application where users can enter details and get income prediction instantly.

---

## 📁 Project Structure

git_projects/
└── adult_income_prediction/
    ├── adult.csv
    ├── adult_income.py
    ├── adult_income_final.py
    ├── adult_income_final.pkl
    ├── st_app.py
    └── README.md

### 📌 File Description
- `adult.csv` → dataset used for training
- `adult_income.py` → experimentation and model development file
- `adult_income_final.py` → cleaned final training pipeline
- `adult_income_final.pkl` → saved trained model
- `st_app.py` → Streamlit deployment app
- `README.md` → project documentation

## 🔄 Development Workflow

This project was developed in two major phases:

### 1. Experimentation Phase
The file `adult_income.py` served as the foundation of the project.

This phase included:
- data cleaning
- preprocessing experiments
- model comparison
- Random Forest vs Logistic Regression
- cross-validation
- hyperparameter tuning
- threshold tuning
- SHAP interpretability
- ROC and Precision-Recall curve analysis

### 2. Finalization Phase
After identifying the best-performing approach, the workflow was refactored into `adult_income_final.py`.

This final version includes:
- clean production-ready code
- optimized Logistic Regression model
- threshold = 0.7
- model serialization using joblib
- deployment-ready pipeline
```

## 🎯 Key Learning Outcomes
- preprocessing pipelines
- model evaluation
- cross-validation
- threshold optimization
- deployment debugging
- end-to-end ML workflow

---

## 👨‍💻 Author
Shubham
Machine Learning Engineer in the Making 🚀 
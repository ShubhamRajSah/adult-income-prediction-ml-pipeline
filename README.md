# Adult Income Prediction using Machine Learning

An end-to-end machine learning project to predict whether a person's annual income is **>50K or <=50K** using the Adult Census dataset.

---

## 🎯 Problem Statement
The objective of this project is to build a binary classification model that predicts income level based on demographic and employment-related features.

---

## 📊 Dataset
- Source: Kaggle Adult Census Dataset
- Records: 48,842
- Features: 15 original columns
- Target: `income`

---

## ⚙️ Workflow
1. Data cleaning
2. Missing value handling
3. Feature preprocessing
4. Model training
5. Evaluation
6. Threshold tuning
7. SHAP explainability

---

## 🛠️ Preprocessing
- Replaced `?` with missing values
- Handled null values
- Dropped `fnlwgt`
- Applied OneHotEncoding for categorical features
- Applied scaling for numerical features
- Used sklearn `Pipeline` to prevent data leakage

---

## 🤖 Models Used
- Logistic Regression
- Balanced Logistic Regression
- Random Forest Classifier

Final selected model:
**Random Forest with threshold = 0.7**

---

## 📈 Results
- Accuracy: **85%**
- Better stability compared to logistic regression
- Balanced precision and recall

---

## 🔍 Explainability
Used **SHAP (SHapley Additive exPlanations)** to understand feature impact.

Important features:
- Age
- Educational Number
- Marital Status
- Capital Gain
- Hours per Week

---

## 💡 Key Learnings
This project helped me understand:
- End-to-end ML pipeline building
- Model selection
- Threshold tuning
- Explainable AI
- Real-world debugging and patience

---

## 🚀 Future Improvements
- Hyperparameter tuning
- Cross-validation
- More feature engineering
- Deployment using Streamlit / Flask
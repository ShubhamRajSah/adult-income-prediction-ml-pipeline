import streamlit as st
import pandas as pd
import joblib

# Load model
saved_obj = joblib.load("adult_income_final.pkl")
model = saved_obj['model']
threshold = saved_obj['threshold']
st.set_page_config(page_title="Adult Income Predictor")

st.title("💼 Adult Income Prediction")
st.write("Predict whether annual income is >50K")

age = st.number_input("Age", min_value=18, max_value=100, value=30)

education = st.selectbox(
    "Education",
    [
        "Bachelors",
        "HS-grad",
        "Some-college",
        "Masters",
        "Assoc-voc",
        "11th",
        "Doctorate"
    ]
)

education_num = st.number_input(
    "Educational Number",
    min_value=1,
    max_value=16,
    value=10
)

hours = st.number_input(
    "Hours per week",
    min_value=1,
    max_value=100,
    value=40
)

workclass = st.selectbox(
    "Workclass",
    ["Private", "Self-emp-not-inc", "Government", "Missing"]
)

occupation = st.selectbox(
    "Occupation",
    ["Tech-support", "Sales", "Exec-managerial", "Missing"]
)

country = st.selectbox(
    "Country",
    ["US", "Other"]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Never-married", "Married-civ-spouse", "Divorced"]
)

relationship = st.selectbox(
    "Relationship",
    ["Not-in-family", "Husband", "Wife", "Own-child"]
)

race = st.selectbox(
    "Race",
    ["White", "Black", "Asian-Pac-Islander", "Other"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

# feature engineering
age_group = pd.cut(
    [age],
    bins=[0, 25, 40, 60, 100],
    labels=["Young", "Adult", "Mid", "Senior"]
)[0]

hours_group = pd.cut(
    [hours],
    bins=[0, 25, 40, 60, 100],
    labels=["Part-time", "Normal", "Overtime", "Heavy"]
)[0]

if st.button("Predict"):
    input_df = pd.DataFrame({
        "age": [age],
        "education": [education],
        "educational-num": [education_num],
        "capital-gain": [0],
        "capital-loss": [0],
        "hours-per-week": [hours],
        "workclass": [workclass],
        "occupation": [occupation],
        "native-country": [country],
        "marital-status": [marital_status],
        "relationship": [relationship],
        "race": [race],
        "gender": [gender],
        "age_group": [age_group],
        "hours_group": [hours_group]
    })

    prob = model.predict_proba(input_df)[:, 1][0]

    
    prediction = 1 if prob >= threshold else 0

    st.write(f"Prediction Probability: {prob:.2f}")

    if prediction == 1:
        st.success("Predicted Income: >50K")
    else:
        st.error("Predicted Income: <=50K")


import streamlit as st
import joblib
import pandas as pd
import numpy as np

#Load the model and label encoder

model=joblib.load("attrition_model.pkl")
label_encoder=joblib.load("label_encoder.pkl")
feature_columns=joblib.load("feature_columns.pkl")

st.title("Employee Attrition Prediction")
st.markdown("Enter the employee details to predict if they are " "likely to leave the company")

#to get user input
st.sidebar.header("Employee Details")

# Load first
feature_columns = joblib.load("feature_columns.pkl")

def get_user_input(feature_columns):
    inputs = {}

    inputs['Age'] = st.sidebar.number_input("Age", 18, 65, 30)
    inputs['MonthlyIncome'] = st.sidebar.number_input("Monthly Income", 1000, 20000, 5000)
    inputs['JobSatisfaction'] = st.sidebar.selectbox("Job Satisfaction", [1, 2, 3, 4])
    inputs['OverTime'] = st.sidebar.selectbox("Over Time", ["Yes", "No"])
    inputs['DistanceFromHome'] = st.sidebar.number_input("Distance From Home", 0, 50, 10)

    data = {}
    for feat in feature_columns:
        if feat in inputs:
            data[feat] = inputs[feat]
        else:
            data[feat] = 0

    return pd.DataFrame(data, index=[0])


# Call function
user_input = get_user_input(feature_columns)

# Encode
user_input['OverTime'] = label_encoder.transform(
    [user_input['OverTime'][0]]
)

# Predict
if st.button("Predict Attrition"):
    prediction = model.predict(user_input)

    if prediction[0] == 1:
        st.error("The employee is likely to leave the company.")
    else:
        st.success("The employee is likely to stay with company.")
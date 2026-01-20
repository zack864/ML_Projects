import streamlit as st
import pandas as pd
import pickle
import joblib
model = joblib.load('credit_model.joblib')
with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)
st.title("Credit Risk Prediction App")
gender = st.selectbox("Gender", ["M", "F"])
car = st.selectbox("Own a Car?", ["Y", "N"])
realty = st.selectbox("Own Real Estate?", ["Y", "N"])
children = st.number_input("Number of Children", min_value=0, value=0)
income = st.number_input("Annual Income", min_value=0.0, value=50000.0)
income_type = st.selectbox("Income Type", encoders['NAME_INCOME_TYPE'].classes_)
education = st.selectbox("Education Level", encoders['NAME_EDUCATION_TYPE'].classes_)
family_status = st.selectbox("Marital Status", encoders['NAME_FAMILY_STATUS'].classes_)
housing_type = st.selectbox("Housing Type", encoders['NAME_HOUSING_TYPE'].classes_)
age_years = st.number_input("Age (Years)", min_value=18, max_value=100, value=30)
employed_years = st.number_input("Years Employed", min_value=0, max_value=50, value=5)
occupation = st.selectbox("Occupation", encoders['OCCUPATION_TYPE'].classes_)
family_members = st.number_input("Family Members", min_value=1, value=1)
input_data = pd.DataFrame({
    'CODE_GENDER': [gender],
    'FLAG_OWN_CAR': [car],
    'FLAG_OWN_REALTY': [realty],
    'CNT_CHILDREN': [children],
    'AMT_INCOME_TOTAL': [income],
    'NAME_INCOME_TYPE': [income_type],
    'NAME_EDUCATION_TYPE': [education],
    'NAME_FAMILY_STATUS': [family_status],
    'NAME_HOUSING_TYPE': [housing_type],
    'DAYS_BIRTH': [-1 * age_years * 365],
    'DAYS_EMPLOYED': [-1 * employed_years * 365],
    'FLAG_MOBIL': [1],
    'FLAG_WORK_PHONE': [1],
    'FLAG_PHONE': [1],
    'FLAG_EMAIL': [1],
    'OCCUPATION_TYPE': [occupation],
    'CNT_FAM_MEMBERS': [family_members]
})
for col, le in encoders.items():
    if col in input_data.columns:
        input_data[col] = le.transform(input_data[col].astype(str))
if st.button("Predict Risk"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    if prediction[0] == 1:
        st.error(f"High Risk! (Probability: {probability:.2%})")
    else:

        st.success(f"Low Risk - Approved (Probability: {probability:.2%})")

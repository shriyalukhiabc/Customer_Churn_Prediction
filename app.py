import streamlit as st
import pickle
import numpy as np

st.title("Customer Churn Predictor")

with open("classifier.pkl","rb") as file:
    classifier=pickle.load(file)

Age=st.number_input("Enter Age:")
MonthlyCharges=st.number_input("Enter MonthlyCharges:")
TotalCharges=st.number_input("Enter TotalCharges:")
Tenure=st.number_input("Enter Tenure:")

if st.button("predict"):
    input_data=np.array([[Age,MonthlyCharges,TotalCharges,Tenure]])
    prediction=classifier.predict(input_data)
    st.write(f"The predicted churn is: {prediction}")


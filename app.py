import streamlit as st
import pickle
import numpy as np

model = pickle.load(open(r'D:/data science/Machin learning/1. Regression/slr/linear_regression_model.pkl', 'rb'))
st.title('Salary Prediction App')

st.write("This app predicts the salary based on years of experience using a simple linear regression model.")

years_experience = st.number_input("Enter Years of Experience:", min_value=0.0, max_value=50.0, value=1.0, step=0.5)
 
if st.button("Predict Salary"):
    
    experience_input = np.array([[years_experience]])
    Prediction = model.predict(experience_input)
    
    st.success(f"The predicted salary for {years_experience} years of exprience is: ${Prediction[0]:,.2f}")
    
st.write("The model was trained using a dataset of salaries and years of exprience.")
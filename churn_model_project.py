import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv(r"D:\data science\Project\Churn Modeling\Churn_Modelling.csv")
x = df.iloc[:, 3:-1].values
y = df.iloc[:, -1].values

le = LabelEncoder()
x[:,2] = le.fit_transform(x[:, 2])

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

sc = StandardScaler()
x = sc.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)


ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=5, activation='relu'))
ann.add(tf.keras.layers.Dense(units=4, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

ann.compile(optimize='adam', loss='binary_crossentropy', metrics=['accuracy'])

ann.fit(x_train, y_train, batch_size=32, epachs=15, validation_data=(x_test, y_test))

st.title("churn prediction App")

st.sidebar.header("Input Features")
credit_score = st.sidebar.number_input("credit score", min_value=0)
geography = st.sidebar.selectbox("Geography", ("france", "germany", "spain"))
gender  = st.sidebar.selectbox("Gender", ("Female", "male"))
age = st.sidebar.number_input("age", min_value=0)
tenure = st.sidebar.number_input("Tenure", min_value=0)
balance = st.sidebar.number_input("Balance", min_value=0.0, format="%.2f")
num_of_products = st.sidebar.number_input("Number of products", min_value=1, max_value=4)
has_cr_card = st.sidebar.selectbox("Has credit card", (0, 1))
is_active_member = st.sidebar.selectbox("Is Active Member", (0, 1))
estimated_salary = st.sidebar.number_input("Estimated salary", min_value=0.0, format="%.2f")

user_data = np.array([[credit_score, geography, gender, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary]])
user_data[:, 2] = le.transform(user_data[:, 2])
user_data = np.array(ct.transform(user_data))
user_data = sc.transform(user_data)

if st.button("Predict"):
    prediction = ann.prediction(user_data)
    prediction = (prediction > 0.5)
    result = "churn" if prediction else "No churn"
    st.write(f"The Prediction is: **{result}**")
    
    y_pred = ann.predict(x_test)
    y_pred = (y_pred > 0.5)
    accuracy = accuracy_score(y_test, y_pred)
    
    y_pred = ann.predict(x_test)
    y_pred = (y_pred > 0.5)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"model Accuracy: **{accuracy:.2f}**")
    
    cm = confusion_matrix(y_test, y_pred)
    st.write("confusion matrix:")
    st.write(cm)


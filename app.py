import streamlit as st
import pandas as pd
import numpy as np
#from sklearn.model_selection import train_test_split
import joblib

def main():
    # Import the dataset
    df = pd.read_csv('new_data.csv')
    st.header("Data Sample")
    st.dataframe(df.head())

    st.header("EDA")
    st.dataframe(df.describe())

    # Load the saved model
    model = joblib.load('model.pkl')

    # Add input fields for your features and a validation button at the end of the form
    st.header('Make a prediction')

    CreditScore = st.number_input('CreditScore')
    Gender = st.selectbox('Gender', [0, 1])
    Age = st.number_input('Age')
    Tenure = st.number_input('Tenure')
    Balance = st.number_input('Balance')
    NumOfProducts = st.number_input('NumOfProducts')
    HasCrCard = st.selectbox('HasCrCard', [0, 1])
    IsActiveMember = st.selectbox('IsActiveMember', [0, 1])
    EstimatedSalary = st.number_input('EstimatedSalary')
    France = st.selectbox('France', [0, 1])
    Germany = st.selectbox('Germany', [0, 1])
    Spain = st.selectbox('Spain', [0, 1])

    if st.button('Predict'):
        # Start making predictions given the provided feature values
        features = np.array([CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, France, Germany, Spain])
        features = features.reshape(1, -1)
        prediction = model.predict(features)
        st.write('Prediction:', prediction)

if __name__ == "__main__":
    # Add sidebar with churn rate app description
    st.sidebar.title("Churn Rate App")
    st.sidebar.image('churnimg.png',  use_column_width=True)
    
    st.sidebar.write("The churn rate app is a valuable tool designed to predict customer churn, which refers to the rate at which customers discontinue their relationship with a business or service. Customer churn is a critical metric for businesses across various industries, as it directly impacts revenue and profitability. By accurately predicting churn, companies can take proactive measures to retain customers and reduce attrition.")
    main()

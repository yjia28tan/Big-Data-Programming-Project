import pandas as pd
import streamlit as st
import joblib
from xgboost import XGBRegressor
import os

# Load the trained XGBoost model
# change file path
model = joblib.load("C:\\Users\\Devus Lee\\Downloads\\xgboost_model.joblib")

def preprocess_input(order_id, total_purchase_value_per_order, payment_value, order_purchase_timestamp):
    # Preprocess 'order_purchase_timestamp'
    order_purchase_timestamp = pd.to_datetime(order_purchase_timestamp)
    order_month = order_purchase_timestamp.month
    order_hour = order_purchase_timestamp.hour

    # Other preprocessing steps if needed

    return total_purchase_value_per_order, payment_value, order_month, order_hour

def main():
    st.title('XGBoost Model Deployment')

    # Get user input
    order_id = st.text_input('Enter number of orders')
    total_purchase_value_per_order = st.number_input('Enter the total purchased price of order', step=1.0)
    payment_value = st.number_input('Enter the sum of money that spent on all purchases', step=1.0)
    order_purchase_timestamp = st.text_input('Enter the latest purchase date and time - FORMAT: YYYY-MM-DD HH:mm:ss')

    # Preprocess user input
    features = preprocess_input(order_id, total_purchase_value_per_order, payment_value, order_purchase_timestamp)

    # Make a prediction
    prediction = predict(features)

    # Display the result
    display_result(prediction)

def predict(features):
    # Make a prediction using the loaded model
    prediction = model.predict([features])[0]
    return prediction

def display_result(prediction):
    rounded_prediction = "{:.2f}".format(prediction)
    st.subheader('Prediction Result')
    st.write(f'The predicted number of products purchased is: {rounded_prediction}')

if __name__ == '__main__':
    main()
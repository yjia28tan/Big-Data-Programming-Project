# Import necessary libraries
import streamlit as st
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the trained model
model = joblib.load("C:\\Users\\User\\OneDrive\\Documents\\YiJia\\INTI\\Sem 6\\5011CEM Big Data Programming Project\\CW\\Big-Data-Programming-Project\\Code&Dataset\\cleaned dataset\\OlistSatisfaction.bkl")

def predict_satisfaction(freight_value, payment_type, payment_installments, payment_value, 
    estimated_days, arrival_days, arrival_status, seller_to_carrier_status, estimated_delivery_rate, arrival_delivery_rate, shipping_delivery_rate):

        prediction_classification = model.predict(pd.DataFrame({'freight_value' :[freight_value], 'payment_type' :[payment_type], 'payment_installments' :[payment_installments], 'payment_value' :[payment_value], 'estimated_days' :[estimated_days], 'arrival_days' :[arrival_days], 'arrival_status' :[arrival_status], 'seller_to_carrier_status' :[seller_to_carrier_status], 'estimated_delivery_rate' :[estimated_delivery_rate], 'arrival_delivery_rate' :[arrival_delivery_rate], 'shipping_delivery_rate' :[shipping_delivery_rate]}))
        return prediction_classification

        

def main():
    st.title('Customer Satisfaction Prediction')

    # User input form
    st.header("User Input Features")

    freight_value = st.text_input("Freight Value", '')
    payment_value = st.text_input("Payment Value", '')
    payment_installments = st.slider('payment_installments', 1,24,1)
    estimated_days = st.slider('estimated_days', 3,60,1)
    arrival_days = st.slider('arrival_days', 0,60,1)

    payment_type = st.selectbox("Payment Type", ['credit_card', 'boleto', 'voucher', 'debit_card'])
    seller_to_carrier_status = st.selectbox("Seller to Carrier Status", ['OnTime/Early', 'Late'])
    arrival_status = st.selectbox("Arrival Status", ['OnTime/Early', 'Late'])
    estimated_delivery_rate = st.selectbox("Estimated Delivery Rate", ['Very Slow', 'Slow', 'Neutral', 'Fast', 'Very Fast'])
    arrival_delivery_rate = st.selectbox("Arrival Delivery Rate", ['Very Slow', 'Slow', 'Neutral', 'Fast', 'Very Fast'])
    shipping_delivery_rate = st.selectbox("Shipping Delivery Rate", ['Very Slow', 'Slow', 'Neutral', 'Fast', 'Very Fast'])
    prediction_result = ''

    #   Predict Customer Satsifaction
    if st.button('Predict_Satisfaction'):
        prediction_result = predict_satisfaction(freight_value, payment_type, payment_installments, payment_value, estimated_days, arrival_days, 
                                      arrival_status, seller_to_carrier_status, estimated_delivery_rate, arrival_delivery_rate, shipping_delivery_rate)
                                    
    # Display the prediction
    if prediction_result == 0:
        prediction_result_text = 'Not Satisfied'
        st.markdown(f'<div class="not-satisfied">The Customer is {prediction_result_text} with their order.</div>', unsafe_allow_html=True)
    else:
        prediction_result_text = 'Satisfied'
        st.markdown(f'<div class="satisfied">The Customer is {prediction_result_text} with their order.</div>', unsafe_allow_html=True)

    # Add conditional CSS class based on the prediction
    st.markdown("""
        <style>
            .not-satisfied {
                background-color: #FFD6D6;
                padding: 10px;
                border-radius: 5px;
                color: #696969;
            }
            
            .satisfied {
                background-color: #D6FFDA;
                padding: 10px;
                border-radius: 5px;
                color: #696969;
            }
        </style>
    """, unsafe_allow_html=True)

    
if __name__ == '__main__':
    main()

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open(r"C:\Users\premier\OneDrive\Desktop\ML project\Churn Customer\churn_model.pkl", "rb") as file:
    model = pickle.load(file)

# Title and description
st.title("Customer Churn Prediction")
st.write("Enter customer details to predict if they are likely to churn.")

# Function to take user inputs
def user_input_features():
    gender = st.selectbox("Gender", ("Male", "Female"))
    senior_citizen = st.selectbox("Senior Citizen", ("Yes", "No"))
    partner = st.selectbox("Partner", ("Yes", "No"))
    dependents = st.selectbox("Dependents", ("Yes", "No"))
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    phone_service = st.selectbox("Phone Service", ("Yes", "No"))
    paperless_billing = st.selectbox("Paperless Billing", ("Yes", "No"))
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=500.0)
    
    # One-hot encode categorical features with more than two options (example categories)
    multiple_lines = st.selectbox("Multiple Lines", ("No phone service", "No", "Yes"))
    internet_service = st.selectbox("Internet Service", ("DSL", "Fiber optic", "No"))
    online_security = st.selectbox("Online Security", ("No internet service", "No", "Yes"))
    online_backup = st.selectbox("Online Backup", ("No internet service", "No", "Yes"))
    device_protection = st.selectbox("Device Protection", ("No internet service", "No", "Yes"))
    tech_support = st.selectbox("Tech Support", ("No internet service", "No", "Yes"))
    streaming_tv = st.selectbox("Streaming TV", ("No internet service", "No", "Yes"))
    streaming_movies = st.selectbox("Streaming Movies", ("No internet service", "No", "Yes"))
    contract = st.selectbox("Contract", ("Month-to-month", "One year", "Two year"))
    payment_method = st.selectbox("Payment Method", ("Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"))

    # Convert binary choices to 0 and 1
    binary_mapping = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    user_data = {
        "gender": binary_mapping[gender],
        "SeniorCitizen": binary_mapping[senior_citizen],
        "Partner": binary_mapping[partner],
        "Dependents": binary_mapping[dependents],
        "tenure": tenure,
        "PhoneService": binary_mapping[phone_service],
        "PaperlessBilling": binary_mapping[paperless_billing],
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "MultipleLines_No phone service": 1 if multiple_lines == "No phone service" else 0,
        "MultipleLines_No": 1 if multiple_lines == "No" else 0,
        "MultipleLines_Yes": 1 if multiple_lines == "Yes" else 0,
        # Add similar one-hot encodings for other categorical columns as necessary...
    }
    
    # Convert the input to a DataFrame for prediction
    features = pd.DataFrame(user_data, index=[0])
    return features

# Collect user input features
input_df = user_input_features()

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Display the prediction
    if prediction[0] == 1:
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is unlikely to churn.")

    # Display the probability
    st.write(f"Churn Probability: {prediction_proba[0][1]:.2f}")
    st.write(f"Retention Probability: {prediction_proba[0][0]:.2f}")

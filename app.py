
import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np

# 1. Load the model and feature config
@st.cache_resource
def load_assets():
    model = joblib.load('model.pkl')
    with open('features.json', 'r') as f:
        features = json.load(f)
    return model, features

try:
    pipeline, feature_config = load_assets()
    num_feats = feature_config['numeric_features']
    cat_feats = feature_config['categorical_features']
    all_feats = feature_config['all_features']
except FileNotFoundError:
    st.error("Error: model.pkl or features.json not found. Please ensure they are in the same directory.")
    st.stop()

# 2. App Title and Description
st.title("🍔 Food Delivery Churn Predictor")
st.markdown("Enter customer details below to predict if they are likely to churn (become inactive).")

# 3. Create Input Form
with st.form("prediction_form"):
    st.header("Customer Details")
    
    # Dynamic Dictionary to store inputs
    user_input = {}
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Numeric Inputs (Adjust min/max/value based on your data knowledge)
        user_input['total_orders'] = st.number_input("Total Orders", min_value=0, value=10)
        user_input['total_amount'] = st.number_input("Total Amount Spent", min_value=0.0, value=250.0)
        user_input['avg_order_value'] = st.number_input("Avg Order Value", min_value=0.0, value=25.0)
        user_input['avg_quantity'] = st.number_input("Avg Quantity per Order", min_value=0.0, value=1.5)
        user_input['avg_rating'] = st.slider("Average Rating", 1.0, 5.0, 4.5)
        
    with col2:
        user_input['recency_days'] = st.number_input("Days Since Last Order", min_value=0, value=7)
        user_input['tenure_days'] = st.number_input("Days Since Signup", min_value=0, value=180)
        user_input['orders_per_month'] = st.number_input("Orders Per Month", min_value=0.0, value=2.0)
        user_input['num_restaurants'] = st.number_input("Unique Restaurants Tried", min_value=1, value=3)

    st.subheader("Demographics & Behavior")
    c1, c2, c3 = st.columns(3)
    
    # Categorical Inputs (You might want to hardcode specific options if known, 
    # otherwise free text is risky. Here are assumed common values)
    with c1:
        user_input['most_common_payment'] = st.selectbox("Payment Method", ["Card", "Cash", "Wallet"])
    with c2:
        user_input['city'] = st.selectbox("City", ["Karachi", "Lahore", "Islamabad", "Multan", "Peshawar"]) 
    with c3:
        user_input['gender'] = st.selectbox("Gender", ["Male", "Female", "Other"])

    # Submit Button
    submit = st.form_submit_button("Predict Churn Status")

if submit:
    # 4. Convert inputs to DataFrame
    df_input = pd.DataFrame([user_input])
    
    # Ensure columns are in the exact order the model expects
    df_input = df_input[all_feats]
    
    # 5. Predict
    try:
        # Get probability of class 1 (Inactive/Churn)
        proba = pipeline.predict_proba(df_input)[0][1]
        prediction = pipeline.predict(df_input)[0]
        
        st.subheader("Prediction Result")
        
        # Display logic
        if prediction == 1:
            st.error(f"⚠️ High Risk of Churn! (Probability: {proba:.2%})")
            st.write("This customer is likely to become inactive.")
        else:
            st.success(f"✅ Low Risk of Churn. (Probability of churn: {proba:.2%})")
            st.write("This customer is likely to stay active.")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

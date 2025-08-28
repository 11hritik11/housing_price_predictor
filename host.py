import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the saved model and scaler
with open('housing_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('housing_scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# --- User Interface ---
st.title("California Housing Price Predictor 🏡")
st.write("Enter the details of the house to predict its price.")

# Create input fields for the user
col1, col2 = st.columns(2)
with col1:
    longitude = st.number_input("Longitude", value=-122.23)
    latitude = st.number_input("Latitude", value=37.88)
    housing_median_age = st.number_input("Housing Median Age", value=41.0)
    total_rooms = st.number_input("Total Rooms", value=880.0)
    total_bedrooms = st.number_input("Total Bedrooms", value=129.0)

with col2:
    population = st.number_input("Population", value=322.0)
    households = st.number_input("Households", value=126.0)
    median_income = st.number_input("Median Income (in tens of thousands)", value=8.3252)
    ocean_proximity = st.selectbox("Ocean Proximity",
                                   ['<1H OCEAN', 'NEAR BAY', 'INLAND', 'NEAR OCEAN', 'ISLAND'])

# --- Prediction Logic ---
if st.button("Predict Price"):
    # 1. Prepare the input data
    # Create a dictionary for ocean proximity encoding
    proximity_map = {
        '<1H OCEAN': [1, 0, 0, 0], 'INLAND': [0, 1, 0, 0],
        'ISLAND': [0, 0, 1, 0], 'NEAR BAY': [0, 0, 0, 1],
        'NEAR OCEAN': [0, 0, 0, 0] # Base case for drop_first=True
    }
    
    # Create the initial feature list based on the order from training
    input_features = [
        longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
        population, households, median_income
    ]
    input_features.extend(proximity_map[ocean_proximity])
    
    # Convert to a DataFrame to apply transformations
    feature_names = [
        'longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
        'population', 'households', 'median_income', 'ocean_proximity_<1H OCEAN',
        'ocean_proximity_INLAND', 'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY'
    ] # Note: 'NEAR OCEAN' is the dropped category
    
    input_df = pd.DataFrame([input_features], columns=feature_names)

    # 2. Apply the same transformations as in training
    # Log transform skewed features
    skewed_features = ["total_rooms", "total_bedrooms", "population", "households"]
    for feature in skewed_features:
        input_df[feature] = np.log(input_df[feature] + 1)
        
    # Feature engineering
    input_df["bedroom_ratio"] = input_df["total_bedrooms"] / input_df["total_rooms"]
    input_df["household_rooms"] = input_df["total_rooms"] / input_df["households"]

    # Reorder columns to match the training data exactly
    final_feature_order = scaler.feature_names_in_ # Get order from the fitted scaler
    input_df = input_df.reindex(columns=final_feature_order, fill_value=0)

    # 3. Scale the features
    input_scaled = scaler.transform(input_df)

    # 4. Make a prediction
    prediction = model.predict(input_scaled)

    # Display the result
    st.success(f"The predicted median house value is: ${prediction[0]:,.2f}")
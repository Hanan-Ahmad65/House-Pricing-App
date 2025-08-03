#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import joblib

# --- Load model, feature names, scaler, and scaled columns ---
with open("linear_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_names.json", "r") as f:
    feature_names = json.load(f)

with open("scaled_numerical_cols.json", "r") as f:
    scaled_cols = json.load(f)

scaler = joblib.load("scaler.pkl")

# --- Title ---
st.title("🏡 Housing Price Prediction App")

st.markdown("Adjust the features below to estimate the predicted price:")
city_code = 1001

# --- Input sliders ---
area = st.slider("Total Area (sqft)", min_value=200, max_value=200000, value=3000, step=100)
rooms = st.slider("Number of Rooms", min_value=1, max_value=100, value=4)
floors = st.slider("Number of Floors", min_value=1, max_value=100, value=1)
prev_owners = st.slider("Number of Previous Owners", 0, 5, 1)
property_age = st.slider("Property Age (Years)", 0, 100, 5)
luxury_features = st.slider("Luxury Features Count", 0, 4, 1)
city_code= city_code

# Binary checkboxes
has_yard = st.checkbox("Has Yard")
has_pool = st.checkbox("Has Pool")
is_new = st.checkbox("Is Newly Built")
storm_protector = st.checkbox("Has Storm Protector")
storage_room = st.checkbox("Has Storage Room")
guest_room = st.checkbox("Has Guest Room")

# One-hot encoding for cityPartRange
city_part = st.selectbox("City Part Range", [f"cityPartRange_{i}" for i in range(1, 11)])

# --- Build input dict ---
input_data = {
    'total_area': area,
    'numberOfRooms': rooms,
    'floors': floors,
    'cityCode': city_code,
    'numPrevOwners': prev_owners,
    'property_age': property_age,
    'luxury_features_count': luxury_features,
    'hasYard': int(has_yard),
    'hasPool': int(has_pool),
    'isNewBuilt': int(is_new),
    'hasStormProtector': int(storm_protector),
    'hasStorageRoom': int(storage_room),
    'hasGuestRoom': int(guest_room),
}

# Add all one-hot encoded cityPartRange columns as 0, then set selected to 1
for i in range(1, 11):
    col_name = f"cityPartRange_{i}"
    input_data[col_name] = 1 if city_part == col_name else 0

# --- Convert to DataFrame and reorder ---
input_df = pd.DataFrame([input_data])[feature_names]

# --- Scale numerical features ---
input_df_scaled = input_df.copy()
input_df_scaled[scaled_cols] = scaler.transform(input_df_scaled[scaled_cols])

# --- Predict ---
predicted_price = model.predict(input_df_scaled)[0]

# --- Output ---
st.subheader("🏷️ Predicted Price:")
st.success(f"${predicted_price:,.2f}")


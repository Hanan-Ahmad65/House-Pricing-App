#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and feature names
model = joblib.load('model.pkl')
feature_names = joblib.load('feature_names.pkl')

st.title("🏠 House Price Prediction App")

# Create input fields
st.header("Enter House Details:")

input_data = {}

for feature in feature_names:
    if feature.startswith('cityPartRange_'):
        continue  # skip one-hot fields, will be handled later

    if feature in ['hasYard', 'hasPool', 'isNewBuilt', 'hasStormProtector', 'hasStorageRoom', 'hasGuestRoom']:
        input_data[feature] = st.selectbox(f"{feature}:", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    elif feature in ['cityCode', 'floors', 'numPrevOwners', 'numberOfRooms', 'luxury_features_count']:
        input_data[feature] = st.number_input(f"{feature}:", step=1, format="%d")
    elif feature in ['property_age', 'total_area']:
        input_data[feature] = st.number_input(f"{feature}:", step=1.0, format="%.1f")

# City Part (one-hot encode)
city_part = st.selectbox("City Part Range:", [2, 3, 4, 5, 6, 7, 8, 9, 10])
for i in range(2, 11):
    col_name = f'cityPartRange_{i}'
    input_data[col_name] = 1 if i == city_part else 0

# Predict
if st.button("Predict Price"):
    try:
        input_df = pd.DataFrame([input_data])[feature_names]
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Predicted House Price: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")


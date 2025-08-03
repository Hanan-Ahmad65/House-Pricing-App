#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# app.py 
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))  # Make sure this file exists in your GitHub repo

# App title
st.title('🏠 House Price Prediction App')

# Sidebar for user input
st.sidebar.header('User Input Features')

def user_input_features():
    bedrooms = st.sidebar.slider('Bedrooms', 1, 10, 3)
    bathrooms = st.sidebar.slider('Bathrooms', 1, 5, 2)
    sqft = st.sidebar.number_input('Square Footage', 500, 10000, 1500)
    floors = st.sidebar.selectbox('Number of Floors', [1, 2, 3])
    age = st.sidebar.slider('Age of House (years)', 0, 100, 10)

    data = {
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft': sqft,
        'floors': floors,
        'age': age
    }
    return pd.DataFrame([data])

# Get user input
input_df = user_input_features()

# Display input
st.subheader('User Input Features')
st.write(input_df)

# Prediction
prediction = model.predict(input_df)
st.subheader('Predicted Price')
st.write(f"${prediction[0]:,.2f}")


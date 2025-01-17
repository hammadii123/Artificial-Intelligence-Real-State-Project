
# 3)))
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


model = pickle.load(open('LinearRegression.pkl', 'rb'))


df_cleaned = pd.read_csv('Cleaned_Data.csv')


df_cleaned['Address_Code'] = df_cleaned['Address'].astype('category').cat.codes
address_mapping = dict(enumerate(df_cleaned['Address'].astype('category').cat.categories))

# Extract unique areas for dropdown
unique_areas = df_cleaned['Address'].unique()

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
        }
        .title {
            text-align: center;
            color: #e1dada;
            font-size: 36px;
            font-weight: bold;
        }
        .prediction-container {
            background-color: #f1c40f;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        h2 {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Real Estate Price Prediction</div>', unsafe_allow_html=True)

# Area selection
st.subheader("Select Area:")
selected_area = st.selectbox("Choose an Area", options=unique_areas)
selected_area_code = df_cleaned[df_cleaned['Address'] == selected_area]['Address_Code'].iloc[0]

# Property type selection: House or Flat
st.subheader("Select Property Type:")
property_type = st.radio("Choose Property Type:", options=["House", "Flat"])

# Floor selection for Flat
floor = 0
if property_type == "Flat":
    st.subheader("Select Floor:")
    floor = st.slider("Choose Floor (1-10):", 1, 10, 1)

# User input for bedrooms and area (bathrooms are auto-set equal to bedrooms)
st.subheader("Enter the details for prediction:")
bedrooms = st.slider('No. of Bedrooms', 1, 10, 3)
bathrooms = bedrooms  
area = st.slider('Area (in square yards)', 50, 1000, 200)

# Make a prediction
input_data = pd.DataFrame([[selected_area_code, bedrooms, bathrooms, area]], 
                          columns=['Address_Code', 'NoOfBedrooms', 'NoOfBathrooms', 'AreaSqYards'])
predicted_price = model.predict(input_data)

# Adjust price for flats based on property type and floor
if property_type == "Flat":
    flat_base_discount = 0.50  # 50% reduction for flats
    floor_discount = (floor - 1) * 0.03  # 3% decrease per floor above ground
    adjusted_price = predicted_price[0] * (1 - flat_base_discount - floor_discount)
else:
    adjusted_price = predicted_price[0]  # For houses, price remains the same

# Display the result
st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
st.markdown(f'<h2>Predicted Price for {property_type} in {selected_area}: </h2>', unsafe_allow_html=True)
st.markdown(f'<h2>Rs.{adjusted_price+20000000:,.2f}</h2>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)



# Content-based Recommendation System

# Preprocess the input data to calculate similarity
user_input = pd.DataFrame([[selected_area, bedrooms, bathrooms, area, adjusted_price]], 
                          columns=['Address', 'NoOfBedrooms', 'NoOfBathrooms', 'AreaSqYards', 'Price'])

# Combine user input with the dataset for similarity comparison
df_cleaned['Price'] = df_cleaned['Price']  # Keep price as it is
similarity_df = pd.concat([df_cleaned, user_input], ignore_index=True)

# Calculate cosine similarity
similarity_features = similarity_df[['NoOfBedrooms', 'NoOfBathrooms', 'AreaSqYards', 'Price']]
cos_sim = cosine_similarity(similarity_features)

# Extract similarity scores for the last row (user input)
similarity_scores = cos_sim[-1, :-1]  

# Get the indices of the most similar properties (top 2)
most_similar_indices = similarity_scores.argsort()[-3:-1][::-1] 


recommended_properties = df_cleaned.iloc[most_similar_indices]
st.subheader("Recommended Properties Based on Your Input:")
st.write(recommended_properties[['Address', 'Price']])

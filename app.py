# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Load the preprocessed training data
@st.cache
def load_data():
    df = pd.read_csv('./src/cleaned_train.csv')
    
    # Only drop 'Id' if it exists in the dataframe
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])
    
    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice']
    return X, y

# Train the model
@st.cache
def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Streamlit app for house price prediction
st.title("House Price Prediction App")

# Load the data and train the model
X_train, y_train = load_data()
model = train_model(X_train, y_train)

# Sidebar for input features
st.sidebar.header("Input Features")

# Create input fields dynamically based on the training features
input_data = {}
for feature in X_train.columns:
    # You can adjust the input type based on the feature (e.g., numeric, slider)
    input_data[feature] = st.sidebar.number_input(f"Input {feature}", min_value=0, value=1000, step=50)

# When the user clicks the "Predict" button
if st.sidebar.button("Predict House Price"):
    input_df = pd.DataFrame([input_data])

    # Make the prediction
    prediction = model.predict(input_df)[0]
    
    # Display the predicted house price
    st.write(f"The predicted house price is: ${prediction:,.2f}")

# Model information
st.subheader("Model Information")
st.write("This model was trained using RandomForestRegressor.")

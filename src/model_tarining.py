# model_tuning.py

# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the preprocessed training data
@st.cache
def load_data():
    df = pd.read_csv('cleaned_train.csv')
    
    # Only drop 'Id' if it exists in the dataframe
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])
    
    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice']
    return X, y

# Train the tuned model
@st.cache
def train_model(X, y):
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Use the best hyperparameters found from RandomizedSearchCV
    best_params = {
        'n_estimators': 100,
        'max_depth': 40,
        'min_samples_split': 2,
        'min_samples_leaf': 2,
        'bootstrap': True
    }

    # Train the RandomForestRegressor with the best hyperparameters
    model = RandomForestRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return model, rmse

# Streamlit app for house price prediction
st.title("House Price Prediction App")

# Load the data and train the model
X_train, y_train = load_data()
model, rmse = train_model(X_train, y_train)

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

# Display the RMSE of the model
st.subheader("Model Evaluation")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Model information
st.subheader("Model Information")
st.write("This model was trained using RandomForestRegressor with hyperparameter tuning.")

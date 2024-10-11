# House Price Prediction App

This project is a **machine learning** application built using **Streamlit** and **RandomForestRegressor**. It predicts house prices based on various features such as the number of bedrooms, living area square footage, and other relevant data from the Ames Housing Dataset.

The app is deployed on **Streamlit Cloud** and uses a **tuned RandomForestRegressor** for predicting the house prices. The project includes data cleaning, preprocessing, hyperparameter tuning, and model evaluation using Root Mean Squared Error (RMSE).

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Steps](#steps)
  - [1. Data Cleaning and Preprocessing](#1-data-cleaning-and-preprocessing)
  - [2. Feature Engineering](#2-feature-engineering)
  - [3. Model Training and Hyperparameter Tuning](#3-model-training-and-hyperparameter-tuning)
  - [4. Streamlit App Development](#4-streamlit-app-development)
  - [5. Deployment to Streamlit Cloud](#5-deployment-to-streamlit-cloud)
- [Usage](#usage)

## Overview

This project demonstrates the full lifecycle of a machine learning project from data cleaning and preprocessing to model tuning and deployment. It predicts house prices using the Ames Housing dataset, and the model is deployed as a web application using **Streamlit Cloud**.

## Features

- Data preprocessing and feature engineering.
- Hyperparameter tuning using **RandomizedSearchCV** for model optimization.
- **RandomForestRegressor** model to predict house prices.
- RMSE (Root Mean Squared Error) for model evaluation.
- Deployed as an interactive app using **Streamlit**.

## Installation

To run this project locally, follow these steps:

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/your-repo-name.git

  Navigate into the project directory:

2. Navigate into the project directory:
    ```bash
    cd your-repo-name
3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    
4. Run the app locally using Streamlit:
    ```bash
    streamlit run app.py


  ## Project Structure 
   ```bash
  .
  ├── app.py                     # Main Streamlit app script
  ├── data_cleaning_preprocessing.py  # Script for cleaning and preprocessing the data
  ├── model_tuning.py            # Script for model training and hyperparameter tuning
  ├── cleaned_train.csv          # Cleaned training dataset
  ├── cleaned_test.csv           # Cleaned test dataset
  ├── requirements.txt           # Python dependencies
  └── README.md                  # Project documentation (this file)







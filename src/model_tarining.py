# model_tuning.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error

# Load the cleaned dataset
df = pd.read_csv('./cleaned_train.csv')

# Drop the 'Id' column if it exists
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

# Define features and target
X = df.drop(columns=['SalePrice'])
y = df['SalePrice']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize the RandomForestRegressor model
rf = RandomForestRegressor(random_state=42)

# Set up RandomizedSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid,
                                   n_iter=50, cv=3, verbose=2, random_state=42,
                                   n_jobs=-1, scoring='neg_mean_squared_error')

# Fit the model
random_search.fit(X_train, y_train)

# Best hyperparameters
best_params = random_search.best_params_
print(f"Best Hyperparameters: {best_params}")

# Predict on the test set using the best model
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error: {rmse}")

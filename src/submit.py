# submission.py

# submit.py

# submit.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load test data
test_df = pd.read_csv('./cleaned_test.csv')

# Extract the 'Id' column for submission, but don't use it for predictions
test_ids = test_df['Id']

# Drop 'Id' from the test data (since it's not a feature)
X_test = test_df.drop('Id', axis=1)

# Load the trained model
model = LinearRegression()  # Example: Replace with your actual trained model

# Load training data (in case you need to retrain the model)
train_df = pd.read_csv('./cleaned_train.csv')

# Prepare training data
X_train = train_df.drop(['SalePrice', 'Id'], axis=1)  # Drop 'Id' from training data as well
y_train = train_df['SalePrice']

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# If you applied log transformation earlier, reverse the log transformation
predictions = np.expm1(predictions)  # If your model was trained on log-transformed SalePrice

# Create a submission DataFrame
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': predictions
})

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")

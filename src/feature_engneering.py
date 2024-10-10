# feature_engineering.py
# feature_engineering.py

import pandas as pd

def engineer_features(filepath):
    # Load the cleaned dataset
    df = pd.read_csv(filepath)
    
    # Check the columns available in the dataset
    print("Columns in the dataset:", df.columns)

    # Ensure 'FullBath' and 'HalfBath' exist before performing feature engineering
    if 'FullBath' not in df.columns or 'HalfBath' not in df.columns:
        raise KeyError("One of the required columns ('FullBath' or 'HalfBath') is missing")

    # Feature Engineering: Combine FullBath and HalfBath into TotalBath
    df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath']
    
    # Drop the original bathroom columns
    df.drop(['FullBath', 'HalfBath'], axis=1, inplace=True)
    
    return df

if __name__ == "__main__":
    # Load the cleaned dataset and apply feature engineering
    engineered_data = engineer_features('./cleaned_train.csv')
    
    # Save the data with engineered features
    engineered_data.to_csv('engineered_data.csv', index=False)
    print("Feature engineering completed and saved as 'engineered_data.csv'")

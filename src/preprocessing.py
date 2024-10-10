# data_cleaning_preprocessing.py

# data_cleaning_preprocessing.py

import pandas as pd

def load_and_clean_data(filepath, is_train=True):
    """
    Load and clean the data. For training data, remove the SalePrice column.
    is_train: Boolean flag to indicate if the data is training data or test data.
    """
    # Load the data
    df = pd.read_csv(filepath)

    # Drop irrelevant or sparse columns
    irrelevant_columns = ['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature']
    df = df.drop(columns=irrelevant_columns)

    # Handle missing values
    # Fill numerical missing values with median
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

    # Fill categorical missing values with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

    # One-hot encoding for categorical variables
    df = pd.get_dummies(df, drop_first=True)

    if is_train:
        return df
    else:
        return df

if __name__ == "__main__":
    # Clean the training data
    train_df_cleaned = load_and_clean_data('../data/train.csv')
    train_df_cleaned.to_csv('cleaned_train.csv', index=False)

    # Clean the test data (without SalePrice)
    test_df_cleaned = load_and_clean_data('../data/test.csv', is_train=False)
    test_df_cleaned.to_csv('cleaned_test.csv', index=False)

    print("Data cleaned and saved to 'cleaned_train.csv' and 'cleaned_test.csv'.")

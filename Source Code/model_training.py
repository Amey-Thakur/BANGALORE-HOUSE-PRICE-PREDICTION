"""
Project: Bangalore House Price Prediction
Description: comprehensive pipeline for training the property price prediction model.
             This script encompasses data ingestion, cleaning, feature engineering, 
             outlier detection, and final model serialization.

Authors:
  - Amey Thakur (https://github.com/Amey-Thakur)
  - Mega Satish (https://github.com/msatmod)

Repository: https://github.com/Amey-Thakur/BANGALORE-HOUSE-PRICE-PREDICTION
Release Date: August 7, 2021
License: MIT License
"""

# ------------------------------------------------------------------------------
# Scholarly References & Technical Context
# ------------------------------------------------------------------------------
# The methodology implemented herein follows standard Data Science practices:
# 1. Data Cleaning: Handling missing values and inconsistencies (e.g., usage of 'mean' strategy).
# 2. Feature Engineering: Dimensionality reduction and deriving new metrics (price per sqft).
# 3. Outlier Removal: Using statistical heuristics (standard deviation) to filter anomalies.
# 4. One-Hot Encoding: Transforming categorical variables (Location) into numerical format.
# 5. Model Training: Ordinary Least Squares (OLS) regression via Scikit-Learn.
# ------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import pickle
import json

# ------------------------------------------------------------------------------
# 1. Data Ingestion
# ------------------------------------------------------------------------------
def load_data(filepath):
    """
    Loads the empirical dataset from a CSV file.
    
    Args:
        filepath (str): Relative path to the CSV dataset.
        
    Returns:
        DataFrame: A pandas DataFrame containing the raw housing data.
    """
    df = pd.read_csv(filepath)
    print(f"Dataset successfully loaded with shape: {df.shape}")
    return df

# ------------------------------------------------------------------------------
# 2. Data Preprocessing & Cleaning
# ------------------------------------------------------------------------------
def preprocess_data(df):
    """
    Executes the data cleaning pipeline:
    - Drops irrelevant columns not contributing to price variance (e.g., 'area_type', 'society').
    - Handles missing values by dropping null rows to maintain data integrity.
    - Standardizes the 'size' feature into a numerical 'bhk' (Bedroom, Hall, Kitchen) metric.
    - Normalizes non-standard 'total_sqft' entries (ranges) into single float values.
    """
    # Dropping columns with low predictive power for this specific regression task.
    df2 = df.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')
    
    # Dropping rows with missing values (Null/NaN) to ensure model stability.
    df3 = df2.dropna()
    
    # Feature Extraction: Deriving 'bhk' from 'size' string (e.g., "2 BHK" -> 2).
    df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
    
    def convert_sqft_to_num(x):
        """Helper to convert range strings (e.g., '1000-1200') to their mean."""
        tokens = x.split('-')
        if len(tokens) == 2:
            return (float(tokens[0]) + float(tokens[1])) / 2
        try:
            return float(x)
        except:
            return None

    # Applying the conversion logic to 'total_sqft'.
    df4 = df3.copy()
    df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
    df4 = df4.dropna() # Drop any rows where sqft could not be parsed
    
    return df4

# ------------------------------------------------------------------------------
# 3. Feature Engineering & Outlier Removal
# ------------------------------------------------------------------------------
def remove_outliers(df):
    """
    Implements domain-specific heuristics to remove statistical outliers:
    - Price per sqft thresholding.
    - Standard Deviation filtering for price per sqft within locations.
    - BHK vs Sqft logic (ensuring appropriate square footage per bedroom).
    """
    # Creating a new feature 'price_per_sqft' for statistical analysis.
    df5 = df.copy()
    df5['price_per_sqft'] = df5['price'] * 100000 / df5['total_sqft']
    
    # Dimensionality Reduction: Grouping rare locations into 'other'.
    # This creates a robust 'Location' feature without exploding the feature space.
    location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
    location_stats_less_than_10 = location_stats[location_stats <= 10]
    df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
    
    # Heuristic 1: Removing properties with unrealistic sqft per bedroom (< 300).
    df6 = df5[~(df5.total_sqft / df5.bhk < 300)]
    
    # Heuristic 2: Removing extreme price_per_sqft outliers using Mean and Std Dev.
    def remove_pps_outliers(df):
        df_out = pd.DataFrame()
        for key, subdf in df.groupby('location'):
            m = np.mean(subdf.price_per_sqft)
            st = np.std(subdf.price_per_sqft)
            reduced_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
            df_out = pd.concat([df_out, reduced_df], ignore_index=True)
        return df_out

    df7 = remove_pps_outliers(df6)
    
    # Heuristic 3: Removing anomalies where 2 BHK costs more than 3 BHK in same area.
    def remove_bhk_outliers(df):
        exclude_indices = np.array([])
        for location, location_df in df.groupby('location'):
            bhk_stats = {}
            for bhk, bhk_df in location_df.groupby('bhk'):
                bhk_stats[bhk] = {
                    'mean': np.mean(bhk_df.price_per_sqft),
                    'std': np.std(bhk_df.price_per_sqft),
                    'count': bhk_df.shape[0]
                }
            for bhk, bhk_df in location_df.groupby('bhk'):
                stats = bhk_stats.get(bhk - 1)
                if stats and stats['count'] > 5:
                    exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values)
        return df.drop(exclude_indices, axis='index')

    df8 = remove_bhk_outliers(df7)
    
    # Final cleanup: Dropping features used only for filtering.
    df9 = df8.drop(['price_per_sqft'], axis='columns')
    
    return df9

# ------------------------------------------------------------------------------
# 4. Model Training & Serialization
# ------------------------------------------------------------------------------
def train_model(df):
    """
    Training the Linear Regression model:
    - One-Hot Encodes the Categorical 'location' feature.
    - Splits the dataset into Training (X) and Target (y) variables.
    - Fits the Ordinary Least Squares (OLS) model.
    - Serializes the trained model and column structure for production use.
    """
    # One-Hot Encoding for Location.
    dummies = pd.get_dummies(df.location)
    df10 = pd.concat([df, dummies.drop('other', axis='columns')], axis='columns')
    df11 = df10.drop('location', axis='columns')
    
    X = df11.drop('price', axis='columns')
    y = df11.price
    
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, y)
    
    print(f"Model Training Complete. Coefficients: {len(model.coef_)}")
    
    # Serialization: Saving the model artifact.
    with open('bangalore_home_prices_model.pickle', 'wb') as f:
        pickle.dump(model, f)
    print("Model serialized to 'bangalore_home_prices_model.pickle'.")

    # Serialization: Saving the Data Columns (Schema).
    columns = {
        'data_columns': [col.lower() for col in X.columns]
    }
    with open("columns.json", "w") as f:
        f.write(json.dumps(columns))
    print("Column schema saved to 'columns.json'.")

if __name__ == "__main__":
    # Note: Ensure the dataset is present at the specified path.
    # dataset_path = "Bengaluru_House_Data.csv" 
    # df = load_data(dataset_path)
    # df_processed = preprocess_data(df)
    # df_clean = remove_outliers(df_processed)
    # train_model(df_clean)
    print("Script initialized. Uncomment execution lines to run training.")

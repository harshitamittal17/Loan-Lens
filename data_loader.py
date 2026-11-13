"""
data_loader.py
Module for loading, cleaning, and validating the HDFC loan approval dataset.
"""

import pandas as pd
import numpy as np


def load_and_clean_data(filepath: str = "loan_approval_dataset.csv") -> pd.DataFrame:
    """
    Loads the loan approval dataset and applies data cleaning:
    1. Trims whitespace from column names and categorical values.
    2. Clips negative values in residential_assets_value to zero (accounting for data entry anomalies).
    3. Drops non-predictive identifier column 'loan_id'.
    4. Encodes target 'loan_status' into binary (1 = Approved, 0 = Rejected).
    
    Args:
        filepath (str): Path to CSV dataset.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df = pd.read_csv(filepath)
    
    # 1. Clean column headers
    df.columns = df.columns.str.strip()
    
    # 2. Clean string values in object/string columns
    for col in df.select_dtypes(include=['object', 'string', 'str']).columns:
        df[col] = df[col].astype(str).str.strip()
        
    # 3. Clean negative asset entries (e.g. residential_assets_value has negative entries)
    if 'residential_assets_value' in df.columns:
        df['residential_assets_value'] = df['residential_assets_value'].clip(lower=0)
        
    # 4. Drop loan_id if present
    if 'loan_id' in df.columns:
        df = df.drop(columns=['loan_id'])
        
    # 5. Encode target if present
    if 'loan_status' in df.columns:
        df['loan_status'] = df['loan_status'].map({'Approved': 1, 'Rejected': 0})
        
    return df


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Returns summary statistics, class balance, and missing value counts.
    """
    summary = {
        "shape": df.shape,
        "missing_values": df.isnull().sum().to_dict(),
        "class_distribution": (
            df['loan_status'].value_counts(normalize=True).to_dict() 
            if 'loan_status' in df.columns else {}
        ),
        "columns": df.columns.tolist()
    }
    return summary


if __name__ == "__main__":
    df = load_and_clean_data()
    summary = get_dataset_summary(df)
    print("Dataset successfully loaded and cleaned.")
    print("Dimensions:", summary["shape"])
    print("Columns:", summary["columns"])
    print("Class Balance (1=Approved, 0=Rejected):", summary["class_distribution"])

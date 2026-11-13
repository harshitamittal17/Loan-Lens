"""
feature_engineering.py
Module for deriving domain-specific banking and financial indicators
from applicant profiles for HDFC Bank Ltd. credit risk modeling.
"""

import pandas as pd
import numpy as np


def engineer_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes key financial indicators and debt service ratios:
    1. total_assets_value: Combined sum of residential, commercial, luxury, and liquid bank assets.
    2. asset_to_loan_ratio: Collateral coverage ratio (total assets / requested loan amount).
    3. loan_to_income_ratio: Leverage ratio (loan amount / annual income).
    4. annual_debt_service_ratio: Estimated annual principal repayment burden relative to annual income.
    5. liquid_to_loan_ratio: Instant liquidity safety margin (bank liquid assets / loan amount).
    6. cibil_risk_category: Credit rating tiers (Poor, Fair, Good, Excellent).
    
    Args:
        df (pd.DataFrame): Cleaned applicant DataFrame.
        
    Returns:
        pd.DataFrame: DataFrame enriched with financial indicators.
    """
    data = df.copy()
    eps = 1e-6  # small constant to prevent division by zero
    
    # 1. Total Assets Calculation
    asset_cols = [
        'residential_assets_value',
        'commercial_assets_value',
        'luxury_assets_value',
        'bank_asset_value'
    ]
    data['total_assets_value'] = data[asset_cols].sum(axis=1)
    
    # 2. Asset to Loan Ratio (Collateral Coverage)
    data['asset_to_loan_ratio'] = data['total_assets_value'] / (data['loan_amount'] + eps)
    
    # 3. Loan to Income Ratio (Financial Leverage)
    data['loan_to_income_ratio'] = data['loan_amount'] / (data['income_annum'] + eps)
    
    # 4. Annual Debt Service Ratio (Annualized EMI burden vs Income)
    data['annual_debt_service_ratio'] = (data['loan_amount'] / data['loan_term']) / (data['income_annum'] + eps)
    
    # 5. Liquid Assets to Loan Ratio (Immediate Cash Cushion)
    data['liquid_to_loan_ratio'] = data['bank_asset_value'] / (data['loan_amount'] + eps)
    
    # 6. CIBIL Score Categorical Tiers
    # Standard Indian Credit Bureau (CIBIL TransUnion) rating brackets:
    # 300-549: Poor | 550-649: Fair | 650-749: Good | 750-900: Excellent
    data['cibil_risk_tier'] = pd.cut(
        data['cibil_score'],
        bins=[299, 549, 649, 749, 901],
        labels=['Poor', 'Fair', 'Good', 'Excellent']
    ).astype(str)
    
    return data


if __name__ == "__main__":
    from data_loader import load_and_clean_data
    raw = load_and_clean_data()
    featured = engineer_financial_features(raw)
    print("Feature engineering complete.")
    print("New columns:", [c for c in featured.columns if c not in raw.columns])
    print(featured[['asset_to_loan_ratio', 'loan_to_income_ratio', 'annual_debt_service_ratio', 'cibil_risk_tier']].head())

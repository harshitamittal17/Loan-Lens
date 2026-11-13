"""
predictor.py
Interactive and batch inference module for HDFC Bank Ltd. loan applications.
Provides end-to-end decisioning, approval probabilities, and credit risk factors.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Union
from src.feature_engineering import engineer_financial_features
from src.model_trainer import load_model


class HDFCLoanPredictor:
    """
    Production-ready loan underwriting prediction engine for HDFC Bank.
    """
    def __init__(self, model_path: str = "models/hdfc_loan_model.joblib"):
        self.model_path = model_path
        self.pipeline = load_model(model_path)
        
    def predict_applicant(self, applicant_data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Takes raw applicant profile, applies transformations, and generates
        a credit decision with explanatory risk analysis.
        
        Expected fields:
            - no_of_dependents (int)
            - education (str: 'Graduate' or 'Not Graduate')
            - self_employed (str: 'Yes' or 'No')
            - income_annum (float/int)
            - loan_amount (float/int)
            - loan_term (int in years)
            - cibil_score (int: 300 to 900)
            - residential_assets_value (float/int)
            - commercial_assets_value (float/int)
            - luxury_assets_value (float/int)
            - bank_asset_value (float/int)
        """
        if isinstance(applicant_data, dict):
            df = pd.DataFrame([applicant_data])
        else:
            df = applicant_data.copy()
            
        # Strip strings
        for col in df.select_dtypes(include=['object', 'string', 'str']).columns:
            df[col] = df[col].astype(str).str.strip()
            
        # Clip negative residential asset values
        if 'residential_assets_value' in df.columns:
            df['residential_assets_value'] = df['residential_assets_value'].clip(lower=0)
            
        # Remove loan_id if passed
        if 'loan_id' in df.columns:
            df = df.drop(columns=['loan_id'])
            
        # Derive domain features
        df_featured = engineer_financial_features(df)
        
        # Predict class and probability
        pred_label = self.pipeline.predict(df_featured)[0]
        decision = "Approved" if pred_label == 1 else "Rejected"
        
        prob_approved = None
        prob_rejected = None
        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(df_featured)[0]
            # classes_ are [0, 1]
            prob_rejected = float(probs[0])
            prob_approved = float(probs[1])
            
        # Risk factor diagnostics
        row = df_featured.iloc[0]
        risk_factors = []
        if row['cibil_score'] < 550:
            risk_factors.append(f"Sub-prime CIBIL score ({row['cibil_score']} < 550 threshold).")
        if row['asset_to_loan_ratio'] < 1.0:
            risk_factors.append(f"Under-collateralized loan (Asset/Loan ratio = {row['asset_to_loan_ratio']:.2f} < 1.0).")
        if row['loan_to_income_ratio'] > 5.0:
            risk_factors.append(f"Excessive financial leverage (Loan/Income ratio = {row['loan_to_income_ratio']:.2f} > 5.0x).")
        if row['annual_debt_service_ratio'] > 0.5:
            risk_factors.append(f"High annual debt service burden ({row['annual_debt_service_ratio']:.1%} of income).")
            
        positive_factors = []
        if row['cibil_score'] >= 700:
            positive_factors.append(f"Prime CIBIL credit score ({row['cibil_score']}).")
        if row['asset_to_loan_ratio'] >= 2.0:
            positive_factors.append(f"Strong collateral coverage (Asset/Loan ratio = {row['asset_to_loan_ratio']:.2f}x).")
        if row['liquid_to_loan_ratio'] >= 0.3:
            positive_factors.append(f"Healthy liquid bank buffer ({row['liquid_to_loan_ratio']:.1%} of loan amount).")
            
        return {
            "Decision": decision,
            "Approval_Probability": prob_approved,
            "Rejection_Probability": prob_rejected,
            "CIBIL_Score": int(row['cibil_score']),
            "CIBIL_Tier": row['cibil_risk_tier'],
            "Collateral_Coverage_Ratio": float(row['asset_to_loan_ratio']),
            "Loan_To_Income_Ratio": float(row['loan_to_income_ratio']),
            "Risk_Factors": risk_factors,
            "Positive_Factors": positive_factors
        }

"""
main.py
End-to-End Execution Pipeline for HDFC Bank Ltd. Loan Approval Classification.

Fulfills all requirements from the problem statement:
- Predicts whether a loan should be Approved or Rejected based on applicant details
  (income, loan amount, credit history, employment status, property assets, etc.).
- Evaluates multiple supervised ML classification algorithms.
- Performs domain-specific financial indicator feature engineering.
- Saves model artifacts and diagnostic visual charts.
- Runs inference on diverse applicant profiles.
"""

import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import load_and_clean_data, get_dataset_summary
from src.feature_engineering import engineer_financial_features
from src.model_trainer import (
    build_preprocessor,
    train_and_cross_validate,
    tune_champion_model,
    save_model
)
from src.evaluator import (
    evaluate_models,
    plot_confusion_matrices,
    plot_roc_curves,
    get_feature_importances
)
from src.predictor import HDFCLoanPredictor


def run_pipeline():
    print("=" * 80)
    print("   HDFC BANK LTD. - LOAN APPROVAL PREDICTION ML SYSTEM")
    print("=" * 80)
    
    # 1. Ingestion & Data Cleaning
    print("\n[Step 1] Loading and cleaning dataset...")
    df = load_and_clean_data("loan_approval_dataset.csv")
    summary = get_dataset_summary(df)
    print(f"-> Loaded {summary['shape'][0]} applications across {summary['shape'][1]} features.")
    print(f"-> Target distribution: {summary['class_distribution']}")
    
    # 2. Financial Feature Engineering
    print("\n[Step 2] Engineering domain-specific banking ratios & financial indicators...")
    df_featured = engineer_financial_features(df)
    print(f"-> Feature set expanded to {df_featured.shape[1]} columns.")
    
    # 3. Train-Test Split (Holdout Validation)
    print("\n[Step 3] Splitting data (80% Train, 20% Test) with stratification...")
    X = df_featured.drop(columns=['loan_status'])
    y = df_featured['loan_status']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"-> Training Set: {X_train.shape[0]} samples | Test Set: {X_test.shape[0]} samples")
    
    # 4. Pipeline & Model Training
    print("\n[Step 4] Building pipelines & training 6 supervised classification algorithms...")
    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)
    trained_models = train_and_cross_validate(X_train, y_train, preprocessor, cv_folds=5)
    
    # 5. Benchmark Evaluation
    print("\n[Step 5] Benchmarking performance metrics on unseen holdout test set...")
    metrics_df = evaluate_models(trained_models, X_test, y_test)
    print("\n" + metrics_df.to_string(index=False))
    
    # 6. Diagnostic Visualizations
    print("\n[Step 6] Generating confusion matrices & ROC curves...")
    os.makedirs("models", exist_ok=True)
    plot_confusion_matrices(trained_models, X_test, y_test, save_path="models/confusion_matrices.png")
    plot_roc_curves(trained_models, X_test, y_test, save_path="models/roc_curves.png")
    
    # 7. Champion Model Tuning & Feature Importance
    print("\n[Step 7] Hyperparameter tuning champion model (Random Forest via GridSearchCV)...")
    best_pipe, best_params, best_auc = tune_champion_model(X_train, y_train, preprocessor)
    print(f"-> Best Hyperparameters: {best_params}")
    print(f"-> Best Cross-Validation ROC-AUC: {best_auc:.4f}")
    
    # Feature Importance
    feat_imp = get_feature_importances(best_pipe, X_train)
    print("\nTop 5 Most Influential Predictive Financial Indicators:")
    print(feat_imp.head(5).to_string(index=False))
    
    # Save Model
    save_model(best_pipe, "models/hdfc_loan_model.joblib")
    
    # 8. Interactive Inference on Realistic Customer Profiles
    print("\n[Step 8] Evaluating Inference Engine on 3 Representative HDFC Applicant Personas:")
    predictor = HDFCLoanPredictor("models/hdfc_loan_model.joblib")
    
    personas = [
        {
            "name": "Applicant A (Prime Credit Profile)",
            "data": {
                "no_of_dependents": 1,
                "education": "Graduate",
                "self_employed": "No",
                "income_annum": 8500000,
                "loan_amount": 15000000,
                "loan_term": 10,
                "cibil_score": 790,
                "residential_assets_value": 12000000,
                "commercial_assets_value": 6000000,
                "luxury_assets_value": 18000000,
                "bank_asset_value": 7500000
            }
        },
        {
            "name": "Applicant B (Subprime / High Default Risk)",
            "data": {
                "no_of_dependents": 4,
                "education": "Not Graduate",
                "self_employed": "Yes",
                "income_annum": 2200000,
                "loan_amount": 18000000,
                "loan_term": 20,
                "cibil_score": 380,
                "residential_assets_value": 1500000,
                "commercial_assets_value": 0,
                "luxury_assets_value": 1200000,
                "bank_asset_value": 800000
            }
        },
        {
            "name": "Applicant C (Moderate Credit with High Asset Backing)",
            "data": {
                "no_of_dependents": 2,
                "education": "Graduate",
                "self_employed": "Yes",
                "income_annum": 6000000,
                "loan_amount": 10000000,
                "loan_term": 12,
                "cibil_score": 620,
                "residential_assets_value": 9000000,
                "commercial_assets_value": 5000000,
                "luxury_assets_value": 12000000,
                "bank_asset_value": 4500000
            }
        }
    ]
    
    for p in personas:
        result = predictor.predict_applicant(p['data'])
        print(f"\n--- {p['name']} ---")
        print(f"Decision: {result['Decision']}")
        print(f"Approval Probability: {result['Approval_Probability']:.2%}")
        print(f"Rejection Probability: {result['Rejection_Probability']:.2%}")
        print(f"CIBIL Score: {result['CIBIL_Score']} ({result['CIBIL_Tier']})")
        print(f"Collateral Coverage Ratio: {result['Collateral_Coverage_Ratio']:.2f}x")
        print(f"Positive Factors: {result['Positive_Factors']}")
        print(f"Risk Factors: {result['Risk_Factors']}")

    print("\n" + "=" * 80)
    print("   EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()

"""
evaluator.py
Module for comprehensive model evaluation, metric calculations,
confusion matrix analysis, ROC curve visualizations, and feature importance.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)


def evaluate_models(
    trained_models: Dict[str, Dict[str, Any]],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Computes Accuracy, Precision, Recall, F1-Score, and ROC-AUC on hold-out test set
    for all candidate models.
    """
    records = []
    for name, item in trained_models.items():
        pipe = item['pipeline']
        y_pred = pipe.predict(X_test)
        
        # Predict probabilities for ROC-AUC
        if hasattr(pipe, "predict_proba"):
            y_prob = pipe.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_prob)
        else:
            roc_auc = np.nan
            
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        records.append({
            'Model': name,
            'CV Accuracy (Mean)': item['cv_mean'],
            'CV Std Dev': item['cv_std'],
            'Test Accuracy': accuracy_score(y_test, y_pred),
            'Precision (Approved)': precision_score(y_test, y_pred, zero_division=0),
            'Recall (Approved)': recall_score(y_test, y_pred, zero_division=0),
            'Specificity (Rejected)': specificity,
            'F1-Score': f1_score(y_test, y_pred, zero_division=0),
            'ROC-AUC': roc_auc,
            'False Positives (Bad Loans Approved)': fp,
            'False Negatives (Good Loans Rejected)': fn
        })
        
    df_metrics = pd.DataFrame(records)
    df_metrics = df_metrics.sort_values(by='Test Accuracy', ascending=False).reset_index(drop=True)
    return df_metrics


def plot_confusion_matrices(
    trained_models: Dict[str, Dict[str, Any]],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: str = "models/confusion_matrices.png"
) -> None:
    """
    Plots confusion matrices for all models in a grid layout.
    """
    n_models = len(trained_models)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = np.array(axes).flatten()
    
    for idx, (name, item) in enumerate(trained_models.items()):
        pipe = item['pipeline']
        y_pred = pipe.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[idx],
            xticklabels=['Rejected', 'Approved'],
            yticklabels=['Rejected', 'Approved']
        )
        axes[idx].set_title(f"{name}\nConfusion Matrix", fontsize=11, fontweight='bold')
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].set_ylabel("True Label")
        
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrices saved to {save_path}")


def plot_roc_curves(
    trained_models: Dict[str, Dict[str, Any]],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: str = "models/roc_curves.png"
) -> None:
    """
    Plots overlayed ROC curves for all models.
    """
    plt.figure(figsize=(9, 7))
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance (AUC = 0.50)')
    
    for name, item in trained_models.items():
        pipe = item['pipeline']
        if hasattr(pipe, "predict_proba"):
            y_prob = pipe.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_val = roc_auc_score(y_test, y_prob)
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.4f})")
            
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11)
    plt.title('HDFC Loan Approval - Receiver Operating Characteristic (ROC)', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"ROC Curves saved to {save_path}")


def get_feature_importances(pipeline: Any, X_train: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts feature importances or model weights from the trained pipeline.
    """
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']
    
    feature_names = preprocessor.get_feature_names_out()
    
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
        df_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
        return df_imp
    elif hasattr(classifier, 'coef_'):
        coefs = np.abs(classifier.coef_[0])
        df_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': coefs
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
        return df_imp
    return pd.DataFrame()

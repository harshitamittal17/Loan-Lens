"""
model_trainer.py
Module for defining preprocessing pipelines, training multiple supervised
machine learning classification models, conducting cross-validation,
and optimizing hyperparameters for HDFC Bank Ltd. loan approval.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, list, list]:
    """
    Constructs a ColumnTransformer that scales numeric features
    and one-hot encodes categorical features.
    """
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'string', 'str', 'category']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
        ]
    )
    return preprocessor, numeric_cols, categorical_cols


def get_baseline_classifiers() -> Dict[str, Any]:
    """
    Returns a dictionary of distinct supervised ML classification algorithms.
    """
    return {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, C=1.0),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, metric='minkowski'),
        'Support Vector Machine': CalibratedClassifierCV(
            estimator=SVC(kernel='rbf', C=1.0, random_state=42),
            cv=3
        ),
        'Decision Tree': DecisionTreeClassifier(
            criterion='gini',
            max_depth=6,
            min_samples_split=10,
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
    }


def train_and_cross_validate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    cv_folds: int = 5
) -> Dict[str, Dict[str, Any]]:
    """
    Trains each classifier within a Pipeline and performs Stratified K-Fold Cross Validation.
    """
    models = get_baseline_classifiers()
    results = {}
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    for name, clf in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='accuracy', n_jobs=-1)
        pipeline.fit(X_train, y_train)
        
        results[name] = {
            'pipeline': pipeline,
            'cv_mean': float(np.mean(cv_scores)),
            'cv_std': float(np.std(cv_scores)),
            'cv_scores': cv_scores.tolist()
        }
    return results


def tune_champion_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer
) -> Pipeline:
    """
    Performs GridSearchCV hyperparameter tuning on the ensemble model (Random Forest)
    to optimize decision boundaries and prevent overfitting.
    """
    rf_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    
    param_grid = {
        'classifier__n_estimators': [100, 150],
        'classifier__max_depth': [8, 12, None],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        rf_pipe,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def save_model(pipeline: Pipeline, filepath: str = "models/hdfc_loan_model.joblib") -> None:
    """
    Serializes trained pipeline to disk.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(pipeline, filepath)
    print(f"Champion model successfully saved to {filepath}")


def load_model(filepath: str = "models/hdfc_loan_model.joblib") -> Pipeline:
    """
    Loads saved model pipeline from disk.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at {filepath}")
    return joblib.load(filepath)

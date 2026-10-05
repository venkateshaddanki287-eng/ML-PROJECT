"""
CO2: Linear Supervised Learning Models.
Implements Linear Regression, Ridge, Lasso, ElasticNet, Logistic Regression,
and Multinomial Logistic Regression using scikit-learn.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    LogisticRegression
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score


class LinearModelTrainer:
    """
    Trains and evaluates linear supervised learning models for regression and classification.
    Handles strict scaling fitting on training data to prevent data leakage.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def get_regression_models(self) -> Dict[str, Any]:
        """Returns dictionary of linear regression models."""
        return {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0, random_state=self.random_state),
            'Lasso': Lasso(alpha=0.1, random_state=self.random_state),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state)
        }

    def get_classification_models(self) -> Dict[str, Any]:
        """Returns dictionary of linear classification models.

        scikit-learn 1.7+ no longer exposes the legacy ``multi_class`` constructor
        argument for ``LogisticRegression``; for multiclass problems ``lbfgs`` uses
        the multinomial formulation automatically.
        """
        base_logreg = LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            random_state=self.random_state
        )
        return {
            'LogisticRegression': base_logreg,
            'MultinomialLogisticRegression': LogisticRegression(
                solver='lbfgs',
                max_iter=1000,
                random_state=self.random_state
            )
        }

    def train_and_eval_regression(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Pipeline]]:
        """
        Trains all linear regression models with StandardScaler pipeline.
        Returns evaluation metrics and trained pipelines.
        """
        results = {}
        pipelines = {}

        for name, model in self.get_regression_models().items():
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            # Fit strictly on train data
            pipe.fit(X_train, y_train)
            
            y_pred = pipe.predict(X_test)
            
            rmse = float(root_mean_squared_error(y_test, y_pred))
            mae = float(mean_absolute_error(y_test, y_pred))
            r2 = float(r2_score(y_test, y_pred))

            results[name] = {'RMSE': round(rmse, 4), 'MAE': round(mae, 4), 'R2': round(r2, 4)}
            pipelines[name] = pipe

        return results, pipelines

    def train_and_eval_classification(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Pipeline]]:
        """
        Trains all linear classification models with StandardScaler pipeline.
        Returns evaluation metrics and trained pipelines.
        """
        results = {}
        pipelines = {}

        for name, model in self.get_classification_models().items():
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            pipe.fit(X_train, y_train)
            
            y_pred = pipe.predict(X_test)
            acc = float(accuracy_score(y_test, y_pred))
            f1 = float(f1_score(y_test, y_pred, average='weighted'))

            results[name] = {'Accuracy': round(acc, 4), 'F1-Weighted': round(f1, 4)}
            pipelines[name] = pipe

        return results, pipelines

    @staticmethod
    def explain_regularization() -> Dict[str, str]:
        """Provides theoretical explanations of L1 vs L2 regularization."""
        return {
            'L1 (Lasso)': 'Adds penalty equal to absolute value of coefficients (|w|). Promotes sparsity and feature selection.',
            'L2 (Ridge)': 'Adds penalty equal to square of magnitude of coefficients (w^2). Shrinks weights smoothly to handle multicollinearity.',
            'L1 + L2 (ElasticNet)': 'Combines L1 and L2 penalties via l1_ratio. Balances feature selection and group coefficient stability.'
        }

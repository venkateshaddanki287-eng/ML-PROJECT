"""
CO3: Tree-Based Supervised Learning Models.
Implements Decision Trees, Random Forests, Gradient Boosting, LightGBM/XGBoost,
Feature Importance extraction, decision splitting explanations, and ensemble variance reduction demo.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, root_mean_squared_error, r2_score

# Try LightGBM and XGBoost with fallback
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class TreeModelTrainer:
    """
    Trains and evaluates tree-based classifiers and regressors.
    Provides feature importances and ensemble stability comparisons.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def get_classification_models(self) -> Dict[str, Any]:
        """Returns dictionary of tree classification models."""
        models = {
            'DecisionTree': DecisionTreeClassifier(max_depth=6, random_state=self.random_state),
            'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=self.random_state),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=self.random_state)
        }

        if HAS_LIGHTGBM:
            models['LightGBM'] = LGBMClassifier(n_estimators=100, random_state=self.random_state, verbose=-1)
        elif HAS_XGBOOST:
            models['XGBoost'] = XGBClassifier(n_estimators=100, random_state=self.random_state, eval_metric='logloss')

        return models

    def get_regression_models(self) -> Dict[str, Any]:
        """Returns dictionary of tree regression models."""
        models = {
            'DecisionTreeRegressor': DecisionTreeRegressor(max_depth=6, random_state=self.random_state),
            'RandomForestRegressor': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=self.random_state),
            'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=100, random_state=self.random_state)
        }

        if HAS_LIGHTGBM:
            models['LightGBMRegressor'] = LGBMRegressor(n_estimators=100, random_state=self.random_state, verbose=-1)
        elif HAS_XGBOOST:
            models['XGBoostRegressor'] = XGBRegressor(n_estimators=100, random_state=self.random_state)

        return models

    def train_and_eval_classification(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Pipeline], Dict[str, pd.Series]]:
        """
        Trains tree classifiers and extracts feature importances.
        """
        results = {}
        pipelines = {}
        feature_importances = {}

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

            # Extract feature importance if available
            fitted_model = pipe.named_steps['model']
            if hasattr(fitted_model, 'feature_importances_'):
                importances = pd.Series(fitted_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
                feature_importances[name] = importances

        return results, pipelines, feature_importances

    def demonstrate_variance_reduction(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, n_bootstraps: int = 10
    ) -> Dict[str, Any]:
        """
        Demonstrates ensemble variance reduction: single Decision Tree vs Random Forest across bootstraps.
        """
        dt_scores = []
        rf_scores = []

        for i in range(n_bootstraps):
            # Bootstrap sample
            boot_idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
            X_b, y_b = X_train.iloc[boot_idx], y_train.iloc[boot_idx]

            dt = DecisionTreeClassifier(max_depth=None, random_state=i)
            dt.fit(X_b, y_b)
            dt_scores.append(accuracy_score(y_test, dt.predict(X_test)))

            rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=i)
            rf.fit(X_b, y_b)
            rf_scores.append(accuracy_score(y_test, rf.predict(X_test)))

        dt_std = float(np.std(dt_scores))
        rf_std = float(np.std(rf_scores))

        return {
            'DecisionTree_Mean_Accuracy': round(float(np.mean(dt_scores)), 4),
            'DecisionTree_Std_Variance': round(dt_std, 4),
            'RandomForest_Mean_Accuracy': round(float(np.mean(rf_scores)), 4),
            'RandomForest_Std_Variance': round(rf_std, 4),
            'Variance_Reduction_Factor': round(dt_std / max(1e-6, rf_std), 2),
            'Explanation': 'Random Forest reduces variance by averaging predictions across multiple decorrelated bootstrapped trees.'
        }

    @staticmethod
    def explain_tree_splitting() -> Dict[str, str]:
        """Explains decision tree node splitting concepts."""
        return {
            'Gini Impurity': 'Measures probability of misclassifying a randomly chosen element: Gini = 1 - sum(p_i^2). Lower is cleaner.',
            'Entropy': 'Measures information disorder in dataset: Entropy = -sum(p_i * log2(p_i)).',
            'Information Gain': 'Reduction in entropy/impurity achieved by splitting a node: IG = H(Parent) - sum(|N_j|/|N| * H(Child_j)).'
        }

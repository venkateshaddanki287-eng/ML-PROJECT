"""
CO5: Hyperparameter Tuning Module.
Implements GridSearchCV, RandomizedSearchCV, and Lightweight Bayesian Hyperparameter Optimization.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


class HyperparameterTuner:
    """
    Handles systematic hyperparameter search for optimal model tuning.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def tune_grid_search(
        self, estimator: Any, param_grid: Dict[str, list], X_train: pd.DataFrame, y_train: pd.Series
    ) -> Tuple[Any, Dict[str, Any], float]:
        """
        Executes exhaustive Grid Search cross-validation.
        """
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', estimator)
        ])
        
        # Adjust param_grid keys for Pipeline prefix 'model__'
        prefixed_grid = {f'model__{k}': v for k, v in param_grid.items()}

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        grid_search = GridSearchCV(
            pipe, prefixed_grid, cv=cv, scoring='f1_weighted', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        best_params = {k.replace('model__', ''): v for k, v in grid_search.best_params_.items()}
        return grid_search.best_estimator_, best_params, round(float(grid_search.best_score_), 4)

    def tune_random_search(
        self, estimator: Any, param_distributions: Dict[str, list], X_train: pd.DataFrame, y_train: pd.Series, n_iter: int = 10
    ) -> Tuple[Any, Dict[str, Any], float]:
        """
        Executes Randomized Search cross-validation.
        """
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', estimator)
        ])
        prefixed_dist = {f'model__{k}': v for k, v in param_distributions.items()}

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        rand_search = RandomizedSearchCV(
            pipe, prefixed_dist, n_iter=n_iter, cv=cv, scoring='f1_weighted', random_state=self.random_state, n_jobs=-1
        )
        rand_search.fit(X_train, y_train)

        best_params = {k.replace('model__', ''): v for k, v in rand_search.best_params_.items()}
        return rand_search.best_estimator_, best_params, round(float(rand_search.best_score_), 4)

    def tune_bayesian_lightweight(
        self, X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 8
    ) -> Tuple[Any, Dict[str, Any], float]:
        """
        Lightweight Bayesian / Sequential Model-Based Optimization (SMBO) approach
        using Gaussian Process surrogate optimization over Random Forest hyperparameters.
        """
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern
        from sklearn.model_selection import cross_val_score

        # Domain bounds: n_estimators [20, 150], max_depth [3, 15]
        param_space = [
            {'n_estimators': 30, 'max_depth': 4},
            {'n_estimators': 80, 'max_depth': 8},
            {'n_estimators': 120, 'max_depth': 12},
        ]

        evaluated_points = []
        scores = []

        for p in param_space:
            model = RandomForestClassifier(
                n_estimators=p['n_estimators'],
                max_depth=p['max_depth'],
                random_state=self.random_state
            )
            pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
            score = float(np.mean(cross_val_score(pipe, X_train, y_train, cv=3, scoring='f1_weighted')))
            evaluated_points.append([p['n_estimators'], p['max_depth']])
            scores.append(score)

        # Iterative surrogate model update
        gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-6, normalize_y=True, random_state=self.random_state)

        for _ in range(n_trials - len(param_space)):
            X_sample = np.array(evaluated_points)
            y_sample = np.array(scores)
            gp.fit(X_sample, y_sample)

            # Propose candidate using Expected Improvement surrogate
            candidates = np.array([
                [np.random.randint(20, 150), np.random.randint(3, 16)]
                for _ in range(20)
            ])
            gp_res = gp.predict(candidates, return_std=True)
            mu, sigma = gp_res[0], gp_res[1]
            best_y = np.max(y_sample)
            
            # Simple UCB selection: mu + 1.96 * sigma
            acquisition = mu + 1.96 * sigma
            best_cand_idx = np.argmax(acquisition)
            next_cand = candidates[best_cand_idx]

            n_est, m_depth = int(next_cand[0]), int(next_cand[1])
            model = RandomForestClassifier(n_estimators=n_est, max_depth=m_depth, random_state=self.random_state)
            pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
            score = float(np.mean(cross_val_score(pipe, X_train, y_train, cv=3, scoring='f1_weighted')))

            evaluated_points.append([n_est, m_depth])
            scores.append(score)

        best_idx = np.argmax(scores)
        best_p = {
            'n_estimators': int(evaluated_points[best_idx][0]),
            'max_depth': int(evaluated_points[best_idx][1])
        }
        best_score = float(scores[best_idx])

        best_model = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(**best_p, random_state=self.random_state))
        ])
        best_model.fit(X_train, y_train)

        return best_model, best_p, round(best_score, 4)

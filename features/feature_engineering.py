"""
Feature Engineering and Preprocessing Pipeline.
Defines traffic state features, scaling transformations, and training-serving consistency.
"""

from typing import List, Dict, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Authoritative feature list
FEATURE_NAMES: List[str] = [
    'north_queue',
    'south_queue',
    'east_queue',
    'west_queue',
    'north_arrival_rate',
    'south_arrival_rate',
    'east_arrival_rate',
    'west_arrival_rate',
    'current_green_time',
    'previous_queue',
    'queue_growth',
    'waiting_time',
    'traffic_density',
    'current_phase'
]

CATEGORICAL_TARGET = 'signal_action'
REGRESSION_TARGET = 'future_waiting_time'


def get_feature_schema() -> Dict[str, str]:
    """Returns data types for features to ensure schema consistency."""
    schema = {feat: 'float64' for feat in FEATURE_NAMES}
    return schema


def build_preprocessor() -> ColumnTransformer:
    """
    Creates a scikit-learn ColumnTransformer / StandardScaler pipeline.
    Avoids training-serving skew by using identical scaling parameters during inference.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num_scaler', StandardScaler(), FEATURE_NAMES)
        ],
        remainder='passthrough'
    )
    return preprocessor


def create_full_pipeline(estimator: Any) -> Pipeline:
    """
    Wraps preprocessor and estimator into a unified sklearn Pipeline.
    Prevents data leakage by fitting preprocessing only on training folds.
    """
    pipeline = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('model', estimator)
    ])
    return pipeline


def extract_features_df(features_dict: Dict[str, float]) -> pd.DataFrame:
    """
    Converts raw feature dict into a single-row DataFrame aligned with FEATURE_NAMES.
    """
    row = {}
    for feat in FEATURE_NAMES:
        row[feat] = float(features_dict.get(feat, 0.0))
    return pd.DataFrame([row])

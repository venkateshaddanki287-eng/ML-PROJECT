import pytest
import pandas as pd
from features.feature_engineering import FEATURE_NAMES, build_preprocessor, extract_features_df


def test_feature_names_count():
    assert len(FEATURE_NAMES) == 14
    assert 'north_queue' in FEATURE_NAMES
    assert 'current_green_time' in FEATURE_NAMES


def test_extract_features_df():
    sample_input = {
        'north_queue': 10,
        'south_queue': 5,
        'east_queue': 20,
        'west_queue': 15,
        'current_green_time': 8,
        'current_phase': 0
    }
    df = extract_features_df(sample_input)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert list(df.columns) == FEATURE_NAMES
    assert df.iloc[0]['north_queue'] == 10.0


def test_preprocessor_fitting():
    sample_df = pd.DataFrame([{f: 1.0 for f in FEATURE_NAMES}, {f: 2.0 for f in FEATURE_NAMES}])
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(sample_df)
    assert transformed.shape == (2, 14)

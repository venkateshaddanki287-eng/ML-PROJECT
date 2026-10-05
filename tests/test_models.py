import pytest
import pandas as pd
import numpy as np
from features.feature_engineering import FEATURE_NAMES
from models.linear_models import LinearModelTrainer
from models.tree_models import TreeModelTrainer
from models.clustering import UnsupervisedTrainer


@pytest.fixture
def dummy_dataset():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randint(0, 30, size=(100, 14)), columns=FEATURE_NAMES)
    y_cls = pd.Series(np.random.choice([0, 1, 2], size=100))
    y_reg = pd.Series(np.random.uniform(5, 50, size=100))
    return X, y_cls, y_reg


def test_linear_models(dummy_dataset):
    X, y_cls, y_reg = dummy_dataset
    trainer = LinearModelTrainer(random_state=42)
    reg_results, reg_pipes = trainer.train_and_eval_regression(X, y_reg, X, y_reg)
    assert 'LinearRegression' in reg_results
    assert 'Ridge' in reg_results
    assert 'Lasso' in reg_results
    assert 'ElasticNet' in reg_results

    cls_results, cls_pipes = trainer.train_and_eval_classification(X, y_cls, X, y_cls)
    assert 'LogisticRegression' in cls_results


def test_tree_models(dummy_dataset):
    X, y_cls, _ = dummy_dataset
    trainer = TreeModelTrainer(random_state=42)
    results, pipes, importances = trainer.train_and_eval_classification(X, y_cls, X, y_cls)
    assert 'DecisionTree' in results
    assert 'RandomForest' in results


def test_unsupervised_models(dummy_dataset):
    X, _, _ = dummy_dataset
    trainer = UnsupervisedTrainer(random_state=42)
    km, labels, centroids = trainer.fit_kmeans(X, n_clusters=3)
    assert len(labels) == 100
    assert len(centroids) == 3

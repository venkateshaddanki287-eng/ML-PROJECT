"""
End-to-End Training & Pipeline Selection Script.
Coordinates data loading, model training across CO2-CO5, hyperparameter tuning,
evaluation metrics generation, and model packaging for deployment.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from sklearn.model_selection import train_test_split
from features.feature_engineering import FEATURE_NAMES, CATEGORICAL_TARGET, REGRESSION_TARGET
from models.linear_models import LinearModelTrainer
from models.tree_models import TreeModelTrainer
from models.clustering import UnsupervisedTrainer
from models.anomaly_detection import AnomalyDetector
from training.evaluate import (
    evaluate_classification,
    evaluate_regression,
    evaluate_calibration,
    run_cross_validation
)
from training.tune import HyperparameterTuner


def run_training_pipeline() -> Dict[str, Any]:
    """
    Executes complete ML training lifecycle.
    """
    data_path = 'data/processed/traffic_dataset.csv'
    if not os.path.exists(data_path):
        from generate_dataset import generate_dataset
        df = generate_dataset()
        os.makedirs('data/processed', exist_ok=True)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)

    X = df[FEATURE_NAMES]
    y_cls = df[CATEGORICAL_TARGET]
    y_reg = df[REGRESSION_TARGET]

    # Train / Test Split (80% train, 20% test)
    X_train, X_test, y_train_cls, y_test_cls = train_test_split(
        X, y_cls, test_size=0.20, random_state=42, stratify=y_cls
    )
    _, _, y_train_reg, y_test_reg = train_test_split(
        X, y_reg, test_size=0.20, random_state=42
    )

    summary_report = {}

    print("--- 1. Training Linear Supervised Learning Models ---")
    lin_trainer = LinearModelTrainer(random_state=42)
    reg_results, reg_pipes = lin_trainer.train_and_eval_regression(X_train, y_train_reg, X_test, y_test_reg)
    cls_linear_results, cls_linear_pipes = lin_trainer.train_and_eval_classification(X_train, y_train_cls, X_test, y_test_cls)
    summary_report['Linear_Regression'] = reg_results
    summary_report['Linear_Classification'] = cls_linear_results

    print("--- 2. Training Tree-Based Ensemble Models ---")
    tree_trainer = TreeModelTrainer(random_state=42)
    tree_cls_results, tree_cls_pipes, feat_importances = tree_trainer.train_and_eval_classification(
        X_train, y_train_cls, X_test, y_test_cls
    )
    variance_demo = tree_trainer.demonstrate_variance_reduction(X_train, y_train_cls, X_test, y_test_cls)
    summary_report['Tree_Classification'] = tree_cls_results
    summary_report['Variance_Reduction_Demo'] = variance_demo

    print("--- 3. Running Unsupervised Clustering & Anomaly Detection ---")
    unsup_trainer = UnsupervisedTrainer(random_state=42)
    kmeans, km_labels, centroids = unsup_trainer.fit_kmeans(X, n_clusters=3)
    _, db_labels, db_stats = unsup_trainer.fit_dbscan(X)
    _, pca_coords, pca_stats = unsup_trainer.run_pca(X)
    
    anomaly_detector = AnomalyDetector(contamination=0.05, random_state=42)
    anom_labels, anom_scores, anom_stats = anomaly_detector.fit_predict(X)

    summary_report['Unsupervised_Learning'] = {
        'KMeans_Centroids': centroids.to_dict(orient='records'),
        'DBSCAN_Stats': db_stats,
        'PCA_Explained_Variance': pca_stats,
        'Anomaly_Detection_Stats': anom_stats
    }

    print("--- 4. Running Model Performance Evaluation & Hyperparameter Tuning ---")
    tuner = HyperparameterTuner(random_state=42)
    best_rf_grid, grid_params, grid_score = tuner.tune_grid_search(
        tree_trainer.get_classification_models()['RandomForest'],
        {'n_estimators': [50, 100], 'max_depth': [6, 10]},
        X_train, y_train_cls
    )
    
    bayesian_model, bayes_params, bayes_score = tuner.tune_bayesian_lightweight(X_train, y_train_cls, n_trials=6)

    # Detailed evaluation of best pipeline
    rf_eval = evaluate_classification(best_rf_grid, X_test, y_test_cls)
    rf_calib = evaluate_calibration(best_rf_grid, X_test, y_test_cls)
    cv_res = run_cross_validation(best_rf_grid, X_train, y_train_cls, task='classification')

    summary_report['Tuning_And_Evaluation'] = {
        'GridSearch_Best_Params': grid_params,
        'GridSearch_Best_CV_Score': grid_score,
        'Bayesian_Best_Params': bayes_params,
        'Bayesian_Best_CV_Score': bayes_score,
        'Best_Model_Test_Accuracy': rf_eval['Accuracy'],
        'Best_Model_Test_F1': rf_eval['F1_Weighted'],
        'Best_Model_Mean_Brier_Score': rf_calib.get('Mean_Brier_Score', 'N/A'),
        'Cross_Validation': cv_res
    }

    print("--- 5. Packaging Best Model Pipeline for Deployment ---")
    os.makedirs('models_saved', exist_ok=True)
    model_save_path = 'models_saved/best_model.joblib'
    info_save_path = 'models_saved/model_info.json'

    joblib.dump(best_rf_grid, model_save_path)

    model_metadata = {
        'model_name': 'RandomForestClassifier_Pipeline',
        'version': '1.0.0',
        'features': FEATURE_NAMES,
        'accuracy': rf_eval['Accuracy'],
        'f1_weighted': rf_eval['F1_Weighted'],
        'target_classes': {0: 'KEEP', 1: 'SWITCH_TO_NS', 2: 'SWITCH_TO_EW'},
        'best_hyperparameters': grid_params
    }

    with open(info_save_path, 'w', encoding='utf-8') as f:
        json.dump(model_metadata, f, indent=2)

    with open('models_saved/evaluation_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, indent=2)

    print(f"✅ Training completed successfully!")
    print(f"   Model saved to: {model_save_path}")
    print(f"   Metadata saved to: {info_save_path}")
    print(f"   Summary report saved to: models_saved/evaluation_summary.json")

    return summary_report


if __name__ == '__main__':
    run_training_pipeline()

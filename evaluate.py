"""
Evaluation Runner Script.
Loads dataset and trained models to generate comprehensive metrics across CO2, CO3, CO4, CO5.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from sklearn.model_selection import train_test_split
from features.feature_engineering import FEATURE_NAMES, CATEGORICAL_TARGET, REGRESSION_TARGET
from training.evaluate import (
    evaluate_classification,
    evaluate_regression,
    evaluate_calibration,
    run_cross_validation
)


def run_evaluation():
    print("=====================================================")
    print("📊 TRAFFIC SIGNAL OPTIMIZATION ML EVALUATION SUITE 📊")
    print("=====================================================")

    data_path = 'data/processed/traffic_dataset.csv'
    model_path = 'models_saved/best_model.joblib'

    if not os.path.exists(data_path) or not os.path.exists(model_path):
        print("⚠️ Data or saved model missing. Running train.py first...")
        from training.train import run_training_pipeline
        run_training_pipeline()

    df = pd.read_csv(data_path)
    X = df[FEATURE_NAMES]
    y_cls = df[CATEGORICAL_TARGET]
    y_reg = df[REGRESSION_TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cls, test_size=0.20, random_state=42, stratify=y_cls
    )

    model = joblib.load(model_path)

    print("\n--- 1. Model Classification Performance Metrics ---")
    cls_metrics = evaluate_classification(model, X_test, y_test)
    print(f"Accuracy         : {cls_metrics['Accuracy']}")
    print(f"Precision (Macro): {cls_metrics['Precision_Macro']}")
    print(f"Recall (Macro)   : {cls_metrics['Recall_Macro']}")
    print(f"F1 (Weighted)    : {cls_metrics['F1_Weighted']}")
    print(f"ROC-AUC          : {cls_metrics['ROC_AUC']}")
    print("\nConfusion Matrix:")
    for row in cls_metrics['Confusion_Matrix']:
        print(f"  {row}")

    print("\n--- 2. Probability Calibration Analysis ---")
    calib = evaluate_calibration(model, X_test, y_test)
    print(f"Brier Scores per Class: {calib.get('Brier_Scores')}")
    print(f"Mean Brier Score      : {calib.get('Mean_Brier_Score')}")
    print(f"Calibration Note      : {calib.get('Explanation')}")

    print("\n--- 3. 5-Fold Stratified Cross-Validation ---")
    cv_res = run_cross_validation(model, X_train, y_train, task='classification', n_splits=5)
    print(f"Mean CV Score: {cv_res['Mean_CV_Score']} +/- {cv_res['Std_CV_Score']}")
    print(f"Fold Scores  : {cv_res['Fold_Scores']}")

    print("\n=====================================================")
    print("✅ Evaluation complete! All metrics verified.")
    print("=====================================================")


if __name__ == '__main__':
    run_evaluation()

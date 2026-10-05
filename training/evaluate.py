"""
CO5: Model Evaluation Pipeline.
Calculates classification metrics, regression metrics, probability calibration curves,
Brier scores, and cross-validation performance.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score


def evaluate_classification(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Computes classification performance metrics across test set.
    """
    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average='macro', zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average='macro', zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))

    # ROC-AUC & PR-AUC if probabilities available
    roc_auc = None
    pr_auc = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
            n_classes = y_proba.shape[1]
            if n_classes > 2:
                roc_auc = float(roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro'))
            else:
                roc_auc = float(roc_auc_score(y_test, y_proba[:, 1]))
        except Exception:
            pass

    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return {
        'Accuracy': round(acc, 4),
        'Precision_Macro': round(prec_macro, 4),
        'Recall_Macro': round(rec_macro, 4),
        'F1_Macro': round(f1_macro, 4),
        'F1_Weighted': round(f1_weighted, 4),
        'ROC_AUC': round(roc_auc, 4) if roc_auc is not None else "N/A",
        'Confusion_Matrix': cm,
        'Classification_Report': report
    }


def evaluate_regression(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Computes regression performance metrics across test set.
    """
    y_pred = model.predict(X_test)
    rmse = float(root_mean_squared_error(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    return {
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        'R2': round(r2, 4)
    }


def evaluate_calibration(model: Any, X_test: pd.DataFrame, y_test: pd.Series, n_bins: int = 5) -> Dict[str, Any]:
    """
    Generates probability calibration curves and computes Brier score for reliability assessment.
    """
    if not hasattr(model, "predict_proba"):
        return {'status': 'Probabilities not supported by model'}

    y_proba = model.predict_proba(X_test)
    
    # Binary / one-vs-rest evaluation for class 0 or class 1
    # Brier score calculation: mean squared difference between predicted probability and actual binary indicator
    brier_scores = {}
    curves = {}
    
    classes = np.unique(y_test)
    for c in classes:
        binary_y = (y_test == c).astype(int)
        c_prob = y_proba[:, int(c)] if int(c) < y_proba.shape[1] else y_proba[:, 0]
        
        b_score = float(brier_score_loss(binary_y, c_prob))
        brier_scores[f"Class_{c}"] = round(b_score, 4)

        prob_true, prob_pred = calibration_curve(binary_y, c_prob, n_bins=n_bins, strategy='uniform')
        curves[f"Class_{c}"] = {
            'prob_true': [round(float(x), 4) for x in prob_true],
            'prob_pred': [round(float(x), 4) for x in prob_pred]
        }

    return {
        'Brier_Scores': brier_scores,
        'Mean_Brier_Score': round(float(np.mean(list(brier_scores.values()))), 4),
        'Calibration_Curves': curves,
        'Explanation': 'Brier score measures mean squared error of probability predictions (lower is better, 0 is perfect calibration).'
    }


def run_cross_validation(model: Any, X: pd.DataFrame, y: pd.Series, task: str = 'classification', n_splits: int = 5) -> Dict[str, float]:
    """
    Executes StratifiedKFold or KFold cross-validation without data leakage.
    """
    if task == 'classification':
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')

    return {
        'Mean_CV_Score': round(float(np.mean(scores)), 4),
        'Std_CV_Score': round(float(np.std(scores)), 4),
        'Fold_Scores': [round(float(s), 4) for s in scores]
    }

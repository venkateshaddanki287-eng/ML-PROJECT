"""
CO4: Anomaly Detection for Unusual Traffic Patterns.
Implements Isolation Forest anomaly detection.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AnomalyDetector:
    """
    Detects abnormal traffic events (spikes, extreme gridlocks) using Isolation Forest.
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(contamination=contamination, random_state=random_state)
        self.scaler = StandardScaler()

    def fit_predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Fits Isolation Forest and predicts anomaly status.
        Returns: (labels where -1=anomaly, 1=normal, anomaly_scores, stats_dict)
        """
        X_scaled = self.scaler.fit_transform(X)
        labels = self.model.fit_predict(X_scaled)
        scores = self.model.decision_function(X_scaled)

        n_anomalies = int((labels == -1).sum())
        n_normal = int((labels == 1).sum())

        stats = {
            'n_total': len(X),
            'n_normal': n_normal,
            'n_anomalies': n_anomalies,
            'anomaly_percentage': round(100.0 * n_anomalies / len(X), 2),
            'mean_anomaly_score': round(float(np.mean(scores)), 4)
        }

        return labels, scores, stats

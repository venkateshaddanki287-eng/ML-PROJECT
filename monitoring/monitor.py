"""
CO6: Prediction & Data Drift Monitoring.
Logs API prediction requests and calculates Population Stability Index (PSI) and KS-test for drift detection.
"""

import os
import json
import time
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


class PredictionMonitor:
    """
    Monitors live inference requests, tracks prediction distributions, latencies,
    and detects feature drift against reference baseline training distribution.
    """

    def __init__(
        self,
        log_file: str = "monitoring/prediction_logs.jsonl",
        baseline_path: str = "data/processed/traffic_dataset.csv"
    ):
        self.log_file = log_file
        self.baseline_path = baseline_path
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def log_prediction(self, features: Dict[str, float], prediction: str, confidence: float, latency_ms: float):
        """Appends a prediction record to the JSONL log file."""
        record = {
            'timestamp': time.time(),
            'features': features,
            'prediction': prediction,
            'confidence': float(confidence),
            'latency_ms': float(latency_ms)
        }
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')

    def read_logs(self) -> List[Dict[str, Any]]:
        """Reads all prediction logs."""
        if not os.path.exists(self.log_file):
            return []
        records = []
        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
        return records

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Calculates prediction count, latency statistics, and prediction distribution."""
        logs = self.read_logs()
        if not logs:
            return {
                'prediction_count': 0,
                'avg_latency_ms': 0.0,
                'avg_confidence': 0.0,
                'prediction_distribution': {}
            }

        latencies = [r['latency_ms'] for r in logs]
        confidences = [r.get('confidence', 1.0) for r in logs]
        preds = [r['prediction'] for r in logs]

        dist = pd.Series(preds).value_counts(normalize=True).round(4).to_dict()

        return {
            'prediction_count': len(logs),
            'avg_latency_ms': round(float(np.mean(latencies)), 2),
            'p95_latency_ms': round(float(np.percentile(latencies, 95)), 2),
            'avg_confidence': round(float(np.mean(confidences)), 4),
            'prediction_distribution': dist
        }

    def calculate_psi(self, reference: np.ndarray, current: np.ndarray, num_bins: int = 10) -> float:
        """
        Calculates Population Stability Index (PSI) between reference and current feature distributions.
        PSI < 0.1: No significant distribution change.
        0.1 <= PSI < 0.2: Moderate drift.
        PSI >= 0.2: Significant data drift detected!
        """
        if len(reference) < 5 or len(current) < 5:
            return 0.0

        percentiles = np.linspace(0, 100, num_bins + 1)
        bins = np.percentile(reference, percentiles)
        bins[0] = -np.inf
        bins[-1] = np.inf

        ref_counts, _ = np.histogram(reference, bins=bins)
        curr_counts, _ = np.histogram(current, bins=bins)

        ref_pct = ref_counts / max(1, len(reference))
        curr_pct = curr_counts / max(1, len(current))

        # Handle zeroes smoothly
        ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
        curr_pct = np.where(curr_pct == 0, 0.0001, curr_pct)

        psi_val = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
        return float(psi_val)

    def detect_drift(self) -> Dict[str, Any]:
        """
        Performs KS-test and PSI calculations across key input features.
        """
        logs = self.read_logs()
        if len(logs) < 10 or not os.path.exists(self.baseline_path):
            return {
                'drift_detected': False,
                'note': 'Insufficient prediction samples (<10) for statistically significant drift test.'
            }

        ref_df = pd.read_csv(self.baseline_path)
        curr_df = pd.DataFrame([r['features'] for r in logs])

        feature_drift_report = {}
        any_drift = False

        for col in ['north_queue', 'south_queue', 'east_queue', 'west_queue', 'waiting_time']:
            if col in ref_df.columns and col in curr_df.columns:
                ref_vals = ref_df[col].dropna().values
                curr_vals = curr_df[col].dropna().values

                psi_score = self.calculate_psi(ref_vals, curr_vals)
                stat, p_val = ks_2samp(ref_vals, curr_vals)

                is_drifted = bool(psi_score >= 0.2 or p_val < 0.01)
                if is_drifted:
                    any_drift = True

                feature_drift_report[col] = {
                    'PSI': round(psi_score, 4),
                    'KS_statistic': round(float(stat), 4),
                    'p_value': round(float(p_val), 4),
                    'drift_status': 'DRIFT DETECTED' if is_drifted else 'STABLE'
                }

        return {
            'drift_detected': any_drift,
            'features_analyzed': feature_drift_report
        }

import pytest
import os
import numpy as np
from monitoring.monitor import PredictionMonitor


def test_monitor_logging(tmp_path):
    log_file = tmp_path / "test_logs.jsonl"
    monitor = PredictionMonitor(log_file=str(log_file))

    sample_feats = {'north_queue': 5.0, 'south_queue': 2.0}
    monitor.log_prediction(sample_feats, "KEEP", 0.92, 12.5)

    logs = monitor.read_logs()
    assert len(logs) == 1
    assert logs[0]['prediction'] == "KEEP"
    assert logs[0]['confidence'] == 0.92


def test_psi_calculation():
    monitor = PredictionMonitor()
    ref = np.random.normal(10, 2, 100)
    curr = np.random.normal(10, 2, 100)
    psi = monitor.calculate_psi(ref, curr)
    assert psi < 0.1  # Same distribution -> low PSI

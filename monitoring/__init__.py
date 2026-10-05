"""
Monitoring and Drift Detection Package.
Logs inference requests and performs statistically sound data drift analysis (PSI / KS-test).
"""

from .monitor import PredictionMonitor

__all__ = ["PredictionMonitor"]

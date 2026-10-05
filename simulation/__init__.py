"""
Simulation Module for Traffic Signal Optimization.
Includes environment (TrafficIntersection), baseline rule-based and ML-driven brain (TrafficController),
and simulation visualization utilities.
"""

from .intersection import TrafficIntersection, PHASE_NS, PHASE_EW
from .controller import TrafficController

__all__ = ["TrafficIntersection", "TrafficController", "PHASE_NS", "PHASE_EW"]

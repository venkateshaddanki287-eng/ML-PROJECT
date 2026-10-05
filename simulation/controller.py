"""
Traffic Controller Module.
Supports baseline Rule-Based controller logic and ML-Based controller logic.
"""

from typing import Dict, Tuple, Optional
import requests
from .intersection import TrafficIntersection, PHASE_NS, PHASE_EW


class TrafficController:
    """
    The Controller / Decision Engine.
    Evaluates current intersection state using Rule-Based heuristics or ML Model predictions.
    """

    def __init__(
        self,
        config: Dict[str, int],
        mode: str = 'rule_based',
        ml_model=None,
        api_url: Optional[str] = None
    ):
        """
        config profile: MIN_GREEN, MAX_GREEN, YELLOW_TIME, THRESHOLD
        mode: 'rule_based' or 'ml_based'
        ml_model: pre-loaded sklearn Pipeline or model wrapper
        api_url: optional FastAPI endpoint URL (e.g. 'http://127.0.0.1:8000/predict')
        """
        self.config = config
        self.mode = mode
        self.ml_model = ml_model
        self.api_url = api_url
        self.last_confidence: Optional[float] = None

    def set_mode(self, mode: str):
        """Switch controller mode between 'rule_based' and 'ml_based'."""
        if mode in ['rule_based', 'ml_based']:
            self.mode = mode

    def evaluate(self, intersection: TrafficIntersection, arrivals: Dict = None) -> Tuple[str, str]:
        """
        Evaluates current state of intersection and returns necessary action.
        Returns: (ACTION, REASON_LOG)
        ACTION can be 'MAINTAIN', 'YELLOW', 'SWITCH'
        """
        self.last_confidence = None
        # 1. HARD SAFETY RULE: Clear intersection during YELLOW state
        if intersection.state == 'YELLOW':
            if intersection.phase_timer >= self.config.get('YELLOW_TIME', 2):
                return 'SWITCH', "CLEARANCE COMPLETE: Verified yellow phase ended. Safe to switch."
            else:
                return 'MAINTAIN', "CLEARING INTERSECTION: Safety clearance in progress (YELLOW)."

        # 2. ML-Based Decision Logic
        if self.mode == 'ml_based':
            return self._evaluate_ml(intersection, arrivals)
            
        # 3. Rule-Based Baseline Logic
        return self._evaluate_rule_based(intersection)

    def _evaluate_rule_based(self, intersection: TrafficIntersection) -> Tuple[str, str]:
        """Evaluates heuristic rule-based logic."""
        active_total, waiting_total = intersection.get_totals()
        trend = intersection.get_trend()
        timer = intersection.phase_timer
        
        min_green = self.config.get('MIN_GREEN', 5)
        max_green = self.config.get('MAX_GREEN', 15)
        threshold = self.config.get('THRESHOLD', 5)

        if timer >= max_green:
            return 'YELLOW', f"RULE-BASED (MAX GREEN): Max time {max_green}s reached. Forcing switch."
            
        elif timer >= min_green:
            if waiting_total >= active_total + threshold:
                return 'YELLOW', f"RULE-BASED (OPTIMIZATION): Waiting traffic ({waiting_total}) > Active ({active_total}) + {threshold}."
                
            elif active_total == 0 and waiting_total > 0:
                return 'YELLOW', "RULE-BASED (EMPTY LANE): Active lanes empty, waiting traffic present."
                
            elif trend == "growing exponentially":
                return 'YELLOW', "RULE-BASED (TREND): Waiting queue growing exponentially. Preemptive switch."
                
            else:
                return 'MAINTAIN', f"RULE-BASED (STABLE): Traffic flowing. Maintaining phase (Timer: {timer}s)."
                
        else:
            return 'MAINTAIN', f"RULE-BASED (MIN LOCK): Holding green to prevent rapid toggling ({timer}/{min_green}s)."

    def _evaluate_ml(self, intersection: TrafficIntersection, arrivals: Dict = None) -> Tuple[str, str]:
        """Evaluates trained ML model predictions."""
        timer = intersection.phase_timer
        min_green = self.config.get('MIN_GREEN', 5)
        max_green = self.config.get('MAX_GREEN', 15)

        # Enforce max green safety lock
        if timer >= max_green:
            return 'YELLOW', f"ML-OVERRIDE (MAX GREEN): Max time {max_green}s reached. Forcing switch."

        # Enforce min green lock
        if timer < min_green:
            return 'MAINTAIN', f"ML-LOCK (MIN GREEN): Holding green lock ({timer}/{min_green}s)."

        # Extract features
        features = intersection.get_features(arrivals)

        prediction_action = "KEEP"
        confidence = 1.0
        model_name = "ML_Model"

        # Try API request if URL provided
        if self.api_url:
            try:
                resp = requests.post(self.api_url, json=features, timeout=0.5)
                if resp.status_code == 200:
                    data = resp.json()
                    prediction_action = data.get("prediction", "KEEP")
                    confidence = data.get("confidence", 1.0)
                    model_name = data.get("model", "FastAPI_Model")
            except Exception:
                # Fallback to local model if API is unreachable
                prediction_action, confidence, model_name = self._predict_local(features)
        else:
            prediction_action, confidence, model_name = self._predict_local(features)

        self.last_confidence = float(confidence)

        # Convert ML prediction to traffic controller action
        # Actions: KEEP, SWITCH, SWITCH_TO_NS, SWITCH_TO_EW
        should_switch = False
        if prediction_action == "SWITCH":
            should_switch = True
        elif prediction_action == "SWITCH_TO_NS" and intersection.active_phase != PHASE_NS:
            should_switch = True
        elif prediction_action == "SWITCH_TO_EW" and intersection.active_phase != PHASE_EW:
            should_switch = True

        if should_switch:
            return 'YELLOW', f"ML-PREDICTION [{model_name}] (Conf: {confidence:.2f}): Recommends phase switch."
        else:
            return 'MAINTAIN', f"ML-PREDICTION [{model_name}] (Conf: {confidence:.2f}): Recommends maintaining phase."

    def _predict_local(self, features: Dict[str, float]) -> Tuple[str, float, str]:
        """Local model prediction fallback."""
        if self.ml_model is not None:
            try:
                import pandas as pd
                df = pd.DataFrame([features])
                if hasattr(self.ml_model, "predict_proba"):
                    probs = self.ml_model.predict_proba(df)[0]
                    pred_idx = probs.argmax()
                    conf = float(probs[pred_idx])
                    classes = getattr(self.ml_model, "classes_", [0, 1, 2])
                    label_map = {0: "KEEP", 1: "SWITCH_TO_NS", 2: "SWITCH_TO_EW", "KEEP": "KEEP", "SWITCH": "SWITCH"}
                    pred_label = label_map.get(classes[pred_idx], str(classes[pred_idx]))
                    return pred_label, conf, self.ml_model.__class__.__name__
                else:
                    pred = self.ml_model.predict(df)[0]
                    label_map = {0: "KEEP", 1: "SWITCH_TO_NS", 2: "SWITCH_TO_EW", "KEEP": "KEEP", "SWITCH": "SWITCH"}
                    return label_map.get(pred, str(pred)), 0.90, self.ml_model.__class__.__name__
            except Exception:
                pass

        # Smart fallback if model is not yet loaded/trained
        active_tot, wait_tot = features['north_queue'] + features['south_queue'], features['east_queue'] + features['west_queue']
        if features['current_phase'] == 1.0: # EW active
            active_tot, wait_tot = wait_tot, active_tot
        
        if wait_tot > active_tot + self.config.get('THRESHOLD', 5):
            return "SWITCH", 0.85, "Rule_Fallback"
        return "KEEP", 0.90, "Rule_Fallback"

from typing import Dict, Tuple
from intersection import TrafficIntersection

class TrafficController:
    """
    The Brain: Encapsulates the heuristic optimization logic.
    Receives state from the Intersection object and returns an action.
    """

    def __init__(self, config: Dict[str, int]):
        """
        Accepts a configuration profile for the controller logic.
        Expected keys: MIN_GREEN, MAX_GREEN, YELLOW_TIME, THRESHOLD
        """
        self.config = config

    def evaluate(self, intersection: TrafficIntersection) -> Tuple[str, str]:
        """
        Evaluates the current state of the intersection and returns the necessary action.
        Returns: (ACTION, REASON_LOG)
        ACTION can be 'MAINTAIN', 'YELLOW', 'SWITCH'
        """
        state = intersection.state
        timer = intersection.phase_timer
        
        # 1. Collision / Safety Logic: Hard Stop Rule
        if state == 'YELLOW':
            if timer >= self.config.get('YELLOW_TIME', 2):
                return 'SWITCH', "CLEARANCE COMPLETE: Verified yellow phase ended. Safe to switch."
            else:
                return 'MAINTAIN', "CLEARING INTERSECTION: Safety clearance in progress (YELLOW)."
        
        # 2. Extract Data
        active_total, waiting_total = intersection.get_totals()
        trend = intersection.get_trend()
        
        min_green = self.config.get('MIN_GREEN', 5)
        max_green = self.config.get('MAX_GREEN', 15)
        threshold = self.config.get('THRESHOLD', 5)

        # 3. Apply Heuristics
        if timer >= max_green:
            return 'YELLOW', f"MAX TIME REACHED ({max_green}s): Forcing phase switch to prevent starvation."
            
        elif timer >= min_green:
            if waiting_total >= active_total + threshold:
                return 'YELLOW', f"OPTIMIZATION: Cross-street traffic high ({waiting_total} vs {active_total}). Switching early."
                
            elif active_total == 0 and waiting_total > 0:
                return 'YELLOW', "EMPTY LANE OVERRIDE: Active lanes are empty. Switching to serve waiting traffic."
                
            elif trend == "growing exponentially":
                return 'YELLOW', "TREND DETECTED: Waiting queue is growing exponentially. Preemptive switch."
                
            else:
                return 'MAINTAIN', f"STABLE (Trend: {trend}): Traffic flowing smoothly. Maintaining current phase."
                
        else:
            return 'MAINTAIN', f"MINIMUM TIME LOCK: Holding green to prevent rapid toggling (Timer: {timer}/{min_green}s)."

"""
Traffic Intersection Environment.
Manages intersection state (queues, lights, timers, trends, performance tracking).
"""

from typing import Dict, List, Tuple
import random

PHASE_NS = 'North/South'
PHASE_EW = 'East/West'


class TrafficIntersection:
    """
    The Environment: Manages the state of the intersection (queues, lights, timers).
    Handles traffic flow physics (adding/removing cars) and history tracking.
    """

    def __init__(self, max_capacity: int = 100):
        # Track cars waiting in each lane
        self.queues: Dict[str, int] = {'North': 0, 'South': 0, 'East': 0, 'West': 0}
        
        # State tracking
        self.active_phase: str = PHASE_NS
        self.state: str = 'GREEN'  # 'GREEN', 'YELLOW', 'RED'
        
        # Timers
        self.phase_timer: int = 0
        self.total_time: int = 0
        
        # Capacity limit for density calculation
        self.max_capacity: int = max_capacity

        # Performance statistics
        self.total_cars_processed: int = 0
        self.total_wait_accumulated: float = 0.0
        self.max_queue_observed: int = 0
        self.phase_switch_count: int = 0
        
        # History tracking (last 10 seconds of queue sizes)
        self.history: List[Dict[str, int]] = []

    def record_history(self):
        """Saves a snapshot of current queue sizes."""
        self.history.append(self.queues.copy())
        if len(self.history) > 10:
            self.history.pop(0)

    def add_traffic(self, arrivals: Dict[str, Tuple[int, int]]):
        """
        Randomly generates arriving traffic based on provided scenario bounds.
        arrivals should look like {'North': (min, max), 'South': (min, max), ...}
        """
        for direction, bounds in arrivals.items():
            arrivals_num = random.randint(bounds[0], bounds[1])
            self.queues[direction] += arrivals_num

    def process_traffic(self, throughput: int = 3):
        """Removes cars from the queues that currently have a GREEN light."""
        if self.state == 'GREEN':
            active_directions = ['North', 'South'] if self.active_phase == PHASE_NS else ['East', 'West']
            for direction in active_directions:
                processed = min(self.queues[direction], random.randint(1, throughput))
                self.queues[direction] -= processed
                self.total_cars_processed += processed

    def get_totals(self) -> Tuple[int, int]:
        """Returns a tuple of (active_queue_total, waiting_queue_total)."""
        ns_total = self.queues['North'] + self.queues['South']
        ew_total = self.queues['East'] + self.queues['West']
        
        if self.active_phase == PHASE_NS:
            return ns_total, ew_total
        else:
            return ew_total, ns_total

    def get_trend(self) -> str:
        """Analyzes history to check if the waiting queue is growing."""
        if len(self.history) < 5:
            return "stable"
            
        past_waiting = []
        for snapshot in self.history[-5:]:
            ns = snapshot['North'] + snapshot['South']
            ew = snapshot['East'] + snapshot['West']
            past_waiting.append(ew if self.active_phase == PHASE_NS else ns)
            
        if past_waiting[-1] > past_waiting[0] + 5:
            return "growing exponentially"
        elif past_waiting[-1] > past_waiting[0]:
            return "growing steadily"
        else:
            return "stable"

    def get_queue_growth(self) -> float:
        """Calculates queue length change rate over recent history."""
        if len(self.history) < 2:
            return 0.0
        current_total = sum(self.queues.values())
        previous_total = sum(self.history[-2].values())
        return float(current_total - previous_total)

    def get_features(self, arrivals: Dict[str, Tuple[int, int]] = None) -> Dict[str, float]:
        """
        Extracts feature snapshot for ML model prediction.
        """
        active_total, waiting_total = self.get_totals()
        total_queue = sum(self.queues.values())
        
        # Update statistics
        self.total_wait_accumulated += total_queue
        if total_queue > self.max_queue_observed:
            self.max_queue_observed = total_queue
            
        arr_rates = {}
        if arrivals:
            for d in ['North', 'South', 'East', 'West']:
                bounds = arrivals.get(d, (0, 2))
                arr_rates[f"{d.lower()}_arrival_rate"] = (bounds[0] + bounds[1]) / 2.0
        else:
            for d in ['North', 'South', 'East', 'West']:
                arr_rates[f"{d.lower()}_arrival_rate"] = 1.0

        previous_q = sum(self.history[-2].values()) if len(self.history) >= 2 else total_queue
        q_growth = self.get_queue_growth()
        avg_wait_time = float(self.total_wait_accumulated) / max(1, self.total_cars_processed + total_queue)
        density = min(1.0, float(total_queue) / max(1, self.max_capacity))

        return {
            'north_queue': float(self.queues['North']),
            'south_queue': float(self.queues['South']),
            'east_queue': float(self.queues['East']),
            'west_queue': float(self.queues['West']),
            'north_arrival_rate': arr_rates.get('north_arrival_rate', 1.0),
            'south_arrival_rate': arr_rates.get('south_arrival_rate', 1.0),
            'east_arrival_rate': arr_rates.get('east_arrival_rate', 1.0),
            'west_arrival_rate': arr_rates.get('west_arrival_rate', 1.0),
            'current_green_time': float(self.phase_timer if self.state == 'GREEN' else 0),
            'previous_queue': float(previous_q),
            'queue_growth': float(q_growth),
            'waiting_time': float(avg_wait_time),
            'traffic_density': float(density),
            'current_phase': 0.0 if self.active_phase == PHASE_NS else 1.0
        }

    def transition_to_yellow(self):
        """Forces the intersection into yellow clearance state."""
        self.state = 'YELLOW'
        self.phase_timer = 0

    def switch_phase(self):
        """Switches active phase to cross street."""
        self.active_phase = PHASE_EW if self.active_phase == PHASE_NS else PHASE_NS
        self.state = 'GREEN'
        self.phase_timer = 0
        self.phase_switch_count += 1

    def tick(self, arrivals: Dict[str, Tuple[int, int]]):
        """Advances intersection state by 1 second."""
        self.total_time += 1
        self.phase_timer += 1
        self.add_traffic(arrivals)
        self.process_traffic()
        self.max_queue_observed = max(self.max_queue_observed, sum(self.queues.values()))
        self.record_history()

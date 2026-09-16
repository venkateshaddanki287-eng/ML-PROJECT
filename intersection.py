from typing import Dict, List, Tuple
import random

PHASE_NS = 'North/South'
PHASE_EW = 'East/West'

class TrafficIntersection:
    """
    The Environment: Manages the state of the intersection (queues, lights, timers).
    It handles the physics of 'traffic' (adding/removing cars) and holds the data structure.
    It is completely agnostic of the 'rules'.
    """

    def __init__(self):
        # Track cars waiting in each lane
        self.queues: Dict[str, int] = {'North': 0, 'South': 0, 'East': 0, 'West': 0}
        
        # State tracking
        self.active_phase: str = PHASE_NS
        self.state: str = 'GREEN'  # 'GREEN', 'YELLOW', 'RED'
        
        # Timers
        self.phase_timer: int = 0
        self.total_time: int = 0
        
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
            self.queues[direction] += random.randint(bounds[0], bounds[1])

    def process_traffic(self, throughput: int = 3):
        """Removes cars from the queues that currently have a GREEN light."""
        if self.state == 'GREEN':
            active_directions = ['North', 'South'] if self.active_phase == PHASE_NS else ['East', 'West']
            for direction in active_directions:
                # Remove up to `throughput` cars per second if the queue is not empty
                processed = min(self.queues[direction], random.randint(1, throughput))
                self.queues[direction] -= processed

    def get_totals(self) -> Tuple[int, int]:
        """Returns a tuple of (active_queue_total, waiting_queue_total)."""
        ns_total = self.queues['North'] + self.queues['South']
        ew_total = self.queues['East'] + self.queues['West']
        
        if self.active_phase == PHASE_NS:
            return ns_total, ew_total
        else:
            return ew_total, ns_total

    def get_trend(self) -> str:
        """Analyzes the history to see if the waiting queue is growing."""
        if len(self.history) < 5:
            return "stable"
            
        # Get waiting totals over the last 5 seconds
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

    def transition_to_yellow(self):
        """Forces the intersection into a yellow clearance state."""
        self.state = 'YELLOW'
        self.phase_timer = 0

    def switch_phase(self):
        """Switches the active phase to the cross street."""
        self.active_phase = PHASE_EW if self.active_phase == PHASE_NS else PHASE_NS
        self.state = 'GREEN'
        self.phase_timer = 0

    def tick(self, arrivals: Dict[str, Tuple[int, int]]):
        """Advances the intersection by one second."""
        self.total_time += 1
        self.phase_timer += 1
        self.add_traffic(arrivals)
        self.process_traffic()
        self.record_history()

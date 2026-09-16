import time
import random
import sys

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ==========================================
# Constants and Configuration
# ==========================================
PHASE_NS = 'North/South'
PHASE_EW = 'East/West'

# Visual representation of light states
COLOR_GREEN = 'GREEN  🟢'
COLOR_YELLOW = 'YELLOW 🟡'
COLOR_RED = 'RED    🔴'

# Heuristic Rules Constants
MIN_GREEN_TIME = 5    # Minimum seconds a light stays green
MAX_GREEN_TIME = 15   # Maximum seconds before forcing a switch
YELLOW_TIME = 2       # Clearance phase duration
QUEUE_THRESHOLD = 5   # Wait vs Active queue difference to trigger switch


class TrafficSimulation:
    def __init__(self):
        # Track cars waiting in each lane
        self.queues = {'North': 0, 'South': 0, 'East': 0, 'West': 0}
        
        # Initial phase setup
        self.active_phase = PHASE_NS
        self.state = 'GREEN'  # Current state of the active phase ('GREEN' or 'YELLOW')
        
        # Timing trackers
        self.phase_timer = 0
        self.total_time = 0
        
        # UI logging
        self.decision_log = "System initialized. Starting simulation..."

    def add_traffic(self):
        """Randomly generates arriving traffic on each tick."""
        for direction in self.queues:
            # Add between 0 and 2 cars randomly to each queue every second
            self.queues[direction] += random.randint(0, 2)

    def process_traffic(self):
        """Removes cars from the queues that currently have a GREEN light."""
        if self.state == 'GREEN':
            # Determine which lanes are currently moving
            active_directions = ['North', 'South'] if self.active_phase == PHASE_NS else ['East', 'West']
            
            for direction in active_directions:
                # Process 1 to 3 cars per second if the queue is not empty
                processed = min(self.queues[direction], random.randint(1, 3))
                self.queues[direction] -= processed

    def get_queue_totals(self):
        """Returns a tuple of (active_queue_total, waiting_queue_total)."""
        ns_total = self.queues['North'] + self.queues['South']
        ew_total = self.queues['East'] + self.queues['West']
        
        if self.active_phase == PHASE_NS:
            return ns_total, ew_total
        else:
            return ew_total, ns_total

    def evaluate_heuristics(self):
        """The Decision Engine: Evaluates traffic rules and handles phase switching."""
        
        # 1. Handle Yellow Clearance Phase
        if self.state == 'YELLOW':
            if self.phase_timer >= YELLOW_TIME:
                # Yellow phase complete, switch the active phase to the cross-street
                self.active_phase = PHASE_EW if self.active_phase == PHASE_NS else PHASE_NS
                self.state = 'GREEN'
                self.phase_timer = 0
                self.decision_log = "CLEARANCE COMPLETE: Switching to new phase."
            else:
                self.decision_log = "CLEARING INTERSECTION: Safety clearance in progress (YELLOW)."
            return

        # 2. Evaluate Green Phase Heuristics
        active_total, waiting_total = self.get_queue_totals()

        # Rule A: Maximum Green Time
        if self.phase_timer >= MAX_GREEN_TIME:
            self.state = 'YELLOW'
            self.phase_timer = 0
            self.decision_log = f"MAX TIME REACHED ({MAX_GREEN_TIME}s): Forcing phase switch to prevent starvation."
            
        # Rule B: Optimization check (only if Minimum Green Time has passed)
        elif self.phase_timer >= MIN_GREEN_TIME:
            
            # Queue Optimization (Preemptive Switch)
            if waiting_total >= active_total + QUEUE_THRESHOLD:
                self.state = 'YELLOW'
                self.phase_timer = 0
                self.decision_log = f"OPTIMIZATION: Cross-street traffic very high ({waiting_total} waiting vs {active_total} active). Switching early."
                
            # Empty Lane Override
            elif active_total == 0 and waiting_total > 0:
                self.state = 'YELLOW'
                self.phase_timer = 0
                self.decision_log = "EMPTY LANE OVERRIDE: Active lanes are empty. Switching to serve waiting traffic."
                
            # Stable State
            else:
                self.decision_log = "STABLE: Traffic flowing smoothly. Maintaining current phase."
                
        # Rule C: Minimum Green Time Lock
        else:
            self.decision_log = f"MINIMUM TIME LOCK: Holding green to prevent rapid toggling (Timer: {self.phase_timer}/{MIN_GREEN_TIME}s)."

    def render(self):
        """Draws the terminal UI dashboard."""
        # ANSI escape codes to clear the screen and move cursor to top left
        sys.stdout.write('\033[H\033[J')
        sys.stdout.flush()

        # Determine visual light states for output
        ns_color = COLOR_RED
        ew_color = COLOR_RED
        
        if self.active_phase == PHASE_NS:
            ns_color = COLOR_GREEN if self.state == 'GREEN' else COLOR_YELLOW
        else:
            ew_color = COLOR_GREEN if self.state == 'GREEN' else COLOR_YELLOW

        # Dashboard layout
        dashboard = f"""
🚦 TRAFFIC SIGNAL OPTIMIZATION SIMULATOR 🚦
===========================================
Total Elapsed Time : {self.total_time:03d}s
Active Phase       : {self.active_phase}
Phase Timer        : {self.phase_timer:02d}s
State              : {self.state}
===========================================

  🚗 North Queue: {self.queues['North']:3d} cars  [{ns_color}]
  🚗 South Queue: {self.queues['South']:3d} cars  [{ns_color}]
-------------------------------------------
  🚗 East Queue : {self.queues['East']:3d} cars  [{ew_color}]
  🚗 West Queue : {self.queues['West']:3d} cars  [{ew_color}]

===========================================
📝 DECISION LOG:
> {self.decision_log}
===========================================
[Press Ctrl+C to stop the simulation]
"""
        sys.stdout.write(dashboard)
        sys.stdout.flush()

    def run(self):
        """Main simulation loop."""
        try:
            while True:
                # 1. Render the current state to the terminal
                self.render()
                
                # 2. Wait for 1 second (1 tick)
                time.sleep(1)
                
                # 3. Update timers
                self.total_time += 1
                self.phase_timer += 1
                
                # 4. Generate new traffic
                self.add_traffic()
                
                # 5. Let cars pass through green lights
                self.process_traffic()
                
                # 6. Evaluate rules and possibly switch lights for the next tick
                self.evaluate_heuristics()
                
        except KeyboardInterrupt:
            # Handle graceful exit when user presses Ctrl+C
            sys.stdout.write('\n\n🛑 Simulation stopped by user.\n')
            sys.stdout.flush()


if __name__ == "__main__":
    sim = TrafficSimulation()
    sim.run()

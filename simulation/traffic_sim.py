"""
Traffic Simulation Terminal Dashboard & Execution Engine.
Supports rule-based and ML-driven interactive traffic signal control.
"""

import time
import random
import sys
import os
import select
from typing import Dict

from .intersection import TrafficIntersection, PHASE_NS, PHASE_EW
from .controller import TrafficController

COLOR_GREEN = 'GREEN  🟢'
COLOR_YELLOW = 'YELLOW 🟡'
COLOR_RED = 'RED    🔴'


def get_light_states(active_phase: str, state: str) -> Dict[str, str]:
    """Derive a safe pair of signal lights from the intersection state."""
    active_state = state if state in {'GREEN', 'YELLOW'} else 'RED'
    if active_phase == PHASE_NS:
        return {PHASE_NS: active_state, PHASE_EW: 'RED'}
    return {PHASE_NS: 'RED', PHASE_EW: active_state}


class TrafficSimulation:
    def __init__(self, mode: str = 'rule_based', scenario: str = 'normal'):
        self.intersection = TrafficIntersection()
        
        # Configure arrivals based on scenario
        if scenario == 'heavy_traffic':
            self.arrivals = {dir_name: (2, 5) for dir_name in ['North', 'South', 'East', 'West']}
            self.config = {'MIN_GREEN': 5, 'MAX_GREEN': 20, 'YELLOW_TIME': 2, 'THRESHOLD': 8}
        elif scenario == 'night_mode':
            self.arrivals = {dir_name: (0, 1) for dir_name in ['North', 'South', 'East', 'West']}
            self.config = {'MIN_GREEN': 3, 'MAX_GREEN': 10, 'YELLOW_TIME': 2, 'THRESHOLD': 3}
        elif scenario == 'rush_hour':
            self.arrivals = {'North': (3, 6), 'South': (3, 6), 'East': (0, 2), 'West': (0, 2)}
            self.config = {'MIN_GREEN': 5, 'MAX_GREEN': 15, 'YELLOW_TIME': 2, 'THRESHOLD': 5}
        else: # normal
            self.arrivals = {dir_name: (0, 2) for dir_name in ['North', 'South', 'East', 'West']}
            self.config = {'MIN_GREEN': 5, 'MAX_GREEN': 15, 'YELLOW_TIME': 2, 'THRESHOLD': 5}

        self.controller = TrafficController(self.config, mode=mode)
        self.decision_log = f"Simulation initialized in '{scenario}' mode with '{mode}' controller."
        self.last_action = 'READY'

    def render(self):
        """Draws terminal UI dashboard."""
        os.system('cls' if os.name == 'nt' else 'clear')
        lights = get_light_states(self.intersection.active_phase, self.intersection.state)
        signal_lines = []
        for signal, phase in (('NORTH / SOUTH', PHASE_NS), ('EAST / WEST', PHASE_EW)):
            signal_lines.append(f'{signal:^28}')
            for color in ('RED', 'YELLOW', 'GREEN'):
                lamp = {'RED': '🔴', 'YELLOW': '🟡', 'GREEN': '🟢'}[color]
                if lights[phase] != color:
                    lamp = '○'
                signal_lines.append(f'          {lamp}  {color}')

        dashboard = f"""
🚦 TRAFFIC SIGNAL OPTIMIZATION SIMULATOR 🚦
=====================================================
Total Elapsed Time : {self.intersection.total_time:03d}s
Simulation Step    : {self.intersection.total_time}
Active Controller  : {self.controller.mode.upper()}
Active Phase       : {self.intersection.active_phase}
Phase Timer        : {self.intersection.phase_timer:02d}s
State              : {self.intersection.state}
=====================================================

{signal_lines[0]:<36}{signal_lines[4]}
{signal_lines[1]:<36}{signal_lines[5]}
{signal_lines[2]:<36}{signal_lines[6]}
{signal_lines[3]:<36}{signal_lines[7]}

  🚗 North Queue: {self.intersection.queues['North']:3d} cars
  🚗 South Queue: {self.intersection.queues['South']:3d} cars
-----------------------------------------------------
  🚗 East Queue : {self.intersection.queues['East']:3d} cars
  🚗 West Queue : {self.intersection.queues['West']:3d} cars

=====================================================
📊 PERFORMANCE METRICS:
Total Cars Processed: {self.intersection.total_cars_processed}
Max Queue Observed  : {self.intersection.max_queue_observed}
Phase Switch Count  : {self.intersection.phase_switch_count}
=====================================================
📝 DECISION LOG:
> {self.last_action}: {self.decision_log}
=====================================================
[SPACE] Pause/Resume  [S] Step while paused  [R] Reset  [Q] Quit
"""
        sys.stdout.write(dashboard)
        sys.stdout.flush()

    def _advance(self) -> str:
        self.intersection.tick(self.arrivals)
        action, self.decision_log = self.controller.evaluate(self.intersection, self.arrivals)
        self.last_action = action
        if action == 'YELLOW':
            self.intersection.transition_to_yellow()
        elif action == 'SWITCH':
            self.intersection.switch_phase()
        return action

    def _read_key(self):
        if os.name == 'nt':
            import msvcrt

            if msvcrt.kbhit():
                return msvcrt.getwch().lower()
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        return sys.stdin.read(1).lower() if ready else None

    def _reset(self) -> None:
        self.intersection = TrafficIntersection()
        self.decision_log = f"Simulation reset in '{self.controller.mode}' controller mode."
        self.last_action = 'READY'

    def run_visual(self, tick_delay: float = 0.5):
        """Run the terminal light display with non-blocking keyboard controls."""
        running = True
        try:
            while True:
                self.render()
                deadline = time.monotonic() + max(0.0, tick_delay) if running else None
                key = None
                while key is None and (not running or time.monotonic() < deadline):
                    key = self._read_key()
                    if key is None:
                        time.sleep(0.05)

                if key == 'q':
                    break
                if key == ' ':
                    running = not running
                elif key == 'r':
                    self._reset()
                elif key == 's' and not running:
                    self._advance()
                elif key is None and running:
                    self._advance()
        except KeyboardInterrupt:
            pass
        finally:
            sys.stdout.write('\nTerminal traffic simulation stopped.\n')
            sys.stdout.flush()

    def run(self, max_ticks: int = None, tick_delay: float = 1.0):
        """Main simulation loop."""
        try:
            ticks = 0
            while True:
                self.render()
                if tick_delay > 0:
                    time.sleep(tick_delay)
                
                self._advance()

                ticks += 1
                if max_ticks and ticks >= max_ticks:
                    break

        except KeyboardInterrupt:
            sys.stdout.write('\n\n🛑 Simulation stopped by user.\n')
            sys.stdout.flush()


if __name__ == "__main__":
    sim = TrafficSimulation()
    sim.run()

import time
import sys
import argparse
import os
from intersection import TrafficIntersection, PHASE_NS
from controller import TrafficController

# Visual representations
COLOR_GREEN = 'GREEN  🟢'
COLOR_YELLOW = 'YELLOW 🟡'
COLOR_RED = 'RED    🔴'

def render_dashboard(intersection: TrafficIntersection, decision_log: str):
    """Renders the dashboard using a buffer to prevent flickering."""
    # Clear screen natively
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Build buffer
    buffer = []
    
    ns_color = COLOR_RED
    ew_color = COLOR_RED
    
    if intersection.active_phase == PHASE_NS:
        ns_color = COLOR_GREEN if intersection.state == 'GREEN' else COLOR_YELLOW
    else:
        ew_color = COLOR_GREEN if intersection.state == 'GREEN' else COLOR_YELLOW

    buffer.append("🚦 MODULAR TRAFFIC SIGNAL OPTIMIZATION SIMULATOR 🚦\n")
    buffer.append("=====================================================\n")
    buffer.append(f"Total Elapsed Time : {intersection.total_time:03d}s\n")
    buffer.append(f"Active Phase       : {intersection.active_phase}\n")
    buffer.append(f"Phase Timer        : {intersection.phase_timer:02d}s\n")
    buffer.append(f"State              : {intersection.state}\n")
    buffer.append("=====================================================\n\n")

    buffer.append(f"  🚗 North Queue: {intersection.queues['North']:3d} cars  [{ns_color}]\n")
    buffer.append(f"  🚗 South Queue: {intersection.queues['South']:3d} cars  [{ns_color}]\n")
    buffer.append("-----------------------------------------------------\n")
    buffer.append(f"  🚗 East Queue : {intersection.queues['East']:3d} cars  [{ew_color}]\n")
    buffer.append(f"  🚗 West Queue : {intersection.queues['West']:3d} cars  [{ew_color}]\n\n")

    buffer.append("=====================================================\n")
    buffer.append("📝 DECISION LOG:\n")
    buffer.append(f"> {decision_log}\n")
    buffer.append("=====================================================\n")
    buffer.append("[Press Ctrl+C to stop the simulation]\n")
    
    sys.stdout.write("".join(buffer))
    sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description="Modular Traffic Signal Optimization")
    parser.add_argument('--mode', type=str, choices=['heavy_traffic', 'night_mode', 'rush_hour', 'normal'], 
                        default='normal', help='Select the simulation scenario.')
    args = parser.parse_args()
    
    # Configure scenarios
    if args.mode == 'heavy_traffic':
        arrivals = {dir: (2, 5) for dir in ['North', 'South', 'East', 'West']}
        config = {'MIN_GREEN': 5, 'MAX_GREEN': 20, 'YELLOW_TIME': 2, 'THRESHOLD': 8}
    elif args.mode == 'night_mode':
        # Very sparse arrivals
        arrivals = {dir: (0, 1) for dir in ['North', 'South', 'East', 'West']}
        config = {'MIN_GREEN': 3, 'MAX_GREEN': 10, 'YELLOW_TIME': 2, 'THRESHOLD': 3}
    elif args.mode == 'rush_hour':
        # Heavy North/South, light East/West
        arrivals = {'North': (3, 6), 'South': (3, 6), 'East': (0, 2), 'West': (0, 2)}
        config = {'MIN_GREEN': 5, 'MAX_GREEN': 15, 'YELLOW_TIME': 2, 'THRESHOLD': 5}
    else:
        arrivals = {dir: (0, 2) for dir in ['North', 'South', 'East', 'West']}
        config = {'MIN_GREEN': 5, 'MAX_GREEN': 15, 'YELLOW_TIME': 2, 'THRESHOLD': 5}
        
    intersection = TrafficIntersection()
    controller = TrafficController(config)
    decision_log = f"System initialized in '{args.mode}' mode. Starting simulation..."
    
    try:
        while True:
            # 1. Render UI
            render_dashboard(intersection, decision_log)
            
            # 2. Wait 1 tick
            time.sleep(1)
            
            # 3. Step physics
            intersection.tick(arrivals)
            
            # 4. Step logic
            action, decision_log = controller.evaluate(intersection)
            
            if action == 'YELLOW':
                intersection.transition_to_yellow()
            elif action == 'SWITCH':
                intersection.switch_phase()
                
    except KeyboardInterrupt:
        sys.stdout.write('\n\n🛑 Simulation stopped by user.\n')
        sys.stdout.flush()

if __name__ == '__main__':
    main()

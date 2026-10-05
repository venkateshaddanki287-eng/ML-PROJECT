import sys
import argparse
import os
import joblib
import numpy as np
import pandas as pd

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from simulation.traffic_sim import TrafficSimulation
from simulation.controller import TrafficController
from simulation.intersection import PHASE_NS, PHASE_EW
from features.feature_engineering import FEATURE_NAMES


def _prompt_non_negative_int(label: str) -> int:
    while True:
        value = input(f"{label}: ").strip()
        try:
            number = int(value)
        except ValueError:
            print("Invalid input. Please enter a non-negative integer.")
            continue
        if number < 0:
            print("Invalid input. Vehicle counts and times must be non-negative.")
            continue
        return number


def _prompt_non_negative_float(label: str) -> float:
    while True:
        value = input(f"{label}: ").strip()
        try:
            number = float(value)
        except ValueError:
            print("Invalid input. Please enter a non-negative number.")
            continue
        if number < 0:
            print("Invalid input. Values must be non-negative.")
            continue
        return number


def _load_trained_model():
    model_path = os.path.join(os.path.dirname(__file__), 'models_saved', 'best_model.joblib')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run python train.py first.")
    return joblib.load(model_path)


def _traffic_condition_from_total(total_vehicles: float) -> str:
    if total_vehicles < 12:
        return 'LOW'
    if total_vehicles < 28:
        return 'MEDIUM'
    return 'HIGH'


def _build_feature_vector(input_payload: dict) -> pd.DataFrame:
    total_vehicles = float(input_payload['north_queue'] + input_payload['south_queue'] + input_payload['east_queue'] + input_payload['west_queue'])
    avg_wait = float(
        (
            input_payload['north_waiting_time']
            + input_payload['south_waiting_time']
            + input_payload['east_waiting_time']
            + input_payload['west_waiting_time']
        ) / 4.0
    )
    row = {
        'north_queue': float(input_payload['north_queue']),
        'south_queue': float(input_payload['south_queue']),
        'east_queue': float(input_payload['east_queue']),
        'west_queue': float(input_payload['west_queue']),
        'north_arrival_rate': float(input_payload['north_arrival_rate']),
        'south_arrival_rate': float(input_payload['south_arrival_rate']),
        'east_arrival_rate': float(input_payload['east_arrival_rate']),
        'west_arrival_rate': float(input_payload['west_arrival_rate']),
        'current_green_time': float(input_payload['current_green_time']),
        'previous_queue': float(total_vehicles),
        'queue_growth': float(0.0),
        'waiting_time': avg_wait,
        'traffic_density': float(min(1.0, total_vehicles / 100.0)),
        'current_phase': 0.0 if input_payload['current_phase'] == 'North/South' else 1.0,
    }
    return pd.DataFrame([row], columns=FEATURE_NAMES)


def _run_simulation_for_mode(input_payload: dict, mode: str, model=None):
    sim = TrafficSimulation(mode=mode, scenario='normal')
    sim.arrivals = {
        direction: (int(max(0, rate)), int(max(0, rate)))
        for direction, rate in {
            'North': input_payload['north_arrival_rate'],
            'South': input_payload['south_arrival_rate'],
            'East': input_payload['east_arrival_rate'],
            'West': input_payload['west_arrival_rate'],
        }.items()
    }
    sim.config = {'MIN_GREEN': 5, 'MAX_GREEN': 20, 'YELLOW_TIME': 2, 'THRESHOLD': 5}
    sim.controller = TrafficController(sim.config, mode=mode, ml_model=model)
    sim.intersection.queues = {
        'North': int(input_payload['north_queue']),
        'South': int(input_payload['south_queue']),
        'East': int(input_payload['east_queue']),
        'West': int(input_payload['west_queue']),
    }
    sim.intersection.active_phase = PHASE_NS if input_payload['current_phase'] == 'North/South' else PHASE_EW
    sim.intersection.state = 'GREEN'
    sim.intersection.phase_timer = int(input_payload['current_green_time'])
    sim.intersection.total_time = 0
    sim.intersection.phase_switch_count = 0
    sim.intersection.total_cars_processed = 0
    sim.intersection.total_wait_accumulated = 0.0
    sim.intersection.max_queue_observed = max(sim.intersection.queues.values())

    queue_history = []
    wait_history = []

    for _ in range(int(input_payload['simulation_steps'])):
        queue_history.append(float(sum(sim.intersection.queues.values())))
        wait_history.append(float(sim.intersection.total_wait_accumulated / max(1, sim.intersection.total_cars_processed + max(1, sum(sim.intersection.queues.values())))))
        sim.intersection.tick(sim.arrivals)
        action, _ = sim.controller.evaluate(sim.intersection, sim.arrivals)
        if action == 'YELLOW':
            sim.intersection.transition_to_yellow()
        elif action == 'SWITCH':
            sim.intersection.switch_phase()

    final_total = float(sum(sim.intersection.queues.values()))
    metrics = {
        'total_vehicles': final_total,
        'average_queue_length': float(np.mean(queue_history)) if queue_history else 0.0,
        'average_waiting_time': float(np.mean(wait_history)) if wait_history else 0.0,
        'maximum_queue_length': float(max(queue_history)) if queue_history else 0.0,
        'throughput': float(sim.intersection.total_cars_processed),
        'signal_switches': int(sim.intersection.phase_switch_count),
    }
    return metrics


def _interactive_mode():
    print("\nINTERACTIVE TRAFFIC SIGNAL OPTIMIZATION")
    print("=" * 50)

    while True:
        signal_choice = input("Current signal direction [North/South or East/West]: ").strip()
        if signal_choice in ['North/South', 'East/West']:
            break
        print("Invalid choice. Please enter exactly 'North/South' or 'East/West'.")

    current_phase = signal_choice
    north = _prompt_non_negative_int('North vehicles')
    south = _prompt_non_negative_int('South vehicles')
    east = _prompt_non_negative_int('East vehicles')
    west = _prompt_non_negative_int('West vehicles')

    current_green_time = _prompt_non_negative_int('Current green time (seconds)')
    north_wait = _prompt_non_negative_float('North waiting time')
    south_wait = _prompt_non_negative_float('South waiting time')
    east_wait = _prompt_non_negative_float('East waiting time')
    west_wait = _prompt_non_negative_float('West waiting time')
    north_arr = _prompt_non_negative_float('North arrival rate')
    south_arr = _prompt_non_negative_float('South arrival rate')
    east_arr = _prompt_non_negative_float('East arrival rate')
    west_arr = _prompt_non_negative_float('West arrival rate')
    steps = _prompt_non_negative_int('Number of simulation steps')

    input_payload = {
        'north_queue': north,
        'south_queue': south,
        'east_queue': east,
        'west_queue': west,
        'current_green_time': current_green_time,
        'north_waiting_time': north_wait,
        'south_waiting_time': south_wait,
        'east_waiting_time': east_wait,
        'west_waiting_time': west_wait,
        'north_arrival_rate': north_arr,
        'south_arrival_rate': south_arr,
        'east_arrival_rate': east_arr,
        'west_arrival_rate': west_arr,
        'current_phase': current_phase,
        'simulation_steps': steps,
    }

    print("\nTRAFFIC INPUT")
    print(f"North: {north} vehicles")
    print(f"South: {south} vehicles")
    print(f"East: {east} vehicles")
    print(f"West: {west} vehicles")
    print("\nCURRENT SIGNAL")
    print(current_phase)

    try:
        model = _load_trained_model()
    except FileNotFoundError as exc:
        print(f"\nUnable to run interactive mode: {exc}")
        return

    feature_df = _build_feature_vector(input_payload)
    probs = model.predict_proba(feature_df)[0]
    pred_index = int(np.argmax(probs))
    confidence = float(probs[pred_index])
    target_map = {0: 'KEEP', 1: 'SWITCH_TO_NS', 2: 'SWITCH_TO_EW'}
    predicted_action = target_map.get(pred_index, 'KEEP')
    traffic_condition = _traffic_condition_from_total(north + south + east + west)

    print("\nML PREDICTION")
    print(f"Traffic condition: {traffic_condition}")
    print(f"Predicted action: {predicted_action}")
    print(f"Confidence: {confidence:.4f}")

    before_total = float(north + south + east + west)
    before_avg_queue = before_total / 4.0
    before_wait = float((north_wait + south_wait + east_wait + west_wait) / 4.0)
    print("\nBEFORE OPTIMIZATION")
    print(f"- total vehicles: {before_total:.2f}")
    print(f"- average queue: {before_avg_queue:.2f}")
    print(f"- waiting time: {before_wait:.2f}")

    ml_metrics = _run_simulation_for_mode(input_payload, mode='ml_based', model=model)
    print("\nAFTER ML OPTIMIZATION")
    print(f"- total vehicles: {ml_metrics['total_vehicles']:.2f}")
    print(f"- average queue: {ml_metrics['average_queue_length']:.2f}")
    print(f"- waiting time: {ml_metrics['average_waiting_time']:.2f}")
    print(f"- throughput: {ml_metrics['throughput']:.2f}")
    print(f"- signal switches: {ml_metrics['signal_switches']}")

    rule_metrics = _run_simulation_for_mode(input_payload, mode='rule_based', model=None)
    print("\nOPTIMIZATION COMPARISON")
    print(f"{'Metric':<25}{'Rule-Based':>18}{'ML-Based':>18}")
    print('-' * 62)
    metrics_to_compare = [
        ('Average Waiting Time', 'average_waiting_time'),
        ('Average Queue Length', 'average_queue_length'),
        ('Maximum Queue Length', 'maximum_queue_length'),
        ('Throughput', 'throughput'),
        ('Signal Switches', 'signal_switches'),
    ]

    percentage_change = {}
    for label, key in metrics_to_compare:
        rule_val = float(rule_metrics[key])
        ml_val = float(ml_metrics[key])
        if abs(rule_val) > 1e-12:
            pct = ((ml_val - rule_val) / rule_val) * 100.0
        else:
            pct = None
        percentage_change[key] = pct
        if key in ['average_waiting_time', 'average_queue_length', 'maximum_queue_length']:
            print(f"{label:<25}{rule_val:>17.2f}{ml_val:>17.2f}")
        else:
            print(f"{label:<25}{rule_val:>17.2f}{ml_val:>17.2f}")

    print("\nPERCENTAGE CHANGE (ML vs Rule-Based)")
    for label, key in metrics_to_compare:
        pct = percentage_change[key]
        change_text = f"{pct:+.2f}%" if pct is not None else "N/A (baseline 0)"
        print(f"{label:<25}{change_text:>18}")

    summary_conditions = [
        (ml_metrics['average_waiting_time'] < rule_metrics['average_waiting_time']),
        (ml_metrics['average_queue_length'] <= rule_metrics['average_queue_length']),
        (ml_metrics['throughput'] >= rule_metrics['throughput']),
    ]
    if all(summary_conditions):
        print("\nResult: ML improved the measured traffic outcome for this scenario.")
    else:
        print("\nResult: ML did not outperform the rule-based baseline on this scenario based on the measured metrics.")

    print("\nML signal decision is applied through the live simulation controller using the trained pipeline.")


def main():
    parser = argparse.ArgumentParser(description="Modular Traffic Signal Optimization Simulator")
    parser.add_argument('--scenario', type=str, choices=['heavy_traffic', 'night_mode', 'rush_hour', 'normal'],
                        default='normal', help='Select simulation scenario bounds.')
    parser.add_argument('--mode', type=str, choices=['rule_based', 'ml_based', 'interactive', 'visual'],
                        default='rule_based', help='Select controller mode or launch the visual simulator.')
    parser.add_argument('--controller', choices=['ml_based', 'rule_based'], default='ml_based',
                        help='Controller used by --mode visual (default: ml_based).')
    args = parser.parse_args()

    if args.mode == 'interactive':
        _interactive_mode()
        return

    if args.mode == 'visual':
        sim = TrafficSimulation(mode=args.controller, scenario=args.scenario)
        if args.controller == 'ml_based':
            try:
                sim.controller.ml_model = _load_trained_model()
            except FileNotFoundError as exc:
                parser.error(str(exc))
        sim.run_visual()
        return

    sim = TrafficSimulation(mode=args.mode, scenario=args.scenario)
    sim.run()


if __name__ == '__main__':
    main()

"""
Dataset Generation Script.
Simulates traffic flow across multiple traffic scenarios to generate a reproducible dataset
for classification and regression training.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
from typing import List, Dict

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from simulation.intersection import TrafficIntersection, PHASE_NS, PHASE_EW
from simulation.controller import TrafficController
from features.feature_engineering import FEATURE_NAMES, CATEGORICAL_TARGET, REGRESSION_TARGET


def set_seed(seed: int = 42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def generate_dataset(num_samples_per_scenario: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Runs traffic simulations across multiple scenarios and collects state snapshots.
    """
    set_seed(seed)
    
    scenarios = {
        'normal': {
            'arrivals': {d: (0, 2) for d in ['North', 'South', 'East', 'West']},
            'config': {'MIN_GREEN': 5, 'MAX_GREEN': 15, 'YELLOW_TIME': 2, 'THRESHOLD': 5}
        },
        'heavy_traffic': {
            'arrivals': {d: (2, 5) for d in ['North', 'South', 'East', 'West']},
            'config': {'MIN_GREEN': 5, 'MAX_GREEN': 20, 'YELLOW_TIME': 2, 'THRESHOLD': 8}
        },
        'night_mode': {
            'arrivals': {d: (0, 1) for d in ['North', 'South', 'East', 'West']},
            'config': {'MIN_GREEN': 3, 'MAX_GREEN': 10, 'YELLOW_TIME': 2, 'THRESHOLD': 3}
        },
        'rush_hour_ns': {
            'arrivals': {'North': (3, 6), 'South': (3, 6), 'East': (0, 2), 'West': (0, 2)},
            'config': {'MIN_GREEN': 5, 'MAX_GREEN': 15, 'YELLOW_TIME': 2, 'THRESHOLD': 5}
        },
        'rush_hour_ew': {
            'arrivals': {'North': (0, 2), 'South': (0, 2), 'East': (3, 6), 'West': (3, 6)},
            'config': {'MIN_GREEN': 5, 'MAX_GREEN': 15, 'YELLOW_TIME': 2, 'THRESHOLD': 5}
        }
    }

    rows: List[Dict] = []

    for name, s_info in scenarios.items():
        intersection = TrafficIntersection()
        controller = TrafficController(s_info['config'], mode='rule_based')
        arrivals = s_info['arrivals']

        for step in range(num_samples_per_scenario):
            # Extract features before step
            feats = intersection.get_features(arrivals)

            # Determine baseline action
            action, _ = controller.evaluate(intersection, arrivals)
            
            # Map action to target label
            # Target action encoding: 0 = KEEP, 1 = SWITCH_TO_NS, 2 = SWITCH_TO_EW
            active_phase = intersection.active_phase
            target_action = 0  # KEEP
            if action in ['YELLOW', 'SWITCH']:
                target_action = 2 if active_phase == PHASE_NS else 1

            # Traffic condition classification label: 0=LOW, 1=MEDIUM, 2=HIGH
            total_q = sum(intersection.queues.values())
            if total_q < 12:
                condition_label = 0 # LOW
            elif total_q < 28:
                condition_label = 1 # MEDIUM
            else:
                condition_label = 2 # HIGH

            # Regression target: future waiting time estimate
            active_tot, wait_tot = intersection.get_totals()
            future_wait = float(wait_tot * 1.5 + (0.5 * total_q) + (0.8 * feats['current_green_time']))

            row = feats.copy()
            row[CATEGORICAL_TARGET] = target_action
            row['traffic_condition'] = condition_label
            row[REGRESSION_TARGET] = round(future_wait, 2)
            row['scenario'] = name

            rows.append(row)

            # Advance physics & state
            intersection.tick(arrivals)
            if action == 'YELLOW':
                intersection.transition_to_yellow()
            elif action == 'SWITCH':
                intersection.switch_phase()

    df = pd.DataFrame(rows)
    return df


def main():
    print("🚦 Generating synthetic traffic dataset from simulation...")
    df = generate_dataset(num_samples_per_scenario=1000, seed=42)
    
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    raw_path = 'data/raw/traffic_dataset_raw.csv'
    processed_path = 'data/processed/traffic_dataset.csv'

    df.to_csv(raw_path, index=False)
    
    # Processed: drop identifier columns if any, preserve clean feature columns + targets
    df.to_csv(processed_path, index=False)

    print(f"✅ Dataset successfully generated!")
    print(f"   Raw dataset: {raw_path} ({len(df)} rows)")
    print(f"   Processed dataset: {processed_path}")
    print(f"   Features: {len(FEATURE_NAMES)}")
    print(f"   Target distribution ({CATEGORICAL_TARGET}):\n{df[CATEGORICAL_TARGET].value_counts().to_dict()}")
    print(f"   Target distribution (traffic_condition):\n{df['traffic_condition'].value_counts().to_dict()}")


if __name__ == '__main__':
    main()

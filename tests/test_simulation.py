import pytest
from simulation.intersection import TrafficIntersection, PHASE_NS
from simulation.controller import TrafficController


def test_intersection_initialization():
    inter = TrafficIntersection()
    assert inter.queues['North'] == 0
    assert inter.active_phase == PHASE_NS
    assert inter.state == 'GREEN'


def test_controller_evaluation():
    config = {'MIN_GREEN': 5, 'MAX_GREEN': 15, 'YELLOW_TIME': 2, 'THRESHOLD': 5}
    inter = TrafficIntersection()
    controller = TrafficController(config, mode='rule_based')

    # Initial state -> MIN GREEN lock
    action, _ = controller.evaluate(inter)
    assert action == 'MAINTAIN'

    # Timer beyond MIN GREEN with high waiting queue
    inter.phase_timer = 6
    inter.queues['East'] = 20
    action, _ = controller.evaluate(inter)
    assert action == 'YELLOW'


def test_tick_tracks_maximum_total_queue():
    inter = TrafficIntersection()
    inter.tick({'North': (4, 4), 'South': (0, 0), 'East': (0, 0), 'West': (0, 0)})

    assert inter.max_queue_observed == sum(inter.queues.values())
    assert inter.max_queue_observed > 0

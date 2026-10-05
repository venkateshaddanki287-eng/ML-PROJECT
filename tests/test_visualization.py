from simulation.intersection import PHASE_EW, PHASE_NS
from simulation.traffic_sim import get_light_states


def test_signal_light_pairs_follow_intersection_state():
    assert get_light_states(PHASE_NS, 'GREEN') == {
        PHASE_NS: 'GREEN',
        PHASE_EW: 'RED',
    }
    assert get_light_states(PHASE_NS, 'YELLOW') == {
        PHASE_NS: 'YELLOW',
        PHASE_EW: 'RED',
    }
    assert get_light_states(PHASE_EW, 'GREEN') == {
        PHASE_NS: 'RED',
        PHASE_EW: 'GREEN',
    }
    assert get_light_states(PHASE_EW, 'YELLOW') == {
        PHASE_NS: 'RED',
        PHASE_EW: 'YELLOW',
    }


def test_unknown_signal_state_fails_closed_to_red():
    assert get_light_states(PHASE_NS, 'UNKNOWN') == {
        PHASE_NS: 'RED',
        PHASE_EW: 'RED',
    }
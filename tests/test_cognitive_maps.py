"""Tests for fuzzy cognitive map simulation and attractor analysis."""

import numpy as np
import pandas as pd
import pytest

import cognitive_maps as cm


# Concept 0 excites 1, 1 excites 2, 2 excites 0: a three-step limit cycle.
CYCLE = np.array([[0, 1.0, 0], [0, 0, 1.0], [1.0, 0, 0]])
# Every concept excites itself: any binary state is a fixed point.
IDENTITY = np.eye(3)


def test_threshold_maps_connections_to_trivalent_values():
    connections = np.array([[0.9, -0.9, 0.2], [-0.2, 0.5, -0.51]])
    expected = np.array([[1, -1, 0], [0, 0, -1]])
    np.testing.assert_array_equal(cm._threshold_modules(connections, 0.5), expected)


def test_step_and_simulate_follow_the_cycle():
    fcm = cm.FuzzyCognitiveMap(CYCLE)
    np.testing.assert_array_equal(fcm.state, np.zeros(3))
    assert fcm.var_names() is None

    np.testing.assert_array_equal(fcm.set_and_step(np.array([1, 0, 0])), [0, 1, 0])
    np.testing.assert_array_equal(fcm.step(), [0, 0, 1])
    np.testing.assert_array_equal(fcm.set_and_simulate(np.array([1, 0, 0]), 3), [1, 0, 0])
    np.testing.assert_array_equal(fcm.simulate(0), [1, 0, 0])

    fcm.clear_state()
    np.testing.assert_array_equal(fcm.state, np.zeros(3))


def test_negative_connections_propagate_inhibition():
    fcm = cm.FuzzyCognitiveMap(np.array([[0, -1.0], [0, 0]]))
    np.testing.assert_array_equal(fcm.set_and_step(np.array([1, 0])), [0, -1])


def test_dataframe_connections_label_the_states():
    frame = pd.DataFrame(CYCLE, columns=list('abc'), index=list('abc'))
    fcm = cm.FuzzyCognitiveMap(frame, threshold=0.2)
    assert fcm.var_names() == ['a', 'b', 'c']

    state = fcm.set_and_step(pd.Series([1, 0, 0], index=list('abc')))
    assert isinstance(state, pd.Series)
    assert list(state.index) == ['a', 'b', 'c']
    np.testing.assert_array_equal(state.to_numpy(), [0, 1, 0])
    pd.testing.assert_series_equal(fcm.simulate(2), pd.Series([1, 0, 0], index=list('abc')), check_dtype=False)


def test_adding_maps_sums_connections_and_keeps_threshold():
    first = cm.FuzzyCognitiveMap(CYCLE, threshold=0.7)
    second = cm.FuzzyCognitiveMap(IDENTITY)

    combined = first + second
    np.testing.assert_array_equal(combined.connections, CYCLE + IDENTITY)
    assert combined.threshold == 0.7

    first.add(second)
    np.testing.assert_array_equal(first.connections, CYCLE + IDENTITY)


def test_attractor_and_period_detection():
    fixed = [np.array([1, 0])] * 4
    assert cm._look_attractors(fixed)[0]
    np.testing.assert_array_equal(cm._look_attractors(fixed)[1][0], [1, 0])

    cycle = [np.array([1, 0]), np.array([0, 1])] * 3
    assert cm._look_attractors(cycle) == (False, [])
    period = cm._look_periods(cycle)
    assert len(period) == 2
    np.testing.assert_array_equal(period[1], [0, 1])

    aperiodic = [np.array([ix]) for ix in range(6)]
    assert cm._look_periods(aperiodic) is None
    assert cm._look_periods(cycle * 2, max_period_len=1) is None


def test_look_pattern_states_finds_fixed_points_and_cycles():
    fixed = cm.FuzzyCognitiveMap(IDENTITY)
    fixed.set_state(np.array([1, 0, 1]))
    pattern = cm.look_pattern_states(fixed, sim_steps=2, pattern_len=5)
    assert len(pattern) == 1
    np.testing.assert_array_equal(pattern[0], [1, 0, 1])

    cycle = cm.FuzzyCognitiveMap(CYCLE)
    cycle.set_state(np.array([1, 0, 0]))
    pattern = cm.look_pattern_states(cycle, sim_steps=1, pattern_len=9)
    assert len(pattern) == 3

    cycle.set_state(np.array([1, 0, 0]))
    assert cm.look_pattern_states(cycle, sim_steps=0, pattern_len=9, max_period_size=2) is None


def test_study_and_report_attractors():
    np.random.seed(0)
    attractors = cm.study_attractors_FCM(cm.FuzzyCognitiveMap(IDENTITY), max_steps=2, random_inits=6)
    assert 1 <= len(attractors) <= 6
    for init_state, pattern in attractors.items():
        assert isinstance(init_state, tuple)
        np.testing.assert_array_equal(pattern[0], init_state)

    patterns, states = cm.attractors_report({
        (0, 0): None,
        (1, 0): [np.array([1, 0])],
        (0, 1): [np.array([1, 0]), np.array([0, 1])],
    })
    assert patterns['Chaotic'] == pytest.approx(1 / 3)
    assert sum(patterns.values()) == pytest.approx(1.0)
    assert states[str(np.array([1, 0]))] == pytest.approx(2 / 3)
    assert states[str(np.array([0, 1]))] == pytest.approx(1 / 3)

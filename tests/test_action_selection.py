import pytest
import numpy as np
from scipy.stats import mode
from multi_agent_mdp.algorithms.action_selection import *

@pytest.fixture
def q_values_fixture():
    q_vals = np.zeros((5, 4))  # 5 states, 4 actions
    q_vals[0, :] = np.array([4.5, 3.5, 2.5, 1.5])
    q_vals[1, :] = np.array([0, 0, 0, 0])
    q_vals[2, :] = np.array([1, 1, 1, 1])
    q_vals[3, :] = np.array([200, 50, 2, 1])
    q_vals[4, :] = np.array([-20, 0, 20, 1])
    return q_vals

@pytest.fixture()
def q_values_increasing():
    q_values = np.zeros((2, 5))
    q_values[0, :] = np.arange(0, 5)
    q_values[1, :] = np.arange(4, -1, -1)

    return q_values

@pytest.fixture()
def q_values_equal():
    q_values = np.ones((2, 5))
    return q_values

def test_max_action_selector_action_p(q_values_fixture):

    selector = MaxActionSelector()
    action_p = selector.get_pi_p(q_values_fixture)

    expected_action_p = np.zeros((5, 4))  # 5 states, 4 actions
    expected_action_p[0, 0] = 1
    expected_action_p[1, 0] = 1
    expected_action_p[2, 0] = 1
    expected_action_p[3, 0] = 1
    expected_action_p[4, 2] = 1

    assert np.all(action_p == expected_action_p)

def test_max_action_selector_action(q_values_fixture):

    selector = MaxActionSelector()
    action = selector.get_pi(q_values_fixture)

    expected_action = np.array([0, 0, 0, 0, 2])

    assert np.all(action == expected_action)


def test_softmax_action_selector_action_p(q_values_fixture):

    selector = SoftmaxActionSelector(temperature=1, seed=123)
    action_p = selector.get_pi_p(q_values_fixture)

    assert np.all(np.diff(action_p[0, :]) < 0)
    assert np.all(action_p[1, :] == action_p[1, 0])
    assert np.all(action_p[2, :] == action_p[1, 0])
    assert np.all(action_p >= 0)
    assert np.all(action_p <= 1)

def test_softmax_action_selector_action_p(q_values_fixture):

    selector = SoftmaxActionSelector(temperature=1, seed=123)
    action_p = selector.get_pi_p(q_values_fixture)

    assert np.all(np.diff(action_p[0, :]) < 0)
    assert np.all(action_p[1, :] == action_p[1, 0])
    assert np.all(action_p[2, :] == action_p[1, 0])
    assert np.all(action_p >= 0)
    assert np.all(action_p <= 1)

def test_softmax_action_selector_temperature_action_p(q_values_increasing):

    temp_action_p = np.zeros((3, 2, 5))

    for n, temp in enumerate([0.5, 1, 5]):
            selector = SoftmaxActionSelector(temperature=temp)
            action_p = selector.get_pi_p(q_values_increasing)

            assert np.all(np.diff(action_p[0, :]) > 0)
            assert np.all(np.diff(action_p[1, :]) < 0)
            assert np.all(np.isclose(action_p[0, ::-1], action_p[1]))

            temp_action_p[n, ...] = action_p

    assert np.all(np.diff(temp_action_p[:, 0, 1]) > 0)
    assert np.all(np.diff(temp_action_p[:, 1, 1]) > 0)

def test_softmax_action_selector_action(q_values_increasing):

    selector = SoftmaxActionSelector(seed=123)

    simulated_actions = np.zeros((2, 2000))

    for i in range(2000):
        simulated_actions[:, i] = selector.get_pi(q_values_increasing)

    assert ~np.all(simulated_actions[0, :] == 4)
    assert ~np.all(simulated_actions[1, :] == 0)

    assert mode(simulated_actions, axis=1)[0][0] == 4
    assert mode(simulated_actions, axis=1)[0][1] == 0


def test_softmax_action_selector_gives_random_results(q_values_equal):

    selector_seeded = SoftmaxActionSelector(seed=123)

    simulated_actions_seeded = np.zeros((2, 2000))

    for i in range(2000):
        simulated_actions_seeded[:, i] = selector_seeded.get_pi(q_values_equal)

    assert np.all(np.isin(np.arange(5), simulated_actions_seeded))

    selector_unseeded = SoftmaxActionSelector(seed=None)

    simulated_actions_unseeded = np.zeros((2, 5000))

    for i in range(5000):
        simulated_actions_unseeded[:, i] = selector_unseeded.get_pi(q_values_equal)

    assert np.all(np.isin(np.arange(5), simulated_actions_unseeded))
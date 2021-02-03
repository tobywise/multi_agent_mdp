import pytest
import numpy as np
from maMDP.algorithms.dynamic_programming import *

@pytest.fixture
def sas_fixture():
    """ Simple transition function, 4 states in a square grid """
    sas = np.zeros((4, 4, 4))
    sas[(0, 0, 1, 1, 2, 2, 3, 3), 
        (0, 1, 1, 2, 0, 3, 2, 3),
        (1, 2, 3, 0, 3, 0, 2, 1)] = 1
    return sas

@pytest.fixture
def reward_fixture():
    """ Reward for 4 state MDP, reward of 1 for single state, 0 elsewhere """
    return np.array([0, 0, 0, 1])

@pytest.mark.parametrize("s,values,discount,correct_values", [
    (0, np.zeros(4), 0.5, np.zeros(4)),
    (1, np.zeros(4), 0.5, np.array([0, 1, 0, 0])),
    (2, np.zeros(4), 0.5, np.array([1, 0, 0, 0])),
    (1, np.ones(4), 0.5, np.array([0, 1.5, 0.5, 0])),
])
def test_state_action_values(sas_fixture, reward_fixture, s, values, discount, correct_values):
    action_values = get_state_action_values(s, sas_fixture.shape[1], sas_fixture, reward_fixture, discount, values)
    assert np.all(action_values == correct_values)

@pytest.mark.parametrize("old_delta,old_v,new_v,correct", [
    (0.5, 0, 0, 0.5),
    (0.5, 1.5, 0.5, 1),
])
def test_delta_update(old_delta, old_v, new_v, correct):
    assert delta_update(old_delta, old_v, new_v) == correct


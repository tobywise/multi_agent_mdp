from inspect import getargvalues
from typing import ValuesView
from numpy.lib.function_base import extract
from numpy.lib.ufunclike import fix
import pytest
import numpy as np
from multi_agent_mdp.algorithms.mcts import *

@pytest.fixture
def sas_fixture():
    """ Simple transition function, 4 states in a square grid """
    sas = np.zeros((4, 4, 4))
    sas[(0, 0, 1, 1, 2, 2, 3, 3), 
        (0, 1, 1, 2, 0, 3, 2, 3),
        (1, 2, 3, 0, 3, 0, 2, 1)] = 1
    return sas

@pytest.fixture
def example_agent_info():
    return {'testAgent1': (2, 3, [1, 4], np.array([0, 1, 0, 1, 2]), 2),
            'testAgent2': (22, 1, [0], np.array([1, 1, 0, 0, 0]), 3),
            'testAgent3': (10, 3, [], np.array([0, 1, 0, 0, 0]), 4)}

@pytest.fixture
def example_info_for_agent_values():
    agents = range(3)
    agent_idx = {2: 0, 3: 1, 4: 2}
    primary_agent_reward_function = np.array([0, 1, 0, 2, 3])
    return (agents, agent_idx, primary_agent_reward_function)

@pytest.fixture
def example_info_for_agent_consumes():
    consumes_features = ((1, 3, 4), 
                         (), 
                         (0, 2))    
    agent_idx = {2: 0, 3: 1, 4: 2}
    return consumes_features, agent_idx

@pytest.fixture
def state_value_fixture():
    V = np.array([1, 1, 1, 1, 5, 2, 1, 3, 2, 0])
    return V

@pytest.fixture
def state_visitation_fixture():
    N = np.array([0, 10, 3, 5, 3, 3, 3, 3, 10, 2])
    return N

def test_extract_agent_info(example_agent_info):
    single_agent_info = dict([list(example_agent_info.items())[0]])
    agent_idx, current_node, n_moves, consumes_features, reward_functions = extract_agent_info(single_agent_info)

    assert agent_idx == {2: 0}
    assert current_node == np.array([2])
    assert n_moves == (3, )

    expected_consumes_features = np.zeros((1, 5))
    expected_consumes_features[0, [1, 4]] = 1

    assert np.all(consumes_features == expected_consumes_features)
    assert reward_functions.shape == (1, 5)
    assert np.all(reward_functions[0, :] == np.array([0, 1, 0, 1, 2]))


def test_get_agent_values(example_info_for_agent_values):
    agents, agent_idx, primary_agent_reward_function = example_info_for_agent_values

    agent_values = get_agent_values(agents, agent_idx, primary_agent_reward_function)

    assert agent_values == (0, 2, 3)

def test_get_agent_consumes(example_info_for_agent_consumes):

    consumes_features, agent_idx = example_info_for_agent_consumes

    consumes_agents = get_agent_consumes(consumes_features, agent_idx)

    expected = np.zeros((3, 3))
    expected[0, 1] = 1
    expected[0, 2] = 1
    expected[2, 0] = 1
    expected = expected.astype(bool)

    assert np.all(consumes_agents == expected)

def test_get_actions_states(sas_fixture):

    actions_states, actions, states = get_actions_states(sas_fixture, 0)

    assert np.all(actions == np.array([0, 1]))
    assert np.all(states == np.array([1, 2]))

def test_UCB(state_value_fixture, state_visitation_fixture):

    V = state_value_fixture
    N = state_visitation_fixture

    # Should be determined purely by N
    ucb1 = UCB(V, N, np.array([0, 1, 2, 3]), 1, 5)

    # Should be determined purely by V
    ucb2 = UCB(V, N, np.array([4, 5, 6, 7]), 1, 1)

    assert np.all(ucb1.argsort().argsort() == np.array([3, 0, 2, 1]))
    assert np.all(ucb2.argsort().argsort() == np.array([3, 1, 0, 2]))

def test_UCB_C(state_value_fixture, state_visitation_fixture):

    V = state_value_fixture
    N = state_visitation_fixture

    # Should be determined purely by N
    ucb1 = UCB(V, N, np.array([0, 1, 2, 3]), 1, 5)

    # Should be determined purely by V
    ucb2 = UCB(V, N, np.array([0, 1, 2, 3]), 10, 5)

    assert ucb2[0] - ucb1[0] > ucb2[1] - ucb1[1]  # Increase for unvisited should be higher

def test_MCTS_next_node(state_value_fixture, state_visitation_fixture):

    V = state_value_fixture
    N = state_visitation_fixture

    # Should choose first (unvisited) state
    next_node, expand = MCTS_next_node(True, V, N, np.array([0, 1, 2, 3]), 1, 5)
    assert next_node == 0
    assert expand == False

    # Should choose according to UCT
    next_node, expand = MCTS_next_node(True, V, N, np.array([4, 5, 6, 7]), 1, 2)
    assert next_node == 4
    assert expand == True

    # Should be random
    nodes = []
    for _ in range(1000):
        next_node, expand = MCTS_next_node(False, V, N, np.array([4, 5, 6, 7]), 1, 2)
        nodes.append(next_node)

    assert np.all(np.isin(np.arange(4, 8), nodes))
    assert expand ==False

def test_get_MCTS_action_values(sas_fixture):

    V = np.array([1, 1, 1, 1])
    N = np.array([1, 0, 1, 1])

    _, action_values, _ = get_MCTS_action_values(0, sas_fixture, V, N)

    assert len(action_values) == 2
    assert action_values[0] > action_values[1]

def test_check_agent_overlap():
    consumes_agents = np.zeros((3, 3))
    consumes_agents[0, 1] = 1
    consumes_agents[2, 0] = 1
    consumes_agents = consumes_agents.astype(bool)
    
    current_node = np.zeros(3)
    current_node[1] = 10
    caught_cost = -50
    agents_active = np.ones(3).astype(bool)
    agent_values = np.array([0, 15, 0])

    added_reward, agents_active, caught = check_agent_overlap(consumes_agents, current_node, caught_cost, 
                                                              agents_active, agent_values)

    assert added_reward == -50
    assert np.all(agents_active == np.array([False, True, True]))
    assert caught == True

    agents_active = np.ones(3).astype(bool)
    current_node[0] = 10
    added_reward, agents_active, caught = check_agent_overlap(consumes_agents, current_node, caught_cost, 
                                                              agents_active, agent_values)

    
    assert added_reward == 15
    assert np.all(agents_active == np.array([True, False, True]))
    assert caught == False
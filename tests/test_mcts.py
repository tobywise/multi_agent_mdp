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
    return {'testAgent1': (2, 3, [1, 4], np.array([0, 1, 0, 1, 2]), 0),
            'testAgent2': (22, 1, [0], np.array([1, 1, 0, 0, 0]), 1),
            'testAgent3': (10, 3, [], np.array([0, 1, 0, 0, 0]), 2)}

@pytest.fixture
def example_info_for_agent_values():
    names = ['testAgent1', 'testAgent2', 'testAgent3']
    primary_agent_name = 'testAgent1'
    agent_idx = {2: 'testAgent1', 3: 'testAgent2', 4: 'testAgent3'}
    primary_agent_reward_function = np.array([0, 1, 0, 2, 3])
    return (names, primary_agent_name, agent_idx, primary_agent_reward_function)

@pytest.fixture
def example_info_for_agent_consumes():
    consumes_features = {'testAgent1': [1, 3, 4], 
                         'testAgent2': [], 
                         'testAgent3': [0, 2]}    
    agent_idx = {2: 'testAgent1', 3: 'testAgent2', 4: 'testAgent3'}
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
    names, agent_idx, current_node, n_moves, consumes_features, reward_functions = extract_agent_info(single_agent_info)

    assert names == ['testAgent1']
    assert agent_idx == {0: 'testAgent1'}
    assert current_node == {'testAgent1': 2}
    assert n_moves == {'testAgent1': 3}
    assert consumes_features == {'testAgent1': [1, 4]}
    assert list(reward_functions.keys()) == ['testAgent1']
    assert np.all(list(reward_functions.values()) == np.array([0, 1, 0, 1, 2]))


def test_get_agent_values(example_info_for_agent_values):
    names, primary_agent_name, agent_idx, primary_agent_reward_function = example_info_for_agent_values

    agent_values = get_agent_values(names, primary_agent_name, agent_idx, primary_agent_reward_function)

    assert agent_values == {'testAgent2': 2, 'testAgent3': 3}

def test_get_agent_consumes(example_info_for_agent_consumes):

    consumes_features, agent_idx = example_info_for_agent_consumes

    consumes_agents = get_agent_consumes(consumes_features, agent_idx)

    assert consumes_agents == {'testAgent1': ['testAgent2', 'testAgent3'], 
                               'testAgent2': [],
                               'testAgent3': ['testAgent1']}

def test_get_agent_order(example_info_for_agent_values):

    agent_names, _, _, _  = example_info_for_agent_values

    agent_order = get_agent_order(agent_names)

    assert agent_order == {'testAgent1': 'testAgent2',
                           'testAgent2': 'testAgent3',
                           'testAgent3': 'testAgent1'}

def test_get_actions_states(sas_fixture):

    actions_states, actions, states = get_actions_states(sas_fixture, 0)

    assert np.all(actions == np.array([0, 1]))
    assert np.all(states == np.array([1, 2]))

def test_UCB(state_value_fixture, state_visitation_fixture):

    V = state_value_fixture
    N = state_visitation_fixture

    # Should be determined purely by N
    ucb1 = UCB(V, N, [0, 1, 2, 3], 1, 5)

    # Should be determined purely by V
    ucb2 = UCB(V, N, [4, 5, 6, 7], 1, 1)

    assert np.all(ucb1.argsort().argsort() == np.array([3, 0, 2, 1]))
    assert np.all(ucb2.argsort().argsort() == np.array([3, 1, 0, 2]))

def test_UCB_C(state_value_fixture, state_visitation_fixture):

    V = state_value_fixture
    N = state_visitation_fixture

    # Should be determined purely by N
    ucb1 = UCB(V, N, [0, 1, 2, 3], 1, 5)

    # Should be determined purely by V
    ucb2 = UCB(V, N, [0, 1, 2, 3], 10, 5)

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

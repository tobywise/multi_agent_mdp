import pytest
from maMDP.mdp import *

@pytest.fixture
def hexMDP_fixture():
    """ Creates a hex grid MDP with 20 states (5x4), 3 features """
    return HexGridMDP(np.ones((3, 20)), shape=(5, 4), 
                      feature_names=['feature_A', 'feature_B', 'feature_C'], self_transitions=False)

@pytest.fixture
def squareMDP_fixture():
    """ Creates a hex grid MDP with 20 states (5x4), 3 features """
    return SquareGridMDP(np.ones((3, 20)), shape=(5, 4), 
                         feature_names=['feature_A', 'feature_B', 'feature_C'], self_transitions=False)

@pytest.fixture
def hexMDP_fixture_self_transitions():
    """ Creates a hex grid MDP with 20 states (5x4), 3 features, with self-transitions """
    return HexGridMDP(np.ones((3, 20)), shape=(5, 4), 
                      feature_names=['feature_A', 'feature_B', 'feature_C'], self_transitions=True)

@pytest.fixture
def squareMDP_fixture_self_transitions():
    """ Creates a square grid MDP with 20 states (5x4), 3 features, with self-transitions """
    return SquareGridMDP(np.ones((3, 20)), shape=(5, 4), 
                         feature_names=['feature_A', 'feature_B', 'feature_C'], self_transitions=True)

@pytest.fixture
def small_hexMDP_fixture():
    """ Creates a small (2x2) hex MDP"""
    return HexGridMDP(np.ones((3, 4)), shape=(2, 2), 
                    feature_names=['feature_A', 'feature_B', 'feature_C'], self_transitions=False)

@pytest.fixture
def small_squareMDP_fixture():
    """ Creates a small (2x2) square MDP"""
    return SquareGridMDP(np.ones((3, 4)), shape=(2, 2), 
                    feature_names=['feature_A', 'feature_B', 'feature_C'], self_transitions=False)


def test_hexgrid_sas_n_states(hexMDP_fixture):
    assert hexMDP_fixture.sas.shape[0] == hexMDP_fixture.sas.shape[2] == 20

def test_squaregrid_sas_n_states(squareMDP_fixture):
    assert squareMDP_fixture.sas.shape[0] == squareMDP_fixture.sas.shape[2] == 20


def test_hexgrid_sas_n_actions(hexMDP_fixture):
    assert hexMDP_fixture.sas.shape[1] == 6

def test_squaregrid_sas_n_actions(squareMDP_fixture):
    assert squareMDP_fixture.sas.shape[1] == 4


def test_hexgrid_sas_self_transitions_n_actions(hexMDP_fixture_self_transitions):
    assert hexMDP_fixture_self_transitions.sas.shape[1] == 7

def test_squaregrid_sas_self_transitions_n_actions(squareMDP_fixture_self_transitions):
    assert squareMDP_fixture_self_transitions.sas.shape[1] == 5


def test_hexgrid_deterministic(hexMDP_fixture):
    assert np.all((hexMDP_fixture.sas == 0) | (hexMDP_fixture.sas == 1))

def test_squaregrid_deterministic(squareMDP_fixture):
    assert np.all((squareMDP_fixture.sas == 0) | (squareMDP_fixture.sas == 1))


def test_hexgrid_sas(small_hexMDP_fixture):
    correct_sas = np.zeros((4, 6, 4))
    correct_sas[(0, 0, 1, 1, 2, 2, 3, 3), 
                (0, 2, 2, 3, 0, 5, 3, 5),
                (1, 2, 3, 0, 3, 0, 2, 1)
                ] = 1
    assert np.all(small_hexMDP_fixture.sas == correct_sas)

def test_squaregrid_sas(small_squareMDP_fixture):
    correct_sas = np.zeros((4, 4, 4))
    correct_sas[(0, 0, 1, 1, 2, 2, 3, 3), 
                (0, 1, 1, 2, 0, 3, 2, 3),
                (1, 2, 3, 0, 3, 0, 2, 1)
                ] = 1
    assert np.all(small_squareMDP_fixture.sas == correct_sas)
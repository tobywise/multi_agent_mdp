import numpy as np
from typing import Union, Tuple
from numba import njit

@njit
def get_sa_sr(sas: np.ndarray, gamma=0.95) -> np.ndarray:
    """
    Derives the successor representation for state-action pairs.

    Args:
        sas (np.ndarray): State - action - state transitions.
        gamma (float, optional): Discount factor. Defaults to 0.95.

    Returns:
        np.ndarray: Successor representation of shape [n states * n actions, n states * n actions]
    """

    n_states = sas.shape[0]
    n_actions = sas.shape[1]

    sathing = np.zeros((n_states * n_actions, n_states * n_actions))

    for s in range(n_states):
        for a in range(n_actions):
            next_state = np.argwhere(sas[s, a, :])
            if len(next_state):
                next_state = next_state[0][0]
                for a_ in range(n_actions):
                    sathing[(s * n_actions) + a, (next_state * n_actions) + a_] = (
                        1 / n_actions
                    )

    sr = np.linalg.inv(np.eye(sathing.shape[0]) - gamma * sathing)

    return sr

@njit
def get_sr_q_values(
    theta: np.ndarray,
    features: np.ndarray,
    sr: np.ndarray,
    sas: np.ndarray,
    n_states: int,
    n_actions: int,
) -> np.ndarray:
    """
    Gets Q values for a single state, given the successor representation and reward
    weights.

    Args:
        theta (np.ndarray): Reward weights, one entry per feature
        features (np.ndarray): Features
        sr (np.ndarray): Successor representation. Each entry must represent a state-action pair.
        sas (np.ndarray): State - action - state transitions.
        pair rather than a state.
        n_states (int): Number of states
        n_actions (int): Number of actions

    Returns:
        np.ndarray: Returns Q values for every action from the given state.
    """

    assert (
        features.shape[-1] == n_states
    ), "Mismatch between features shape and number of states"

    reward = np.dot(theta, features)
    qs = np.dot(
        sr, reward.repeat(n_actions)
    )  # Repeat so we have the same reward for each action in a state
    qs = qs.reshape((n_states, n_actions))

    # Set invalid transitions to negative infinity
    valid_transitions = sas.sum(axis=-1) == 0
    
    for s in range(n_states):
        for a in range(n_actions):
            if valid_transitions[s, a]:
                qs[s, a] = -np.inf

    return qs

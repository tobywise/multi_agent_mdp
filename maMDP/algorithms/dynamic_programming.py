from collections import namedtuple
import numpy as np 
from fastprogress import progress_bar
import warnings
from scipy.optimize import minimize
from numba import jit, njit, prange
from typing import Union, Tuple

from .base import Algorithm
from ..mdp import MDP

@njit(cache=True)
def get_state_action_values(s:int, n_actions:int, sas:np.ndarray, reward:np.ndarray, 
                            discount:float, values:np.ndarray) -> np.ndarray:
    """
    Calculates the value of each action for a given state. Used within the main value 
    iteration loop.

    Reward is typically conceived of as resulting from taking action A in state S. Here, we for the sake
    of simplicity, we assume that the reward results from visiting state S' - that is, taking action A in
    state S isn't rewarding in itself, but the reward received is dependent on the reward present in state
    S'.

    Args:
        s (int): State ID
        n_actions (int): Number of possible actions
        sas (np.ndarray): State, action, state transition function
        reward (np.ndarray): Reward available at each state
        discount (float): Discount factor
        values (np.ndarray): Current estimate of value function


    Returns:
        np.ndarray: Estimated value of each state 
    """

    action_values = np.zeros(n_actions)

    # Loop over actions for this state
    for a in range(n_actions):
        # Probability of transitions given action - allows non-deterministic MDPs
        p_sprime = sas[s, a, :]  
        # Value of each state given actions
        action_values[a] = np.dot(p_sprime, reward + discount*values)

    return action_values

@njit(cache=True)
def delta_update(delta:float, old_v:float, new_v:float) -> float:
    """
    Calculates delta (difference between state value estimate for current
    and previous iteration).

    Args:
        delta (float): Previous delta
        old_v (float): Previous value estimate
        new_v (float): Current value estimate

    Returns:
        float: New delta
    """

    delta_arr = np.zeros(2)
    delta_arr[0] = delta
    delta_arr[1] = np.abs(old_v - new_v) 
    delta = np.max(delta_arr)

    return delta
    

@njit(cache=True)
def state_value_iterator(values:np.ndarray, delta:float, reward:np.ndarray, 
                         discount:float, sas:np.ndarray, soft=False) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Core value iteration function - calculates value function for the MDP and returns q-values for each action
    in each state.

    This function just runs one iteration of the value iteration algorithm.

    "Soft" value iteration can optionally be performed. This essentially involves taking the softmax of action values
    rather than the max, and is useful for inverse reinforcement learning (see Bloem & Bambos, 2014).

    Args:
        values (np.ndarray): Current estimate of the value function
        delta (float): Delta (difference between old and new estimate of value function) from previous iteration
        reward (np.ndarray): Reward at each state (i.e. features x reward function)
        discount (float): Discount factor
        sas (np.ndarray): State, action, state transition function
        soft (bool, optional): If True, this implements "soft" value iteration rather than standard
        value iteration. Defaults to False.

    Returns:
        Tuple[np.ndarray, float, np.ndarray]: Returns new estimate of the value function, new delta, and new q_values
    """

    # Get number of states and actions
    n_states = sas.shape[0]
    n_actions = sas.shape[1]

    # Empty array for new q values
    q_values = np.zeros((n_states, n_actions))

    # Loop over every state
    for s in range(n_states):

        if np.any(np.isinf(values)):
            print(values)
            raise ValueError

        # Current estimate of V(s)
        v = values[s]

        # Values of each action taken from state s
        action_values = get_state_action_values(s, n_actions, sas, reward, discount, values)

        # Update action values
        if not soft:
            # Max
            values[s] = action_values.max()
        else:
            # Softmax (e.g. Bloem & Bambos, 2014)
            valid_actions = np.argwhere(sas[s, ...])[:, 0]  # Ignore actions that the agent can't take
            # action_values
            values[s] = np.log(np.sum(np.exp(action_values)) + 1e-200)  # Add a small amount here to avoid doing log(0) and getting -inf

        # Update Q value for each action in this state
        q_values[s, :] = action_values

        # Update delta
        delta = delta_update(delta, v, values[s])

    return values, delta, q_values

@njit(cache=True)
def solve_value_iteration(reward_function:np.ndarray, features:np.ndarray, max_iter:int, 
                          discount:float, sas:np.ndarray, tol:float, soft=False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves a given MDP using Value Iteration, given a certain reward function.

    Args:
        reward_function (np.ndarray): Reward function, 1D array with one entry per feature
        features (np.ndarray): 2D array of features of shape (n_features, n_states)
        max_iter (int): Maximum number of iterations to run
        discount (float): Discount factor
        sas (np.ndarray): State, action, state transition function
        tol (float): Tolerance for convergence
        soft (bool, optional): If True, this implements "soft" value iteration rather than standard
        value iteration. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Returns the value function and Q values for each action in each state
    """

    assert discount < 1 and discount > 0, 'Discount must be between 0 and 1'

    # Get number of states and actions
    n_states = sas.shape[0]
    n_actions = sas.shape[1]

    # Initialise state values at zero
    values_ = np.zeros(n_states)

    # Get state rewards based on the supplied reward function
    reward_ = np.dot(reward_function.astype(np.float64), features)

    # Until converged   
    for _ in range(max_iter):

        # Delta checks for convergence
        delta_ = 0

        values_, delta_, q_values_ = state_value_iterator(values_, delta_, reward_, discount, sas, soft)

        if delta_ < tol:
            break

    # Set invalid transitions to negative infinity
    valid_transitions = sas.sum(axis=-1) == 0
    
    for s in range(n_states):
        for a in range(n_actions):
            if valid_transitions[s, a]:
                q_values_[s, a] = -np.inf

    return values_, q_values_



# TODO add soft option
class ValueIteration(Algorithm):

    def __init__(self, discount:float=0.9, tol:float=1e-8, max_iter:int=500):
        """
        Value iteration algorithm.

        Args:
            discount (float, optional): Discount for VI algorithm. Defaults to 0.9.
            tol (float, optional): Tolerance for convergence. Defaults to 1e-8.
            max_iter (int, optional): Maximum number of iterations to run. Defaults to 500.
        """

        self.discount = discount
        self.tol = tol
        self.max_iter = max_iter
        self.v_history = []
        self._state_value_func = None
        self.online = False
        self.name = 'value_iteration'
        
        super().__init__()

    def _solve_value_iteration(self, reward_function, features, max_iter, discount, sas, tol):
        """ Method used to solve value iteration - this could be implemented in _fit() but this allows _fit()
        to be overrideen without losing the functionality of the solve method """

        # Solve using value iteration
        values, q_values = solve_value_iteration(reward_function, features, max_iter, discount, sas, tol)

        return values, q_values


    def _fit(self, mdp:MDP, reward_function:np.ndarray, position, n_moves) -> Union[np.ndarray, np.ndarray]:
        """
        Uses value iteration to solve the MDP

        Args:
            mdp (MDP): The MDP to solve
            reward_function (np.ndarray): Reward function of the agent

        Returns:
            Union[np.ndarray, np.ndarray]: Returns value function and q values as numpy arrays
        """

        # Solve using value iteration
        values, q_values = self._solve_value_iteration(np.array(reward_function), mdp.features, self.max_iter, self.discount, mdp.sas, self.tol)

        return values, q_values






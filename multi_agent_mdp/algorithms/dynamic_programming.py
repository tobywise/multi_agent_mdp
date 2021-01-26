from collections import namedtuple
import numpy as np 
from fastprogress import progress_bar
import warnings
from scipy.optimize import minimize
from numba import jit, njit, prange
from typing import Union, Tuple
from .base import Algorithm
from multi_agent_mdp.mdp import MDP

@njit
def get_state_action_values(s:int, n_actions:int, sas:np.ndarray, reward:np.ndarray, 
                            discount:float, values:np.ndarray) -> np.ndarray:
    """
    Calculates the value of each action for a given state. Used within the main value 
    iteration loop.

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

@njit
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
    

@njit
def state_value_iterator(values:np.ndarray, delta:float, reward:np.ndarray, 
                         discount:float, sas:np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Core value iteration function - calculates value function for the MDP and returns q-values for each action
    in each state.

    This function just runs one iteration of the value iteration algorithm.

    Args:
        values (np.ndarray): Current estimate of the value function
        delta (float): Delta (difference between old and new estimate of value function) from previous iteration
        reward (np.ndarray): Reward at each state (i.e. features x reward function)
        discount (float): Discount factor
        sas (np.ndarray): State, action, state transition function

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

        # Current estimate of V(s)
        v = values[s]

        # Values of each action taken from state s
        action_values = get_state_action_values(s, n_actions, sas, reward, discount, values)

        # Update action values
        values[s] = action_values.max()

        # Update Q value for each action in this state
        q_values[s, :] = action_values

        # Update delta
        delta = delta_update(delta, v, values[s])

    return values, delta, q_values


@njit
def solve_value_iteration(reward_function:np.ndarray, features:np.ndarray, max_iter:int, 
                          discount:float, sas:np.ndarray, tol:float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves a given MDP using Value Iteration, given a certain reward function.

    Args:
        reward_function (np.ndarray): Reward function, 1D array with one entry per feature
        features (np.ndarray): 2D array of features of shape (n_features, n_states)
        max_iter (int): Maximum number of iterations to run
        discount (float): Discount factor
        sas (np.ndarray): State, action, state transition function
        tol (float): Tolerance for convergence

    Returns:
        Tuple[np.ndarray, np.ndarray]: Returns the value function and Q values for each action in each state
    """

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

        values_, delta_, q_values_ = state_value_iterator(values_, delta_, reward_, discount, sas)

        if delta_ < tol:
            break
    
    return values_, q_values_


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
        
        super().__init__()

    def _solve_value_iteration(self, reward_function, features, max_iter, discount, sas, tol):
        """ Method used to solve value iteration - this could be implemented in _fit() but this allows _fit()
        to be overrideen without losing the functionality of the solve method """

        # Solve using value iteration
        values, q_values = solve_value_iteration(reward_function, features, max_iter, discount, sas, tol)

        return values, q_values


    def _fit(self, mdp:MDP, reward_function:np.ndarray) -> Union[np.ndarray, np.ndarray]:
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





# class QValues():
#     """ Class used to store Q values - subclasses numpy arrays, adding some useful methods"""


#     # def set_state_values()

class QValues(np.ndarray):
    """ Class used to store Q values - subclasses numpy arrays, adding some useful methods"""

    def __new__(cls, shape:Tuple[int], online:bool=False, current_state:int=None):
        obj = np.ones(shape) * np.nan
        obj = obj.view(cls)
        obj.online = online
        obj.current_state = current_state

        return obj

    def __array_finalize__(self, obj):
        # Add attributes if necessary
        if obj is None: return
        self.online = getattr(obj, 'online', None)
        self.current_state = getattr(obj, 'current_state', None)


    def set_state_values(self, state:int, values:np.ndarray):

        # Check state
        if not isinstance(state, int):
            raise TypeError("State must be an integer, given {0}".format(type(state)))

        if state < 0:
            raise ValueError("State must be positive")

        if state > self.shape[0]:
            raise ValueError("State {0} is not in the MDP".format(state))

        # Check action values
        if not isinstance(values, np.ndarray):
            raise TypeError("Values must be supplied as a numpy array, got {0}".format(type(values)))

        if not values.ndim == 1:
            raise AttributeError("Values must be supplied as 1D array")

        if not len(values) == self.shape[1]:
            raise AttributeError("Values must be the same shape as the action space of the MDP. " 
                                "Got {0} values, MDP has {1} actions available".format(len(values), self.shape[1]))

        # If online, set everything to NaN except the current state
        if self.online:
            self[:, :] = np.nan
            self[state, :] = values

        else:
            self[state, :] = values


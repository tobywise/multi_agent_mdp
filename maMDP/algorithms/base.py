from abc import abstractmethod, abstractproperty, ABCMeta
from ..environments import Environment
import numpy as np
from ..mdp import MDP
import warnings
from typing import Tuple

class Algorithm(metaclass=ABCMeta):

    def __init__(self):

        self.q_values = None
        self.fit_complete = False
        self.values = None
        self._agent = None
        self._environment = None

        super().__init__()

    def _attach(self, agent:'Agent', environment:Environment):
        """ Adds information about the agent and environment """
        self._agent = agent
        self._environment = environment

    def fit(self, mdp:MDP, reward_function:np.ndarray, position:int, n_moves:Tuple[Tuple[int]]=None,  *args, **kwargs):
        """
        Runs the algorithm to determine the value of different actions.

        Args:
            mdp (MDP): The MDP containing states and actions.
            reward_function (np.ndarray): The reward function used to determine state values.
            position (int): Current position of the agent (i.e. the state it is currently in). Used for online algorithms.
            n_moves (Tuple[Tuple[int]]): Used by simulation algorithms to determine how many steps to simulate for each agent.
            This represents the number of moves each agent can take per turn. This is a tuple of tuples, where the first tuple
            corresponds to the primary agent, and the remaining tuples correspond to the other agents. Within each tuple, the first
            element is the number of moves the agent can take in the first turn, the second element is the number of moves for the
            second turn, etc. If None, the n_steps argument is used to determine how many steps to simulate, and the number of moves
            per turn is determined based on information present in the Agent classes.
        """
        # Check inputs
        if not isinstance(reward_function, np.ndarray) or isinstance(reward_function, list):
            raise TypeError('Reward function must be a 1D array or list, got type {0}'.format(reward_function))
        reward_function = np.array(reward_function)
        if not reward_function.ndim == 1:
            raise AttributeError("Reward function array must be 1-dimensional")

        if not len(reward_function) == mdp.n_features:
            raise AttributeError("Reward function should have as many entries as the MDP has features")

        # Use the fit method
        values, q_values = self._fit(mdp, reward_function, position, n_moves, *args, **kwargs)

        # Check outputs before accepting them
        assert isinstance(values, np.ndarray), 'Values should be in the form of a numpy array'
        assert values.ndim == 1, 'Value array should be 1-dimensional'
        assert len(values) == mdp.n_states, 'Value array should have as many entries as the MDP has states'

        # Q values are a 2D numpy array regardless of online/offline
        assert isinstance(q_values, np.ndarray), 'Q values should be in the form of a numpy array'
        assert q_values.ndim == 2, 'Q value array should be 2-dimensional'

        # If this is an online algorithm, we determine Q values for actions only from the current state
        assert q_values.shape[0] == mdp.n_states, 'First dimension of Q value array should have as many entries as the MDP has states'
        assert q_values.shape[1] == mdp.n_actions, 'Second dimension of Q value array should have as many entries as the MDP has actions'

        # Update attributes
        self.fit_complete = True
        self.values = values
        self.q_values = q_values

    @abstractmethod
    def _fit(self, mdp:MDP, reward_function:np.ndarray, position, n_moves, *args, **kwargs):
        """ Fit the algorithm 
        Must return numpy arrays representing:
        1) State values (value function): 1D array (n_states, )
        2) Action values (Q-values): 2D array (n_states, n_actions)
        """
        pass

    def reset(self):
        """
        Reset to starting state
        """
        self.q_values = None
        self.fit_complete = False
        self.values = None

        self._reset()

    def _reset(self):
        """
        Additional things to reset when subclassing
        """
        pass


class Null(Algorithm):
    """
    Implements a null planning model which estimates all state values and Q values to be 1.
    """

    def _fit(self, mdp:MDP, reward_function:np.ndarray, position, n_moves):

        state_vawlues = np.ones(mdp.n_states)
        q_values = np.ones((mdp.n_states, mdp.n_actions))

        return state_vawlues, q_values
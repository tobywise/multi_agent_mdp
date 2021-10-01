from abc import ABC, abstractmethod, abstractproperty, ABCMeta
from maMDP.algorithms.action_selection import ActionSelector
from ..environments import Environment
import numpy as np
from ..mdp import MDP, Trajectory
import warnings

class Algorithm(metaclass=ABCMeta):
    """
    Base class for policy derivation algorithms.
    """

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

    def fit(self, mdp:MDP=None, trajectory:Trajectory=None, reward_weights:np.ndarray=None, 
            position:int=None, n_steps:int=None,  *args, **kwargs):
        """
        Runs the algorithm to determine the value of different actions.

        Args:
            mdp (MDP): The MDP containing states and actions.
            trajectory (Trajectory): Observations of actions within the MDP.
            reward_weights (np.ndarray): The reward function used to determine state values.
            n_steps (int, optional): Number of steps to plan ahead, if used by the algorithm. Can be int or None.
            position (int): Current position of the agent (i.e. the state it is currently in). Used for online algorithms.
        """

        # Check inputs
        if not isinstance(reward_weights, np.ndarray) or isinstance(reward_weights, list):
            raise TypeError('Reward function must be a 1D array or list, got type {0}'.format(reward_weights))
        reward_weights = np.array(reward_weights)
        if not reward_weights.ndim == 1:
            raise AttributeError("Reward function array must be 1-dimensional")

        if not len(reward_weights) == mdp.n_features:
            raise AttributeError("Reward function should have as many entries as the MDP has features")

        # Use the fit method
        values, q_values = self._fit(mdp, reward_weights, position, n_steps, *args, **kwargs)

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
    def _fit(self, mdp:MDP, reward_weights:np.ndarray, position, n_steps, *args, **kwargs):
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

class MFAlgorithm(metaclass=ABCMeta):

    def __init__(self):

        self.q_values = None
        self.fit_complete = False
        self.values = None
        self._agent = None
        self._environment = None

        if not hasattr(self, 'action_selector'):
            self.action_selector = None  # Used for on-policy algorithms

        super().__init__()

    def _attach(self, agent:'Agent', environment:Environment):
        """ Adds information about the agent and environment """
        self._agent = agent
        self._environment = environment
        self.q_values = np.zeros(environment.mdp.sas.shape[:-1])  # Give the agent a starting estimate for Q values
        self.fit_complete = True   # The agent should be able to move before fitting, so pretend the algorithm has been fit

        # For on-policy algorithms, use the same action selection as the agent, unless one is already specified
        if self.action_selector is None:
            self.action_selector = agent.action_selector  

        # Anything else
        self._attach_additional()

    def _attach_additional(self):
        """ Can be overriden when subclassing to do additional things when attaching """
        pass

    def fit(self, mdp:MDP=None, trajectory:Trajectory=None, reward_weights:np.ndarray=None, 
            position:int=None, n_steps:int=None,  *args, **kwargs):
        """
        Runs the algorithm to determine the value of different actions.

        Note: Many of the arguments are there only for compatibility across MF and MB methods and are not necessarily
        required to implement a model-free algorithm.

        Args:
            mdp (MDP): The MDP containing states and actions.
            trajectory (Trajectory): Observations of actions within the MDP.
            reward_weights (np.ndarray): The reward function used to determine state values.
            n_steps (int, optional): Number of steps to plan ahead, if used by the algorithm. Can be int or None.
            position (int): Current position of the agent (i.e. the state it is currently in). Used for online algorithms.
        """

        # Only fit if we have trajectory information
        if trajectory.length > 0:

            # Use the fit method
            values, q_values = self._fit(trajectory, *args, **kwargs)

            # Check outputs before accepting them
            assert isinstance(values, np.ndarray), 'Values should be in the form of a numpy array'
            assert values.ndim == 1, 'Value array should be 1-dimensional'
            assert len(values) == trajectory.mdp.n_states, 'Value array should have as many entries as the MDP has states'

            # Q values are a 2D numpy array regardless of online/offline
            assert isinstance(q_values, np.ndarray), 'Q values should be in the form of a numpy array'
            assert q_values.ndim == 2, 'Q value array should be 2-dimensional'

            # If this is an online algorithm, we determine Q values for actions only from the current state
            assert q_values.shape[0] == trajectory.mdp.n_states, 'First dimension of Q value array should have as many entries as the MDP has states'
            assert q_values.shape[1] == trajectory.mdp.n_actions, 'Second dimension of Q value array should have as many entries as the MDP has actions'

            # Update attributes
            self.fit_complete = True
            self.values = values
            self.q_values = q_values

        # Otherwise warn about lack of observations, but don't throw an error
        else:
            warnings.warn('No observations provided, not fitting')

    @abstractmethod
    def _fit(self, trajectory:Trajectory, *args, **kwargs):
        """Fit the algorithm

        Must return a numpy array representing the state X action X state transition matrix
        """
        pass




class TransitionAlgorithm(metaclass=ABCMeta):
    """
    Base class for transition learning algorithms.
    """

    def __init__(self):
        
        self.sas = None
        self._agent = None
        self._environment = None

    def _attach(self, agent:'Agent', environment:Environment):
        """ Adds information about the agent and environment """
        self._agent = agent
        self._environment = environment

    def reset(self):

        self.sas = None

        self._reset()

    def _reset(self):
        pass


class MFTransitionAlgorithm(TransitionAlgorithm):

    def fit(self, trajectory:Trajectory, *args, **kwargs):
        """
        Fits the algorithm to learn a transition matrix, based on the trajectory information provided.

        Args:
            trajectory (Trajectory): Trajectory within the MDP
        """

        sas = self._fit(trajectory, *args, **kwargs)

        assert sas.shape == trajectory.mdp.sas.shape, 'Shape of estimated transition matrix = {0}, expected shape {1}'.format(sas.shape, trajectory.mdp.sas.shape)

        self.sas = sas

    @abstractmethod
    def _fit(self, trajectory:Trajectory, *args, **kwargs):
        """Fit the algorithm

        Must return a numpy array representing the state X action X state transition matrix
        """
        pass


class MBTransitionAlgorithm(TransitionAlgorithm):

    def fit(self, mdp:MDP, position:int=None, n_steps:int=None, *args, **kwargs):
        """
        Fits the algorithm to learn a transition matrix

        Args:
            mdp (MDP): The MDP containing states and actions.
            position (int): Starting position of the agent, if used by the algorithm.
            n_steps (int, optional): Number of steps to run, if used by the algorithm. Defaults to None.
        """

        sas = self._fit(mdp, position, n_steps, *args, **kwargs)

        assert sas.shape == mdp.sas.shape, 'Shape of estimated transition matrix = {0}, expected shape {1}'.format(sas.shape, mdp.sas.shape)

        self.sas = sas

    @abstractmethod
    def _fit(self, mdp:MDP, position:int, n_steps:int, *args, **kwargs):
        """Fit the algorithm

        Must return a numpy array representing the state X action X state transition matrix
        """
        pass


class Null(Algorithm):
    """
    Implements a null planning model which estimates all state values and Q values to be 1.
    """


    def _fit(self, mdp:MDP, reward_weights:np.ndarray, position, n_steps):

        state_vawlues = np.ones(mdp.n_states)
        q_values = np.ones((mdp.n_states, mdp.n_actions))

        return state_values, q_values
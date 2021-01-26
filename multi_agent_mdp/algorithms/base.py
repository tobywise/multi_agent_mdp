from abc import abstractmethod, abstractproperty, ABCMeta
import numpy as np
from multi_agent_mdp.mdp import MDP
import warnings

class Algorithm(metaclass=ABCMeta):

    def __init__(self):

        self.q_values = None
        self.fit_complete = False
        self.values = None

        super().__init__()

    def fit(self, mdp:MDP, reward_function:np.ndarray, *args, **kwargs):

        # Check inputs
        if not isinstance(reward_function, np.ndarray) or isinstance(reward_function, list):
            raise TypeError('Reward function must be a 1D array or list, got type {0}'.format(reward_function))
        reward_function = np.array(reward_function)
        if not reward_function.ndim == 1:
            raise AttributeError("Reward function array must be 1-dimensional")

        if not len(reward_function) == mdp.n_features:
            raise AttributeError("Reward function should have as many entries as the MDP has features")

        # Use the fit method
        values, q_values = self._fit(mdp, reward_function, *args, **kwargs)

        # Check outputs before accepting them
        # We don't calculate state values for online algorithms, they should be set to None
        if self.online:
            if not values is None:
                warnings.warn('Online algorithm provided state values that are not None, setting output to None')
                values = None
        else:
            assert isinstance(values, np.ndarray), 'Values should be in the form of a numpy array'
            assert values.ndim == 1, 'Value array should be 1-dimensional'
            assert len(values) == mdp.n_states, 'Value array should have as many entries as the MDP has states'

        # Q values are a 2D numpy array regardless of online/offline
        assert isinstance(q_values, np.ndarray), 'Q values should be in the form of a numpy array'
        assert q_values.ndim == 2, 'Q value array should be 2-dimensional'

        # If this is an online algorithm, we determine Q values for actions only from the current state
        # This means the first dimension of the Q value array should be 1
        if self.online:
            assert q_values.shape[0] == 1, 'First dimension of Q value array for online algorithm should be of length 1'
        else:
            assert q_values.shape[0] == mdp.n_states, 'First dimension of Q value array should have as many entries as the MDP has states'
        assert q_values.shape[1] == mdp.n_actions, 'Second dimension of Q value array should have as many entries as the MDP has actions'

        # Update attributes
        self.fit_complete = True
        self.values = values
        self.q_values = q_values

    @abstractmethod
    def _fit(self, mdp:MDP, reward_function:np.ndarray, *args, **kwargs):
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

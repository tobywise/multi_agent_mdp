from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Union, Tuple
import numpy as np

def softmax(qs, temperature=1):

    # No valid transitions
    if np.all(np.isneginf(qs)):
        return np.ones_like(qs) / len(qs)
    else:
        # Subtract max value from Qs to avoid overflow
        out = (np.exp((qs - qs.max()) / temperature)) / (np.sum(np.exp((qs - qs.max()) / temperature), axis=0))
        return out

def check_q_values(qs):
    if not isinstance(qs, np.ndarray):
        raise TypeError("Q values must be provided as a numpy array, got {0}".format(type(qs)))
    if not qs.ndim == 2:
        raise AttributeError("Q values must be 2-dimensional, of shape (n_states, n_actions)")
    if np.all(np.isnan(qs)):
        raise ValueError("All Q values are NaN")

def random_max(x:np.ndarray, rng:np.random.RandomState=None):
    """
    Returns a randomly selected index from a 1D array where that index's value equals the array maximum

    Args:
        x (np.ndarray): 1D array
        rng (np.random.RandomState): RNG instance

    Returns:
        [type]: [description]
    """

    if rng is None:
        rng = np.random

    return rng.choice(np.where(x == x.max())[0])

class ActionSelector(metaclass=ABCMeta):
    """
    Base class for action selectors. These take a Q values for each state-action pair
    and calculate action probabilities and selected actions.
    """

    def get_pi(self, q_values:np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Gets a deterministic policy (i.e. one action per state) based on Q values according to the action selection rule.

        Note:
        Wraps the _pi() method, which must be implemented when subclassing the ActionSelector class.

        Args:
            q_values (np.ndarray): 2D array of q values (n_states, n_actions)

        Returns:
            np.ndarray: 1D array of actions for each state
        """

        # Ensure q values are provided in the right format
        check_q_values(q_values)

        nan_states = np.isnan(np.nanmean(q_values, axis=1))

        q_values = q_values.copy()
        q_values[np.isnan(q_values)] = -np.inf

        # Get policy
        pi = self._get_pi(q_values=q_values, *args, **kwargs)

        # Set actions to -1 for states where no action had non-NaN value
        pi[nan_states] = -1
        
        # Check that the action selector returned the right format
        assert isinstance(pi, np.ndarray), "Action selector should return pi as a numpy array, got {0}".format(type(pi))
        assert pi.ndim == 1, "Returned pi must be 1-dimensional, of shape (n_states, )"
        assert len(pi) == q_values.shape[0], 'Pi must have the same number of entries as the number of states implied by ' \
                                             'the input Q values. Pi has {0} state entries, Q values have {1}'.format(len(pi), q_values.shape[0])

        return pi


    def get_pi_p(self, q_values:np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Gets a probabilistic policy (i.e. probability of taking each action in each state) based on Q values 
        according to the action selection rule.

        Args:
            q_values (np.ndarray): 2D array of q values (n_states, n_actions)

        Returns:
            np.ndarray: 2D array of action probabilities for each state, shape (n_states, n_actions)
        """


        # Ensure q values are provided in the right format
        check_q_values(q_values)
        
        nan_states = np.isnan(np.nanmean(q_values, axis=1))

        q_values = q_values.copy()
        q_values[np.isnan(q_values)] = -np.inf

        # Get policy
        pi_p = self._get_pi_p(q_values=q_values, *args, **kwargs)

        # Set actions to NaN for states where no action had non-NaN value
        pi_p[nan_states, :] = np.nan

        # Check that the action selector returned the right format
        assert isinstance(pi_p, np.ndarray), "Action selector should return pi_p as a numpy array, got {0}".format(type(pi_p))
        assert pi_p.ndim == 2, "Returned pi_p must be 2-dimensional, of shape (n_states, n_actions)"
        assert pi_p.shape == q_values.shape, 'Pi_p must be the same shape as the Q values used as input, pi_p shape = {0}, ' \
                                             'Q values shape = {1}'.format(pi_p.shape, q_values.shape)

        return pi_p

    @abstractmethod
    def _get_pi(self, q_values:np.ndarray, seed:int=None, *args, **kwargs):
        """ Calculate action probabilities based on Q values 
        Must return a 1D array
        """
        pass

    @abstractmethod
    def _get_pi_p(self, q_values:np.ndarray, *args, **kwargs):
        """ Get a single action for each state-action pair (based on action probabilities) 
        Must return a 2D array (n_states, n_actions)
        """
        pass

    def get_nstates_actions(self, q_values:np.ndarray) -> Tuple[int, int]:
        """
        Checks format of supplied q values and returns the number of implied states and actions

        Args:
            q_values (np.ndarray): 2D array of q values for the mdp

        Returns:
            Union[int, int]: Number of states, number of actions
        """

        if not isinstance(q_values, np.ndarray):
            raise TypeError("Q values must be supplied as a numpy array, got type {0}".format(type(q_values)))
        if not q_values.ndim == 2:
            raise AttributeError("Q values must be 2D, supplied array has {0} dimensions".format(q_values.ndim))

        n_states = q_values.shape[0]
        n_actions = q_values.shape[1]

        return n_states, n_actions


class MaxActionSelector(ActionSelector):
    """
    Selects the action with the highest Q value. If no single action has the maximum Q value, one of those that does
    is chosen at random.

    """

    def __init__(self, seed:int=None) -> None:
        """
        Args:
            seed (int, optional): RNG seed. Defaults to None.
        """
        self.name = 'max'

        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
        else:
            self.rng = np.random

        super().__init__()
    
    def _get_pi_p(self, q_values:np.ndarray) -> np.ndarray:
        """
        Gets action probabilities - action with max q value = 1, otherwise 0

        Args:
            q_values (np.ndarray): Estimated q-values, shape (n_states, n_actions)

        Returns:
            np.ndarray: Action probabilities
        """

        n_states, n_actions = self.get_nstates_actions(q_values)

        self._pi_p = np.zeros((n_states, n_actions))  # Probability of choosing action A in state S given policy

        for s in range(n_states):
            max_actions = np.where(q_values[s, :] == q_values[s, :].max())[0]
            self._pi_p[s, max_actions] = 1 / len(max_actions)  # Deterministic, only one action chosen

        return self._pi_p

    def _get_pi(self, q_values:np.ndarray) -> np.ndarray:
        """
        Determines the action to be taken in each state

        Args:
            q_values (np.ndarray): Estimated q-values, shape (n_states, n_actions)

        Returns:
            np.ndarray: Action for each state, shape (n_states)
        """
        # print(q_values)
        n_states, _ = self.get_nstates_actions(q_values)

        self._pi = np.zeros(n_states)

        for s in range(n_states):
            # action = np.argmax(q_values[s, :])
            action = random_max(q_values[s, :], self.rng)
            self._pi[s] = action

        self._pi = self._pi.astype(int)

        return self._pi


class SoftmaxActionSelector(ActionSelector):

    def __init__(self, temperature:float=1, seed:int=None) -> None:
        """
        Implements action selection using the softmax rule

        Args:
            temperature (float, optional): Softmax temperature. Defaults to 1.
            seed (int, optional): RNG seed. Defaults to None.
        """
        self.name = 'softmax'

        if temperature <= 0:
            raise ValueError('Temperature parameter must be greater than zero')
        self.temperature = temperature

        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
        else:
            self.rng = np.random

    def _get_pi_p(self, q_values:np.ndarray) -> np.ndarray:
        """
        Gets action probabilities using the softmax rule

        Args:
            q_values (np.ndarray): Estimated q-values, shape (n_states, n_actions)

        Returns:
            np.ndarray: Action probabilities
        """

        n_states, n_actions = self.get_nstates_actions(q_values)

        self._pi_p = np.zeros((n_states, n_actions))  # Probability of choosing action A in state S given policy

        for s in range(n_states):
            self._pi_p[s, :] = softmax(q_values[s, :], temperature=self.temperature)
        return self._pi_p

    def _get_pi(self, q_values:np.ndarray) -> np.ndarray:
        """
        Determines the action to be taken in each state using the softmax rule

        Args:
            q_values (np.ndarray): Estimated q-values, shape (n_states, n_actions)

        Returns:
            np.ndarray: Action for each state, shape (n_states)
        """
        
        self._get_pi_p(q_values)

        n_states, n_actions = self.get_nstates_actions(q_values)

        self._pi = np.zeros(n_states)

        for s in range(n_states):
            self._pi[s] = self.rng.choice(range(n_actions), p=self._pi_p[s, :] / np.sum(self._pi_p[s, :]))

        self._pi = self._pi.astype(int)

        return self._pi
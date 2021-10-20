from os import name
from ..mdp import Observation, Trajectory
import numpy as np
from .base import MFAlgorithm
from .action_selection import MaxActionSelector, ActionSelector
from typing import Dict
from abc import ABC, abstractclassmethod, abstractmethod, abstractstaticmethod
from dataclasses import dataclass
import warnings


class EligibilityTrace(ABC):
    """
    Base class for eligibility trace algorithms
    """

    def __init__(self, n_states:int, n_actions:int):

        self.eligibility_trace = np.zeros((n_states, n_actions))
        self.n_states = n_states
        self.n_actions = n_actions

        super().__init__()

    @abstractmethod
    def update(self, obs:Observation) -> np.ndarray:
        """ Eligibility trace update method. Must return the updated eligibility trace """
        pass

class AccumulatingTrace(EligibilityTrace):

    def __init__(self, n_states:int, n_actions:int, gamma:float=0.9, lambda_:float=0.0):

        self.lambda_ = lambda_
        self.gamma = gamma

        super().__init__(n_states, n_actions)

    def update(self, obs:Observation) -> np.ndarray:
        """ 
        Updates the eligibility trace using the "accumulating traces" method (http://incompleteideas.net/book/first/ebook/node77.html)

        Args:
            obs (Observation): Observation of states, actions and rewards

        Returns:
            np.ndarray: Updated eligibility trace
        """

        # Decay everything
        self.eligibility_trace = self.gamma * self.lambda_ * self.eligibility_trace
        # Add 1 to the state & action from this observation
        self.eligibility_trace[obs.state_1, obs.action] += 1
        
        return self.eligibility_trace

class ReplacingTrace(AccumulatingTrace):


    def update(self, obs:Observation) -> np.ndarray:
        """ 
        Updates the eligibility trace using the "replacing traces" method (http://www.incompleteideas.net/book/ebook/node80.html)

        Args:
            obs (Observation): Observation of states, actions and rewards

        Returns:
            np.ndarray: Updated eligibility trace
        """

        # Decay everything
        self.eligibility_trace = self.gamma * self.lambda_ * self.eligibility_trace
        # Add 1 to the state & action from this observation
        self.eligibility_trace[obs.state_1, obs.action] += 1
        # Set all other actions to zero
        other_actions = [i for i in range(self.n_actions) if not i == obs.action]
        self.eligibility_trace[obs.state_1, other_actions] = 0
        
        return self.eligibility_trace



class QLearning(MFAlgorithm):

    def __init__(self, learning_rate:float=0.7, gamma:float=0.9, eligibility_trace_algorithm:EligibilityTrace=ReplacingTrace, 
                eligibility_trace_kwargs:Dict={}):
        """
        Implements Q learning (off-policy TD learning)

        Args:
            learning_rate (float, optional): Learning rate. Defaults to 0.3.
            gamma (float, optional): Discount factor. Defaults to 0.9.
            eligibility_trace_algorithm (EligibilityTrace, optional): Eligibility trace method to use. If None, does not use 
            eligibility traces. Defaults to ReplacingTrace.
            eligibility_trace_kwargs (Dict, optional): Leyword arguments to be supplied to the eligibility trace algorithm. Defaults to {}.
        """

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.eligibility_trace_algorithm = eligibility_trace_algorithm
        self.eligibility_trace_kwargs = eligibility_trace_kwargs

        # If we know how many states and actions we're expecting, we can set up the ET algorithm now. 
        if hasattr(self, 'q_values') and self.q_values is not None and self.eligibility_trace_algorithm is not None:
            self.initialise_et_algorithm(self.q_values.shape[0], self.q_values.shape[1])
        else:
            self.et_algorithm = None

        super().__init__()

    def _attach_additional(self):
        self.initialise_et_algorithm(self.q_values.shape[0], self.q_values.shape[1])

    def initialise_et_algorithm(self, n_states:int, n_actions:int):
        # print('Initialising eligibility trace algorithm')
        self.et_algorithm = self.eligibility_trace_algorithm(n_states, n_actions, **self.eligibility_trace_kwargs)

    def get_next_state_q(self, obs:Observation) -> float:

        return self.est_Q[obs.state_2, :].max()

    def _fit(self, trajectory:Trajectory, initial_guess:np.ndarray=None) -> np.ndarray:

        if initial_guess is None:
            self.est_Q = self.q_values
        else:
            self.est_Q = initial_guess.copy()

        if not self.est_Q.shape == trajectory.mdp.sas.shape[:-1]:
            raise AttributeError("Initial Q array shape ({0}) is not the same as expected ({1})".format(self.est_Q.shape, 
                                                                                                        trajectory.mdp.sas.shape[:-1]))

        # Initialise ET algorithm if not done already
        if self.et_algorithm is None and self.eligibility_trace_algorithm is not None:
            self.initialise_et_algorithm(self.q_values.shape[0], self.q_values[1])

        if self.et_algorithm is None:
            warnings.warn('No eligibility trace algorithm being used')

        # Remove negative inf from invalid actions to avoid confusing things
        self.est_Q[np.isneginf(self.est_Q)] = 0

        # Loop through observations
        for obs in trajectory:

            # Get Q for next state - this can change depending on algorithm
            next_state_q = self.get_next_state_q(obs)
            
            # Calculate delta
            delta = obs.reward + self.gamma * next_state_q - self.est_Q[obs.state_1, obs.action]

            # Using eligibility trace
            if self.et_algorithm is not None:
                # Update eligibility trace
                self.et_algorithm.update(obs)

                # Update
                self.est_Q = self.est_Q + (self.learning_rate * self.et_algorithm.eligibility_trace) * delta
            
            else:
                # Update
                self.est_Q[obs.state_1, obs.action] = self.est_Q[obs.state_1, obs.action] + self.learning_rate * delta

            assert ~(np.isnan(self.est_Q) & ~np.isinf(self.est_Q)).any(), 'NaNs in estimated Q values'

        # Set invalid actions to -inf
        self.est_Q[(trajectory.mdp.sas.sum(axis=2) == 0)] = -np.inf

        # Estimate value function as max Q value on each state
        values = self.est_Q.max(axis=1)

        return values, self.est_Q


    def update_eligibility_trace(self, obs:Observation):
        pass



class SARSA(QLearning):

    def __init__(self, learning_rate:float=0.7, gamma:float=0.9, 
    action_selector:ActionSelector=None, action_kwargs:Dict={}):
        """
        Implements SARSA (on-policy TD learning)

        Requires an action selection algorithm to be supplied to determine action probabilities,
        as this is an on-policy algorithm.

        Args:
            learning_rate (float, optional): Learning rate. Defaults to 0.3.
            gamma (float, optional): Discount factor. Defaults to 0.9.
            action_selector (ActionSelector, optional): Algorithm to use for action selection. Defaults to MaxActionSelector.
            action_kwargs (Dict, optional): Keyword arguments to be passed to the aciton selection algorithm. Defaults to {}.
        """

        if action_selector is not None:
            self.action_selector = action_selector(**action_kwargs)

        super().__init__(learning_rate=learning_rate, gamma=gamma)


    def get_next_state_q(self, obs:Observation):

        if self.action_selector is None:
            raise AttributeError('No action selector has been defined for the SARSA algorithm')

        # Get policy
        pi_p = self.action_selector.get_pi_p(self.est_Q)

        pi_p[np.isneginf(pi_p)] = 0
        
        if np.isnan(np.sum(self.est_Q[obs.state_2, :] * pi_p[obs.state_2])).any():
            print(self.est_Q[obs.state_2, :], pi_p[obs.state_2])

        return np.sum(self.est_Q[obs.state_2, :] * pi_p[obs.state_2])
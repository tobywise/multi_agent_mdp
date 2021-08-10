from abc import abstractmethod
import numpy as np
from typing import Union, List
from ..mdp import MDP
from .irl import BaseIRL

def degdist(x, y):
    """
    Gives (shortest) distance between two angles 
    """
    return np.abs(((x - y) + 180) % 360 - 180)

def deg_sq_exp_kernel(x, y, ls=1, variance=1):
    """
    Squared exponential kernel for two angles
    """
    
    dist = degdist(x, y)
    
    return variance * np.exp(-(np.power(dist, 2) / 2 * ls ** 2))

class BaseGeneralPolicyLearner() :

    @abstractmethod
    def fit(self, mdp:Union[MDP, List[MDP]], trajectories:list):
        """ Estimates Q values for each action"""
        pass

    def reset(self):
        pass

class TDGeneralPolicyLearner(BaseGeneralPolicyLearner):
    """ 
    TD General Policy learning
    """

    def __init__(self, learning_rate:float=0.3, decay:int=0.5, Q:np.ndarray=None, kernel:bool=False, ls:float=0.02):
        """
        Learns an agent's policy (its preferred action, assuming a constant policy across all states) based on observed trajectories.

        Args:
            learning_rate (float, optional): Learning rate for TD algorithm. Defaults to 0.3.
            decay (int, optional): If greater than 0, learning rate will decay according to learning rate * n^-decay. Defaults to 0.5.
            Q (np.ndarray, optional): Initial estimate for Q values. Defaults to None.
            kernel (bool, optional): If true, generalises action choice preference to adjacent actions using a squared
            exponential kernel. Defaults to False.
            ls (float, optional): Length scale parameter for generalisation kernel. Defaults to 0.02.
        """

        # Settings 
        assert learning_rate > 0, 'Learning rate must be greater than zero'
        assert decay >= 0, 'Decay must be positive'
        self.learning_rate = learning_rate
        self.learning_rate_decay = decay
        self._adjusted_learning_rate = learning_rate
        self.kernel = kernel
        self.ls = ls
        self.original_Q = np.array(Q).copy()
        self.n_states = 1

        self.Q = Q 

    def reset(self):
        """ Resets the estimate to its initial state """
        self.Q = self.original_Q
        self.reset_learning_rate()

    def reset_learning_rate(self):
        """ Resets the adjusted learning rate"""
        self.n_states = 1
        self._adjusted_learning_rate = self.learning_rate


    def _fit_TD_policy(self, mdp:List[MDP], trajectories:List[List[int]]):

        # Initialise Q
        if self.Q is None:
            self.Q = np.zeros(mdp[0].n_actions)

        self.n_states = 1  # Used for adjusting learning rate

        # Loop through trajectories
        for n, trajectory in enumerate(trajectories):

            # Convert state-action pairs
            action = mdp[n]._trajectory_to_state_action(trajectory)[:, 1].astype(int)

            # Loop through individual actions within each trajectory
            for n, a in enumerate(action):
                
                trial_reward = np.zeros(mdp[n].n_actions)

                # Calculate "reward"
                if not self.kernel:
                    trial_reward[int(a)] += 1  # Add 1 to chosen action
                else:
                    trial_reward = deg_sq_exp_kernel(np.arange(mdp[n].n_actions) * (360 / mdp[n].n_actions), 
                                                    a * (360 / mdp[n].n_actions), ls=self.ls) 

                # Error
                delta = trial_reward - self.Q

                # Adjust learning rate
                if self.learning_rate_decay > 0:
                    self._adjusted_learning_rate = self._adjusted_learning_rate * np.power(self.n_states, -self.learning_rate_decay)
                    self.Q += self._adjusted_learning_rate * delta

                # No learning rate adjustment
                else:
                    self.Q += self.learning_rate * delta

                self.n_states += 1


    def fit(self, mdp:Union[MDP, List[MDP]], trajectories:list) -> np.ndarray:
        """
        Uses hypothesis testing IRL to infer the reward function of an agent based on supplied
        trajectories within a given MDP.

        Args:
            mdp (MDP or List[MDP]): MDP in which the agent is acting. Can also be a list of MDPs, one for each trajectory.
            trajectories (list): List of lists of visited states - i.e. a list of trajectories, with each trajectory
            being a list of visited states.

        Returns:
            np.ndarray: Inferred reward function
        """

        if not len(trajectories[0]):
            raise TypeError("Trajectories must be a list of lists - i.e. a list of trajectories, with each trajectory \
                            representing a list of states")

        if isinstance(mdp, list):
            if not len(trajectories) == len(mdp):
                raise AttributeError("Length of trajectory list ({0}) and MDP list ({1}) do not match".format(len(trajectories), len(mdp)))
        else:
            mdp = [mdp] * len(trajectories)  # Repeat the MDP to match the number of trajectories

        self._fit_TD_policy(mdp, trajectories)

        return self.Q





# TODO change the name of this - should be Imitation rather than IRL
def TDPolicyIRL(trajectories, mdp, n_pred=2, learning_rate=0.3, kernel=True, ls=0.02):

    Q = np.zeros(mdp.n_actions)
    predictions = []


    for n, trajectory in enumerate(trajectories):



        trajectory_predictions = []

        state = mdp[n]._trajectory_to_state_action(trajectory)[:, 0].astype(int)
        action = mdp[n]._trajectory_to_state_action(trajectory)[:, 1].astype(int)

        predicted_state = np.nan

        for n, a in enumerate(action):
            
            if n % n_pred == 0:
                start_state = state[n]
            else:
                start_state = predicted_state

            # Set impossible actions to -inf
            trial_Q = Q.copy()
            trial_Q[~np.where(mdp[n].sas[start_state])[0]] = -np.inf

            predicted_state = np.argmax(Q)
            trajectory_predictions.append(predicted_state)

            trial_reward = np.zeros(mdp[n].n_actions)

            if not kernel:
                trial_reward[int(a)] += 1
            else:
                trial_reward = deg_sq_exp_kernel(np.arange(mdp[n].n_actions) * (360 / mdp[n].n_actions), a * 60, ls=ls)

            error = trial_reward - Q

            Q += learning_rate * error

            

        predictions.append(trajectory_predictions)

    return Q, predictions
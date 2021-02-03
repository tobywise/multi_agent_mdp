from multi_agent_mdp.environments import Environment
from .algorithms.dynamic_programming import ValueIteration
from operator import pos
import numpy as np
from .mdp import MDP
from .algorithms.base import Algorithm
from .algorithms.action_selection import *
from typing import Dict
from .plotting import plot_agent
import matplotlib.pyplot as plt
from typing import List


class Agent():
    """
    Represents an agent that can act within a MDP.
    """

    def __init__(self, name:str, n_moves:int=1, algorithm:Algorithm=ValueIteration, algorithm_kwargs:Dict={},
                 action_selector:ActionSelector=MaxActionSelector, action_kwargs:Dict={}):
        """ Defines an Agent. 

        Args:
            name (str): Name of the agent.
            n_moves (int, optional): Number of moves the agent makes on each turn. This can be used for stepping through
            sequences of agents' turns in the environment, and can be used by planning algorithms that account for different
            agents' future actions. Defaults to 1.
            algorithm (Algorithm, optional): The algorithm used to calcualte action values. Defaults to ValueIteration.
            algorithm_kwargs (Dict, optional): Keyword arguments for the algorithm. Defaults to {}.
            action_selector (ActionSelector, optional): Method used to select actions. Defaults to MaxActionSelector.
            action_kwargs (Dict, optional): Keyword arguments for action selection. Defaults to {}.
        """

        # Type checking
        if not isinstance(name, str):
            raise TypeError('Name must be a string indicating the name of the agent')
    
        # Plugin algorithm for solving MDP and action selection methods
        if not isinstance(algorithm_kwargs, dict):
            raise TypeError('Algorithm keyword arguments should be supplied as a dictionary, got type {0}'.format(type(algorithm_kwargs)))
        if not isinstance(action_kwargs, dict):
            raise TypeError('Action selector keyword arguments should be supplied as a dictionary, got type {0}'.format(type(action_kwargs)))

        self.algorithm = algorithm(**algorithm_kwargs)
        self.action_selector = action_selector(**action_kwargs)

        self.name = name
        self.n_moves = n_moves

    def _attach(self, mdp:MDP, env:Environment, index:int, position:int, reward_function:np.ndarray, consumes:List[int]=[]):

        return AttachedAgent(name=self.name, n_moves=self.n_moves, reward_function=reward_function, consumes=consumes,
                             algorithm=self.algorithm, action_selector=self.action_selector, 
                             parent_mdp=mdp, parent_environment=env, position=position, index=index)


class AttachedAgent():
    """ Internal class - do not use directly """

    def __init__(self, name: str, n_moves:int, reward_function:np.ndarray, consumes:List[int],
                algorithm: Algorithm, action_selector: ActionSelector, 
                parent_mdp:MDP, parent_environment:Environment, position:int, index:int):
       
        self.name = name
        self.n_moves = n_moves
        self.reward_function = reward_function
        self.consumes = consumes
        self.consumed = np.zeros(len(reward_function))
        self.algorithm = algorithm
        self.action_selector = action_selector
        self.agent_idx = index
        self.agent_feature_idx = index + parent_mdp._MDP__n_non_agent_features  # Index of the corresponding feature for this agent

        self._parent_mdp = parent_mdp
        self._parent_environment = parent_environment

        self.algorithm._attach(self, parent_environment)

        # Provide the algorithm with information about the environment
        self.algorithm.environment = parent_environment

        # Add feature to the MDP
        self.__parent_mdp.add_agent_feature(position)

        self._attached = True
        
        self.__starting_position = None
        self.position = position
        self.position_history = [self.__starting_position]

        # Policy
        self.pi = None
        self.pi_p = None

    @property
    def _attached(self):
        return self.__attached

    @_attached.setter
    def _attached(self, is_attached:bool):
        assert isinstance(is_attached, bool), 'is_attached should be bool'
        self.__attached = is_attached

    @property
    def _parent_mdp(self):
        return self.__parent_mdp

    @_parent_mdp.setter
    def _parent_mdp(self, mdp:MDP):
        self.__parent_mdp = mdp

    @property
    def position(self):
        return self.__position

    @position.setter
    def position(self, new_position:int):

        if new_position >= self._parent_mdp.n_states:
            raise ValueError("Position must be less than the number of states in the parent MDP. " 
                             "Provided position {0}, MDP has {1} states".format(self._parent_mdp.n_states))

        if self.__starting_position is None:
            self.__starting_position = new_position

        self.__parent_mdp.update_agent_feature(self.agent_idx, new_position)

        self.__position = new_position

    def get_policy(self):

        self.pi_p = self.action_selector.get_pi_p(self.algorithm.q_values)
        self.pi = self.action_selector.get_pi(self.algorithm.q_values)

    def fit(self, **kwargs):
        
        # TODO allow this to handle online and offline

        # Solve MDP
        self.algorithm.fit(self.__parent_mdp, self.reward_function, self.position, **kwargs)

        # Get policy
        self.get_policy()
        
    def step(self):

        if not self.algorithm.fit_complete:
            raise AttributeError("Agent has not been fit yet")

        self.get_policy()

        if self.pi[self.position] < 0:
            raise ValueError("No valid action for this state. State action values for the current state may not have been estimated yet.")

        next_state = self._parent_mdp.get_next_state(self.position, self.pi[self.position])

        # Move the agent
        self.position = int(next_state)
        self.position_history.append(self.position)
        self.__parent_mdp.update_agent_feature(self.agent_idx, self.position)

    def reset(self):
        self.position = self.__starting_position
        self.pi = None
        self.pi_p = None
        self.position_history = []

    def plot(self, ax:plt.axes, marker:str="X", *args, **kwargs):
        """
        Adds the agent to an existing plot, represented by the given marker style.

        Args:
            ax (plt.axes): Existing axes on which to plot the agent.
            marker (str, optional): Marker style used to represent the agent. Defaults to "X".
        """
        
        position = self._parent_mdp.state_to_position(self.position)  # Get position in space
        plot_agent(ax=ax, position=position, marker=marker, *args, **kwargs)


    # def generate_trajectory(self, n_steps=5, start_state=None):

    #     # Assumes deterministic mdp
    #     if not self.solver.fit_complete:
    #         raise AttributeError('Solver has not been fit yet')

    #     if start_state is None:
    #         start_state = self.position

    #     trajectory = [start_state]

    #     for s in range(n_steps):
    #         trajectory.append(np.argmax(self.mdp.sas[trajectory[-1], self.solver._get_pi_[trajectory[-1]], :]))

    #     return trajectory
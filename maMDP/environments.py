import numpy as np
from .mdp import MDP, Observation, Trajectory
from typing import List, Union, Tuple, Dict
import matplotlib.pyplot as plt
from fastprogress import progress_bar
import warnings
from copy import copy


class Environment():

    def __init__(self, mdp:MDP, agents:Dict['Agent', Tuple[int, np.ndarray, List[int]]], name:str=''):
        """
        Create and environment with multiple agents interacting with a common mdp.

        Args:
            mdp (MDP): MDP representing the environment the agents are interacting with.
            agents (Dict[Agent, Tuple[int, np.ndarray, List[int]]]): Dictionary with keys representing Agents and values 
            equal to a tuple representing:
                1) Their position in the MDP (i.e. the state at which they start)
                2) The agent's reward function. This should have as many entries as the MDP has features, plus the number of agents
                in the environment. The first n entries correspond to the features in the MDP, subsequent entries correspond to
                the agents (in the order they're supplied).
                3) Features that the agent consumes upon encountering (i.e. once the agent enters a state with this feature, 
                the feature disappears from the state).
                For example, {Agent1: (23, [1, 0, 0], [0])} would set the object Agent1 to start at state 23, with a reward function
                with a weight of 1 for the first feature, 0 for the second feature, and 0 for itself, and indicate that it 
                consumes the first feature.
            name (str, optional): Name of the environment. Defaults to ''.
        """
        self.name = name
        self._set_mdp_and_agents(agents, mdp)

        # Useful numbers
        self.n_agents = len(self.agents)
        self.n_steps = 0
        self.n_states = mdp.n_states

    @property
    def mdp(self):
        return self.__mdp

    @property
    def agents(self):
        return self.__agents

    def _set_mdp_and_agents(self, agents:Dict['Agent', Tuple[int, np.ndarray, List[int]]], mdp:MDP):

        # Get number of agents
        self.n_agents = len(agents)
        self.agent_names = [agent.name for agent in agents.keys()]
        
        self.__mdp = mdp
        self.__mdp.__environment = self

        # Remove any agent-related features that may already be present in the MDP
        if self.__mdp.n_agents > 0:
            self.__mdp.remove_agent_features()

        # Attach agents
        self.__agents = {}

        original_mdp_n_features = mdp.n_features

        for n, agent_object in enumerate(agents.keys()):

            position, reward_weights, consumes = agents[agent_object]

            # Check supplied values are correct
            if not isinstance(reward_weights, np.ndarray) and not isinstance(reward_weights, list):
                raise TypeError('Agent {0}: Reward function must be a 1D array or list, got type {1}'.format(agent_object.name, reward_weights))

            reward_weights = np.array(reward_weights)

            if not reward_weights.ndim == 1:
                raise AttributeError("Agent {0}: Reward function array must be 1-dimensional".format(agent_object.name))

            if not isinstance(position, int):
                raise TypeError('Agent {0}: Position should be an int'.format(agent_object.name))

            if not len(reward_weights) == original_mdp_n_features + self.n_agents:
                raise AttributeError("Agent {0}: Agent reward function must have the same number of entries"
                                    "as the MDP has features plus number of agents. Reward function has {1} entries, "
                                    "MDP has {2} features, {3} agents provided".format(agent_object.name, len(reward_weights), 
                                    original_mdp_n_features, self.n_agents))

            if any([i > len(reward_weights) - 1 for i in consumes]):
                raise ValueError("Agent {0}: Provided a feature to consume that is out of range. Indexes provided = {1}, reward "
                                "function indicates {2} features".format(agent_object.name, consumes, len(reward_weights)))
            
            
            agent_object = agent_object._attach(self.mdp, self, n, position, reward_weights, consumes)
            self.__agents[agent_object.name] = agent_object

    def _sas_changed(self):
        """ Tell agents that the MDP's transition structure has changed """
        
        for agent in self.agents:
            agent._sas_changed()

    def _features_changed(self):
        """ Tell agents that the MDP's features have changed """
        
        for agent in self.agents:
            agent._features_changed()


    def _check_agent_name(self, agent_name:str):
        """ Utility function to check that a given agent name is in the environment """

        if not agent_name in self.agent_names:
            raise ValueError("Agent named {0} is not present in the environment, valid names are {1}".format(agent_name, self.agent_names))

    def fit(self, agent_name:str, n_steps:int=None, n_observations:int=None):
        """
        Calculates action values for a given agent.

        Args:
            agent_name (str): Name of the agent to fit.
            n_steps (int, optional): Number of steps to plan ahead, if used by the algorithm. Defaults to None.
            n_observations (int, optional): Number of recent observations to learn from, if used by the algorithm. Defaults to None.
        """

        self._check_agent_name(agent_name)
        self.agents[agent_name].fit(n_steps=n_steps, n_observations=n_observations)

    def step(self, agent_name:str, n_steps:int=None) -> int:
        """
        Moves a given agent one step in the environment.

        Args:
            agent_name (str): Name of the agent to step.
            n_steps (int, optional): Number of steps to plan ahead, if used by the algorithm. Defaults to None.

        Returns:
            int: New position
        """

        self._check_agent_name(agent_name)

        obs = self.agents[agent_name].step()

        # Consume features if necessary - i.e. set feature to zero
        self._consume_features(agent_name)

        self.n_steps += 1

        return obs

    def step_proba(self, agent_name:str, position:int=None) -> np.ndarray:
        """
        Generates a probabilistic representation of the results of stepping an agent. 

        Args:
            agent_name (str): Agent to step.
            position (int): Starting position. If None, uses the agent's current position. Defaults to None.

        Returns:
            np.ndarray: Array representing probability of being in each state in the MDP after one step from the provided position.
        """

        self._check_agent_name(agent_name)
        next_states_p = self.agents[agent_name].step_proba(position=position)

        # Consume features if necessary - i.e. set feature to zero
        # self._consume_features(agent_name)

        self.n_steps += 1

        return next_states_p


    def _consume_features(self, agent_name:str):
        """
        Removes consumed features and keeps track of the number consumed.

        Args:
            agent_name (str): Agent to check feature consumption for.
        """ 
        
        if len(self.agents[agent_name].consumes):
            consumed = self.mdp.consume_features(self.agents[agent_name].consumes, self.agents[agent_name].position)
            self.agents[agent_name].consumed += consumed  # Record consumption
        
    def _check_agents_caught(self, agent_name:str):

        # Check whether any agents should be eaten
        caught = False
        for f in self.agents[agent_name].consumes:
            for agent in self.agents.values():
                if agent.agent_feature_idx == f:
                    if self.agents[agent_name].position == agent.position:
                        caught = True

        return caught

    def step_multi(self, agent_name:str, n_steps:int=None, refit:bool=False, progressbar:bool=False,
                  n_observations:int=1, adjust_planning_steps:bool=True, stop_on_caught:bool=True,
                  stop_states:List[int]=None) -> List[int]:
        """
        Moves an agent multiple steps within the environment.

        Args:
            agent_name (str): Name of the agent to step.
            n_steps (int, optional): Number of steps to move. If None, uses the agent's n_moves attribute. Defaults to None.
            refit (bool, optional). If true, refits the action value estimation every step. This is useful for online models,
            which only estimate values for actions that can be taken from the current state, and would otherwise raise an error. 
            Defaults to False.
            progressbar (bool, optional). If true, shows a progress bar. Defaults to False.
            n_observations (int, optional). Number of observations to supply to the algorithm, where 1 represents just the most recent
            algorithm, 2 the 2 the most recent etc. Useful for model-free algorithms that may only learn from the most recent 
            observation. If set to None, all observations are used. By default, only the most recent observation is used. Defaults to 1.
            adjust_planning_steps (bool or int, optional). If true, the number of steps used for the planning algorithm (if it plans
            for a set number of steps, like MCTS) is adjusted based on the number of steps the agent is expected to make (as set in the 
            `agent.n_steps` attribute). If an int is provided, this (minus 1 for each move made) will be used as the number of planning 
            steps instead. Defaults to True.
            stop_on_caught (bool, optional): If true, stops moving agents when one of them gets caught.
            stop_states (List[int], optional): States where the agent should stop. Defaults to None.

        Returns:
            List[int]: States the agent has moved to.
        """

        self._check_agent_name(agent_name)

        if n_steps is None:
            n_steps = self.agents[agent_name].n_moves

        if progressbar:
            steps = progress_bar(range(n_steps))
        else:
            steps = range(n_steps)

        observations = []

        for i in steps:
            if refit:
                if adjust_planning_steps == True:
                    self.fit(agent_name, n_steps - i, n_observations)
                elif isinstance(adjust_planning_steps, int):
                    self.fit(agent_name, adjust_planning_steps - i, n_observations)
                else:
                    self.fit(agent_name, n_observations=n_observations)
            obs = self.step(agent_name)
            observations.append(obs)

            # Check if the agent got caught
            caught = self._check_agents_caught(agent_name)

            if caught and stop_on_caught:
                warnings.warn("Agent was caught, stopping")
                observations[-1].caught = True
                return Trajectory(self.mdp, self.agents[agent_name], observations)

            # Check if the agent reached a state where it should stop
            if stop_states is not None and self.agents[agent_name].position in stop_states:
                return Trajectory(self.mdp, self.agents[agent_name], observations)

        return Trajectory(self.mdp, self.agents[agent_name], observations)

    def step_multi_interactive(self, agent_names:List[str]=None, n_steps:int=10, refit:bool=False, progressbar:bool=False,
                               stop_on_caught:bool=True, adjust_planning_steps:bool=True):
        """
        Moves an agent multiple steps within the environment.

        Args:
            agent_names (List[str], optional): Names of agents to step. By default, includes every agent. Agents move in the
            order given.
            n_steps (int, optional): Number of turns taken (each turn may involve more than 1 step).  Defaults to 10.
            refit (bool, optional). If true, refits the action value estimation every step. This is useful for online models,
            which only estimate values for actions that can be taken from the current state, and would otherwise raise an error. 
            Defaults to False.
            progressbar (bool, optional). If true, shows a progress bar. Defaults to False.
            stop_on_caught (bool, optional): If true, stops moving agents when one of them gets caught.
            adjust_planning_steps (bool, optional). If true, the number of steps used for the planning algorithm (if it plans
            for a set number of steps, like MCTS) is adjusted based on the number of steps simulated. Defaults to True.
        """

        if agent_names is None:
            agent_names = self.agent_names

        for agent_name in agent_names:
            self._check_agent_name(agent_name)

        if progressbar:
            steps = progress_bar(range(n_steps))
        else:
            steps = range(n_steps)

        agent_steps = dict([(a, self.agents[a].n_moves) for a in agent_names])

        for i in steps:
            
            for agent_name in agent_names:

                if refit:
                    if adjust_planning_steps:
                        observations = self.step_multi(agent_name, agent_steps[agent_name], refit=refit, 
                                                    adjust_planning_steps=int((agent_steps[agent_name] * n_steps) - i), stop_on_caught=stop_on_caught)
                    else:
                        observations = self.step_multi(agent_name, agent_steps[agent_name], refit=refit, stop_on_caught=stop_on_caught)

                if observations[-1].caught and stop_on_caught:
                    warnings.warn("Agent was caught, stopping")
                    return

    def reset(self):
        """
        Resets the agents and the MDP to their starting states.
        """

        for agent in self.agents.values():
            agent.reset()
        self.mdp.reset()

    def set_agent_position(self, agent_name:str, new_position:int):
        """
        Moves a given agent to a new position.

        Args:
            agent_name (str): Name of the agent to move.
            new_position (int): New position in the underlying MDP.
        """

        self._check_agent_name(agent_name)
        self.agents[agent_name].position = new_position

    def get_agent_position_history(self, agent_name:str) -> List[int]:
        """
        Gets an agent's position history.

        Args:
            agent_name (str): Name of the agent.

        Returns:
            List[int]: A list of state IDs previously occupied by the agent.
        """

        self._check_agent_name(agent_name)
        return self.agents[agent_name].position_history

    def move_agent(self, agent_name:str, action:int) -> Observation:
        """
        Make an agent take the given action from its current state. Returns an observation representing
        the stating state, action taken, state reached, and reward received.

        Args:
            agent_name (str): Agent to move
            action (int): Action for the agent to take.

        Returns:
            Observation: Observation representing the stating state, action taken, state reached, and reward received.
        """

        self._check_agent_name(agent_name)
        return self.agents[agent_name].move(action)

    def get_agent_observation_history(self, agent_name:str) -> Trajectory:

        self._check_agent_name(agent_name)
        return self.agents[agent_name].observation_history

    def get_agent_q_values(self, agent_name:str) -> np.ndarray:
        """
        Gets estimated action values for a given agent.

        Args:
            agent_name (str): Name of the agent to get Q values for.

        Returns:
            np.ndarray: Array of Q values
        """

        self._check_agent_name(agent_name)
        return self.agents[agent_name].algorithm.q_values

    def get_agent_pi(self, agent_name:str) -> np.ndarray:
        """
        Get deterministic policy for a given agent (i.e. a single action to be taken in each state)

        Args:
            agent_name (str): Name of the agent to getp olicy for.

        Returns:
            np.ndarray: 1D array of actions to be taken in each state
        """

        self._check_agent_name(agent_name)
        if self.agents[agent_name].algorithm.fit_complete:
            return self.agents[agent_name].pi
        else:
            raise AttributeError('Agent has not been fit')

    def get_agent_pi_p(self, agent_name:str) -> np.ndarray:
        """
        Get probabilistic policy for a given agent (i.e. probability of each action to be taken in each state)

        Args:
            agent_name (str): [description]

        Returns:
            np.ndarray: 2D array of action probabilities for each state (n_states, n_actions)
        """
        
        self._check_agent_name(agent_name)
        if self.agents[agent_name].algorithm.fit_complete:
            return self.agents[agent_name].pi_p
        else:
            raise AttributeError('Agent has not been fit')

    def to_dict(self):
        raise NotImplementedError

    def to_json(self):
        raise NotImplementedError


    # Plotting methods
    def plot(self, colours:list=None, alphas:list=None, agent_markers:Dict[str, str]={}, 
            agent_colours:Dict[str, str]={}, mdp_plotting_kwargs:Dict={}, agent_plotting_kwargs:Dict={}, 
            ax:plt.axes=None) -> plt.axes:
        """
        Plots the environment, showing the underlying MDP (if it implements plotting functions) and the
        agents interacting with it.

        Args:
            colours (list, optional): Colours for the features in the MDP. Defaults to None.
            alphas (list, optional): Alpha values for the features in the MDP. Defaults to None.
            agent_markers (Dict[str, str], optional): Markers to represent the agents, keys represent
            agent names, values represent Matplotlib markers. Defaults to {}.
            agent_colours (Dict[str, str], optional): Colours used to represent the agents, keys represent
            agent names, colours represent colours. Defaults to {}.
            mdp_plotting_kwargs (Dict, optional): Keyword arguments to be passed to the underlying MDP plot function.
            agent_plotting_kwargs (Dict, optional): Keyword arguments to be passed to the underlying agent plot function.
            ax (plt.axes, optional): Plotting axes.

        Returns:
        plt.axes: Axes for plot
        """
        ax = self.mdp.plot(colours=colours, alphas=alphas, ax=ax, **mdp_plotting_kwargs)
        for agent_name, agent in self.agents.items():
            if agent_name in agent_colours:
                colour = agent_colours[agent_name]
            else:
                colour = 'black'
            if agent_name in agent_markers:
                agent.plot(ax, marker=agent_markers[agent_name], color=colour, **agent_plotting_kwargs)
            else:
                agent.plot(ax, color=colour, **agent_plotting_kwargs)

        return ax

    def plot_trajectory(self, trajectory:List[int], ax:plt.axes, *args, **kwargs):
        """
        Plots a trajectory in the underlying MDP (if it implements the appropriate plotting function).

        See the plot_trajectory() method of the MDP class for more information.

        Args:
            trajectory (List[int]): List of visited states
            ax (plt.axes): Axes on which to plot the trajectory
        """
        
        # Convert trajectory of state IDs to 
        self.mdp.plot_trajectory(ax=ax, trajectory=trajectory, *args, **kwargs)

    def plot_sequence(self, trajectory:List[int], ax:plt.axes, *args, **kwargs):
        """
        Plots a trajectory in the underlying MDP (if it implements the appropriate plotting function) with
        numbers representing each step in the trajectory.

        See the plot_sequence() method of the MDP class for more information.

        Args:
            trajectory (List[int]): List of visited states
            ax (plt.axes): Axes on which to plot the trajectory
        """
        
        # Convert trajectory of state IDs to 
        self.mdp.plot_sequence(ax=ax, trajectory=trajectory, *args, **kwargs)

    def plot_state_values(self, agent_name:str, *args, **kwargs) -> plt.axes:
        """
        Plots state values for a given agent within the underlying MDP (if it implements the appropriate plotting function).

        Args:
            agent_name (str): Name of the agent to plot state values for

        Returns:
            plt.axes: Figure axes
        """

        self._check_agent_name(agent_name)

        if not self.agents[agent_name].algorithm.fit_complete:
            raise AttributeError("Agent has not yet been fit")
        
        ax = self.mdp.plot_state_values(self.agents[agent_name].algorithm.values, *args, **kwargs)

        return ax


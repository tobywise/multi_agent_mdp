import numpy as np
from .mdp import MDP
from typing import List, Union, Tuple, Dict
import matplotlib.pyplot as plt
from fastprogress import progress_bar
import warnings
from copy import copy


class Environment:
    def __init__(
        self,
        mdp: MDP,
        agents: Dict["Agent", Tuple[int, np.ndarray, List[int]]],
        name: str = "",
    ):
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

    def _set_mdp_and_agents(
        self, agents: Dict["Agent", Tuple[int, np.ndarray, List[int]]], mdp
    ):

        # Get number of agents
        self.n_agents = len(agents)
        self.agent_names = [agent.name for agent in agents.keys()]

        self.__mdp = mdp

        # Remove any agent-related features that may already be present in the MDP
        if self.__mdp.n_agents > 0:
            self.__mdp.remove_agent_features()

        # Attach agents
        self.__agents = {}

        original_mdp_n_features = mdp.n_features

        for n, agent_object in enumerate(agents.keys()):

            position, reward_function, consumes = agents[agent_object]

            # Check supplied values are correct
            if not isinstance(reward_function, np.ndarray) and not isinstance(
                reward_function, list
            ):
                raise TypeError(
                    "Agent {0}: Reward function must be a 1D array or list, got type {1}".format(
                        agent_object.name, reward_function
                    )
                )

            reward_function = np.array(reward_function)

            if not reward_function.ndim == 1:
                raise AttributeError(
                    "Agent {0}: Reward function array must be 1-dimensional".format(
                        agent_object.name
                    )
                )

            if not isinstance(position, int):
                raise TypeError(
                    "Agent {0}: Position should be an int".format(agent_object.name)
                )

            if not len(reward_function) == original_mdp_n_features + self.n_agents:
                raise AttributeError(
                    "Agent {0}: Agent reward function must have the same number of entries"
                    "as the MDP has features plus number of agents. Reward function has {1} entries, "
                    "MDP has {2} features, {3} agents provided".format(
                        agent_object.name,
                        len(reward_function),
                        original_mdp_n_features,
                        self.n_agents,
                    )
                )

            if any([i > len(reward_function) - 1 for i in consumes]):
                raise ValueError(
                    "Agent {0}: Provided a feature to consume that is out of range. Indexes provided = {1}, reward "
                    "function indicates {2} features".format(
                        agent_object.name, consumes, len(reward_function)
                    )
                )

            agent_object = agent_object._attach(
                self.mdp, self, n, position, reward_function, consumes
            )
            self.__agents[agent_object.name] = agent_object

    def _check_agent_name(self, agent_name: str):
        """Utility function to check that a given agent name is in the environment"""

        if not agent_name in self.agent_names:
            raise ValueError(
                "Agent named {0} is not present in the environment, valid names are {1}".format(
                    agent_name, self.agent_names
                )
            )

    def fit(self, agent_name: str, n_moves: Tuple[Tuple[int]]):
        """
        Calculates action values for a given agent.

        Args:
            agent_name (str): Name of the agent to fit.
            n_moves (Tuple[Tuple[int]]): Used by simulation algorithms to determine how many steps to simulate for each agent.
            This represents the number of moves each agent can take per turn. This is a tuple of tuples, where the first tuple
            corresponds to the primary agent, and the remaining tuples correspond to the other agents. Within each tuple, the first
            element is the number of moves the agent can take in the first turn, the second element is the number of moves for the
            second turn, etc. If None, the n_steps argument is used to determine how many steps to simulate, and the number of moves
            per turn is determined based on information present in the Agent classes.
        """

        self._check_agent_name(agent_name)
        self.agents[agent_name].fit(n_moves=n_moves)

    def step(self, agent_name: str, n_moves: Tuple[Tuple[int]] = None) -> int:
        """
        Moves a given agent one step in the environment.

        NOTE: Does not update the agent's action values by fitting. To do this, use the fit method, or the step_multi method with refit=True.

        Args:
            agent_name (str): Name of the agent to step.
            n_steps (int, optional): Number of steps to plan ahead, if used by the algorithm. Defaults to None.

        Returns:
            int: New position
        """

        self._check_agent_name(agent_name)
        self.agents[agent_name].step()

        # Consume features if necessary - i.e. set feature to zero
        self._consume_features(agent_name)

        self.n_steps += 1

        return self.agents[agent_name].position

    def step_proba(self, agent_name: str, position: int = None) -> np.ndarray:
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

    def _consume_features(self, agent_name: str):
        """
        Removes consumed features and keeps track of the number consumed.

        Args:
            agent_name (str): Agent to check feature consumption for.
        """

        if len(self.agents[agent_name].consumes):
            consumed = self.mdp.consume_features(
                self.agents[agent_name].consumes, self.agents[agent_name].position
            )
            self.agents[agent_name].consumed += consumed  # Record consumption

    def _check_agents_caught(self, agent_name: str):

        # Check whether any agents should be eaten
        caught = False
        for f in self.agents[agent_name].consumes:
            for agent in self.agents.values():
                if agent.agent_feature_idx == f:
                    if self.agents[agent_name].position == agent.position:
                        caught = True

        return caught

    def step_multi(
        self,
        agent_name: str,
        n_steps: int = None,
        n_moves_plan: List[List[int]] = None,
        n_turns_plan: int = 1,
        refit: bool = False,
        progressbar: bool = False,
        adjust_planning_steps: bool = True,
        stop_on_caught: bool = True,
    ) -> List[int]:
        """
        Moves an agent multiple steps within the environment.

        Args:
            agent_name (str): Name of the agent to step.
            n_steps (int, optional): Number of steps to move. If None, the number of steps is taken as the number of steps defined
            in the Agent.n_moves attribute. Defaults to None.
            n_moves_plan (List[List[int]]): Used by simulation algorithms to determine how many steps to simulate for each agent.
            This represents the number of moves each agent can take per turn. This is a list of list, where the first list
            corresponds to the primary agent, and the remaining lists correspond to the other agents. Within each list, the first
            element is the number of moves the agent can take in the first turn, the second element is the number of moves for the
            second turn, etc. If None, the number of moves per turn is determined based on information present in the Agent classes
            and the n_turns argument.
            n_turns_plan (int, optional): Number of turns to simulate (used for simulation algorithms, otherwise ignored).
            Only used if n_moves_plan is not specified. Defaults to 1.
            refit (bool, optional). If true, refits the action value estimation every step. This is useful for online models,
            which only estimate values for actions that can be taken from the current state, and would otherwise raise an error.
            Defaults to False.
            progressbar (bool, optional). If true, shows a progress bar. Defaults to False.
            adjust_planning_steps (bool, optional). If true, the number of steps used for the planning algorithm (if it plans
            for a set number of steps, like MCTS) is adjusted based on the number of steps the agent is expected to make (as set in the
            `agent.n_moves` attribute or the `n_moves_plan` argument). If an int is provided, this represents the total expected
            number of moves, and planning steps will only decrement once they go beyond this. For example, if `n_moves_plan` indicates
            that the agent should plan 6 moves ahead but `adjust_planning_steps` is set to 10, the number of planning steps will
            only be decremented after the 4th move. Defaults to True.
            stop_on_caught (bool, optional): If true, stops moving agents when one of them gets caught.

        Returns:
            List[int]: States the agent has moved to.
        """

        # Get number of moves for each agent if not provided
        if n_moves_plan is None:
            agent_names = [agent_name] + [
                i for i in self.agent_names if not i == agent_name
            ]  # Put primary agent first
            n_moves_plan = [
                [self.agents[i].n_moves] * n_turns_plan for i in agent_names
            ]
        else:
            n_moves_plan = copy(
                n_moves_plan
            )  # avoid modifying the original that was provided

        self._check_agent_name(agent_name)

        if n_steps is None:
            n_steps = self.agents[agent_name].n_moves

        if n_moves_plan[0][0] < n_steps and refit and not adjust_planning_steps:
            warnings.warn(
                f"Number of planning moves for the first turn ({n_moves_plan[0][0]}) is less than the number of steps to move ({n_steps})."
            )

        if progressbar:
            steps = progress_bar(range(n_steps))
        else:
            steps = range(n_steps)

        primary_agent_total_planning_moves = sum(n_moves_plan[0])

        positions = []
        for i in steps:
            if refit:
                if adjust_planning_steps == True:
                    if n_moves_plan[0][0] > 1:
                        n_moves_plan[0][0] -= 1  # Adjust number of moves for the first turn
                    else:
                        warnings.warn(
                            "Number of planning moves for the first turn is already 1. Cannot adjust planning steps."
                        )
                    self.fit(agent_name, n_moves=n_moves_plan)

                # If int, decrement value until it reaches the number of steps to move
                # TODO check this works, it might not...
                elif isinstance(adjust_planning_steps, int):
                    self.fit(agent_name, n_moves=n_moves_plan)

                    if adjust_planning_steps == primary_agent_total_planning_moves:
                        adjust_planning_steps = True
                    else:
                        adjust_planning_steps -= 1

                else:
                    self.fit(agent_name, n_moves=n_moves_plan)

            self.step(agent_name)
            positions.append(self.agents[agent_name].position)

            # Check if the agent got caught
            caught = self._check_agents_caught(agent_name)

            if caught and stop_on_caught:
                warnings.warn("Agent was caught, stopping")
                return False

        return positions

    def step_multi_interactive(
        self,
        agent_names: List[str] = None,
        n_moves: Tuple[Tuple[int]] = None,
        n_moves_plan: Tuple[Tuple[int]] = None,
        n_turns: int = 1,
        refit: bool = False,
        progressbar: bool = False,
        stop_on_caught: bool = True,
        adjust_planning_steps: bool = True,
    ):
        """
        Moves an multiple agents multiple steps within the environment.

        Args:
            agent_names (List[str], optional): Names of agents to step. By default, includes every agent. Agents move in the
            order given.
            n_moves (Tuple[Tuple[int]]): The number of moves taken by each agent per turn. This is a tuple of tuples, where the first tuple
            corresponds to the primary agent, and the remaining tuples correspond to the other agents. Within each tuple, the first
            element is the number of moves the agent can take in the first turn, the second element is the number of moves for the
            second turn, etc. If None, the number of moves per turn is determined based on information present in the Agent classes
            and the n_turns argument.
            n_moves_plan (Tuple[Tuple[int]]): Used by simulation algorithms to determine how many steps to simulate for each agent.
            This represents the number of moves each agent can take per turn. This is a tuple of tuples, where the first tuple
            corresponds to the primary agent, and the remaining tuples correspond to the other agents. Within each tuple, the first
            element is the number of moves the agent can take in the first turn, the second element is the number of moves for the
            second turn, etc. If None, the n_steps argument is used to determine how many steps to simulate, and the number of moves
            per turn is determined based on information present in the Agent classes.
            refit (bool, optional). If true, refits the action value estimation every step. This is useful for online models,
            which only estimate values for actions that can be taken from the current state, and would otherwise raise an error.
            Defaults to False.
            progressbar (bool, optional). If true, shows a progress bar. Defaults to False.
            stop_on_caught (bool, optional): If true, stops moving agents when one of them gets caught.
            adjust_planning_steps (bool, optional). If true, the number of steps used for the planning algorithm (if it plans
            for a set number of steps, like MCTS) is adjusted based on the number of steps the agent is expected to make (as set in the
            `agent.n_moves` attribute or the `n_moves_plan` argument). If an int is provided, this represents the total expected
            number of moves, and planning steps will only decrement once they go beyond this. For example, if `n_moves_plan` indicates
            that the agent should plan 6 moves ahead but `adjust_planning_steps` is set to 10, the number of planning steps will
            only be decremented after the 4th move. Defaults to True.
        """

        # Get number of moves for each agent if not provided
        if n_moves is None:
            agent_names = [agent_name] + [
                i for i in self.agent_names if not i == agent_name
            ]  # Put primary agent first
            n_moves = [
                [[self.agents[i].n_moves] * n_turns for i in agent_names]
            ]
        else:
            n_moves = copy(n_moves)  # avoid modifying the original that was provided
            n_turns = len(n_moves[0])  # Get number of turns from n_moves

        # Get the number of planning moves
        if n_moves_plan is None:
            n_moves_plan = copy(n_moves)

        if agent_names is None:
            agent_names = self.agent_names

        for agent_name in agent_names:
            self._check_agent_name(agent_name)

        if progressbar:
            turns = progress_bar(range(n_turns))
        else:
            turns = range(n_turns)

        agent_steps = dict([(a, self.agents[a].n_moves) for a in agent_names])

        # Need to determine the total number of planning steps
        if adjust_planning_steps is True:
            adjust_planning_steps = sum((sum(i) for i in n_moves))

        for t in turns:

            for n, agent_name in enumerate(agent_names):

                positions = self.step_multi(
                    agent_name,
                    agent_steps[agent_name],
                    refit=refit,
                    n_steps=n_moves[n][t],
                    n_moves_plan=n_moves_plan,
                    adjust_planning_steps=adjust_planning_steps,
                    stop_on_caught=stop_on_caught,
                )

                if positions == False and stop_on_caught:
                    warnings.warn("Agent was caught, stopping")
                    return

                # Decrement adjust_planning_steps (this should work...)
                if isinstance(adjust_planning_steps, int):
                    adjust_planning_steps -= n_moves[n][t]

                # Update number of planning moves for each agent
                n_moves_plan[n] = n_moves_plan[n][t+1:]
                

    def reset(self):
        """
        Resets the agents and the MDP to their starting states.
        """

        for agent in self.agents.values():
            agent.reset()
        self.mdp.reset()

    def set_agent_position(self, agent_name: str, new_position: int):
        """
        Moves a given agent to a new position.

        Args:
            agent_name (str): Name of the agent to move.
            new_position (int): New position in the underlying MDP.
        """

        self._check_agent_name(agent_name)
        self.agents[agent_name].position = new_position

    def get_agent_position_history(self, agent_name: str) -> List[int]:
        """
        Gets an agent's position history.

        Args:
            agent_name (str): Name of the agent.

        Returns:
            List[int]: A list of state IDs previously occupied by the agent.
        """

        self._check_agent_name(agent_name)
        return self.agents[agent_name].position_history

    def get_agent_q_values(self, agent_name: str) -> np.ndarray:
        """
        Gets estimated action values for a given agent.

        Args:
            agent_name (str): Name of the agent to get Q values for.

        Returns:
            np.ndarray: Array of Q values
        """

        self._check_agent_name(agent_name)
        return self.agents[agent_name].algorithm.q_values

    def get_agent_pi(self, agent_name: str) -> np.ndarray:
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
            raise AttributeError("Agent has not been fit")

    def get_agent_pi_p(self, agent_name: str) -> np.ndarray:
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
            raise AttributeError("Agent has not been fit")

    def to_dict(self):
        raise NotImplementedError

    def to_json(self):
        raise NotImplementedError

    # Plotting methods
    def plot(
        self,
        colours: list = None,
        alphas: list = None,
        agent_markers: Dict[str, str] = {},
        agent_colours: Dict[str, str] = {},
        mdp_plotting_kwargs: Dict = {},
        agent_plotting_kwargs: Dict = {},
        ax: plt.axes = None,
    ) -> plt.axes:
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
                colour = "black"
            if agent_name in agent_markers:
                agent.plot(
                    ax,
                    marker=agent_markers[agent_name],
                    color=colour,
                    **agent_plotting_kwargs,
                )
            else:
                agent.plot(ax, color=colour, **agent_plotting_kwargs)

        return ax

    def plot_trajectory(self, trajectory: List[int], ax: plt.axes, *args, **kwargs):
        """
        Plots a trajectory in the underlying MDP (if it implements the appropriate plotting function).

        See the plot_trajectory() method of the MDP class for more information.

        Args:
            trajectory (List[int]): List of visited states
            ax (plt.axes): Axes on which to plot the trajectory
        """

        # Convert trajectory of state IDs to
        self.mdp.plot_trajectory(ax=ax, trajectory=trajectory, *args, **kwargs)

    def plot_sequence(self, trajectory: List[int], ax: plt.axes, *args, **kwargs):
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

    def plot_state_values(self, agent_name: str, *args, **kwargs) -> plt.axes:
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

        ax = self.mdp.plot_state_values(
            self.agents[agent_name].algorithm.values, *args, **kwargs
        )

        return ax

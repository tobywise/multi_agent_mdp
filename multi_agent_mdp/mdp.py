from typing import Type, Union, List, Dict
import numpy as np
import warnings
from .grid_utils import get_action_from_states_hex, get_action_from_states_square, grid_coords, hex_adjacency, square_adjacency
from abc import ABCMeta, abstractmethod
from .plotting import plot_grids, plot_trajectory, plot_grid_values, plot_hex_grids, plot_hex_grid_values
import matplotlib.pyplot as plt

# TODO add something to convert MDP SAS to generator for use with MCTS

class MDP():

    def __init__(self, features:np.ndarray, sas:np.ndarray, seed=None):
        """
        Base class representing a Markov Decision Process (MDP) with features. Each state can have multiple features.

        Each state is represented by a unique ID.

        Args:
            features (np.ndarray): Array of features in each state, shape (n_features, n_states)
            sas (np.ndarray): Array representing the probability of transitioning from state S to state S' given action A. Of shape (states, actions, states). 
            seed (int, optional): RNG seed. Defaults to None.
        """


        self.sas = sas
        self.features = features
        self.n_agents = 0

        # Set RNG
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
        else:
            self.rng = np.random
        
    # Use properties to prevent these things from being modified improperly
    @property
    def sas(self):
        return self.__sas 

    @sas.setter
    def sas(self, sas:np.ndarray):

        # TYPE CHECKING
        if not isinstance(sas, np.ndarray):
            raise TypeError("SAS should be a numpy array, got {0}".format(type(sas)))
       
        if  not sas.ndim == 3 or not sas.shape[0] == sas.shape[2]:
            raise TypeError('SAS should be a numpy array of shape (states, actions, states)')

        if np.min(sas) < 0 or np.max(sas) > 1:
            raise AttributeError('SAS should contain probabilities between 0 and 1,'
                                 'min value provided = {0}, max value provided = {1}'.format(np.min(sas), np.max(sas)))

        # If transitions are between 0 and 1 this is not a deterministic MDP
        if np.any((sas > 0) & (sas < 1)):
            self._deterministic = False
        else:
            self._deterministic = True

        # Get adjacency
        self.adjacency = (np.sum(sas, axis=1) > 0).astype(int)
        self.n_states = sas.shape[0]
        self.n_actions = sas.shape[1]
        self.__sas = sas
        
    @property
    def n_features(self):
        return self.__n_features

    @property
    def features(self):
        return self.__feature_array

    @features.setter
    def features(self, features:np.ndarray):

        # Get features
        if not isinstance(features, np.ndarray):
            raise TypeError('Features should be a numpy array of shape (features, states')
        if not features.shape[1] == self.n_states:
            raise AttributeError('Feature array should contain as many states as the adjacency matrix ' +
                                'Expected {0} states, found {1}'.format(self.n_states, features.shape[1]))

        self.__n_features = features.shape[0]
        self.__feature_array = features
        self.__original_feature_array = features.copy()

        # Name features
        self.feature_names = ['Feature_{0}'.format(i) for i in range(self.n_features)]

    def add_agent_feature(self, position:int):
        """
        Add feature representing an agent present in the environment.

        Args:
            positions (int): List of state indices for each agent.
        """

        self.n_agents += 1

        agent_features = np.zeros((self.n_states))
        agent_features[position] = 1

        self.features = np.vstack([self.features, agent_features[None, :]])

    def update_agent_feature(self, agent_idx:int, position:int):
        """
        Updates the feature associated with an agent (i.e. moving the position of the agent)

        Args:
            agent_idx (int): The index of the agent to move.
            positions (int): State index for the agent.
        """

        updated_features = self.features

        updated_features[(self.n_features - self.n_agents) + agent_idx, :] = 0
        updated_features[(self.n_features - self.n_agents) + agent_idx, position] = 1

        self.__feature_array = updated_features      

    def remove_agent_features(self):
        """ Removes agent features """

        self.features = self.features[:-self.n_agents, :]

        self.n_agents = 0


    def consume_features(self, feature_idx:List[int], state_id:int) -> np.ndarray:
        """
        Consume features in a given state. 

        Args:
            feature_idx (list[int]): Feature indices to consume.
            state_id (int): State ID at which to consume the features.
        Returns:
            np.ndarray: 1D array with as many entries as there are features representing the number of 
            each feature consumed.
        """

        consumed = np.zeros(self.n_features, dtype=int)

        for f in feature_idx:
            if self.__feature_array[f, state_id] > 0:
                self.__feature_array[f, state_id] = 0
                consumed[f] += 1

        return consumed
        
    def get_next_state(self, current_state:int, action:int) -> int:
        """
        Gets the next state given the current state and the action taken, based on defined transition probabilities.

        Args:
            current_state (int): Current state
            action (int): Action to be taken.

        Returns:
            int: Next state
        """

        assert action >= 0, 'Actions must be non-negative integers'

        # Get possible transitions given the best action
        possible_states = self.sas[current_state, action, :]

        if self._deterministic:
            next_state = np.argmax(possible_states)
        else:
            next_state = self.rng.random.choice(np.arange(len(possible_states), dtype=np.int), p=possible_states)

        return next_state


    def _trajectory_to_state_action(self, trajectory:list) -> np.ndarray:
        """
        Converts a trajectory of state IDs to an array of state action pairs.

        Args:
            trajectory (list): Trajectory of state IDs

        Returns:
            np.ndarray: A trajectory length X 2 array of state action pairs, 1st entry = state, 2nd entry = action
        """
        
        if not self._deterministic:
            warnings.warn("MDP is non-deterministic, converting trajectory to SA pairs based on most likely action")

        sa_pairs = np.zeros((len(trajectory) - 1, 2))

        for n, state in enumerate(trajectory[:-1]): # Exclude the final state because there is no action taken
            sa_pairs[n, 0] = state
            sa_pairs[n, 1] = self._state_pair_to_action(state, trajectory[n+1])[0]
            sa_pairs[n, 1] = np.argmax(self.sas[state, :, trajectory[n+1]])

        return sa_pairs

    def _state_pair_to_action(self, s1:int, s2:int, return_max:bool=True) -> list:
        """
        Returns actions taking the agent from state 1 to state 2. Can either return the action with the highest transition probability or a list
        of possible actions. States provided must be adjacent.

        Args:
            s1 (int): Start state
            s2 (int): End state
            return_max (bool, optional): Whether to return just the most likely state or all possible states. Defaults to True.

        Raises:
            AttributeError: Does not work if states are not adjacent

        Returns:
            list: List of possible actions, or list with a single entry representing the most likely action.
        """

        possible_actions = self.sas[s1, :, s2]
        if np.max(possible_actions) > 0:
            if return_max:
                return [np.argmax(possible_actions)]
            else:
                return list(np.where(possible_actions > 0)[0])
        else:
            raise AttributeError("States are not adjacent")

    def observation(self):
        pass

    def reset(self):
        self.features = self.__original_feature_array

    # Plotting functions
    def plot(self, colours:list=None, alphas:list=None, *args, **kwargs):
        """ Must be overriden when subclassing to enable plotting methods """
        raise NotImplementedError("This MDP does not implement plotting functions")

    def plot_trajectory(self, ax:plt.axes, trajectory:list, *args, **kwargs):
        """ Must be overriden when subclassing to enable plotting methods """
        raise NotImplementedError("This MDP does not implement plotting functions")

    def plot_state_values(self, values:np.ndarray, *args, **kwargs):
        """ Must be overriden when subclassing to enable plotting methods """
        raise NotImplementedError("This MDP does not implement plotting functions")

    def state_to_position(self):
        """ Must be overriden when subclassing to enable plotting methods """
        raise NotImplementedError("This MDP does not implement plotting functions")


class GridMDP(MDP, metaclass=ABCMeta):

    def __init__(self, features:np.ndarray, shape:tuple=(10, 15), self_transitions:bool=False):
        """
        Creates a deterministic MDP representing a hexagonal grid with a given shape. Uses offset coordinates ("odd-q").

        Self-transitions are optional. If enabled, an additional action is added to the action space (the last available action).

        Args:
            features (np.ndarray): Array of features in each state, shape (n_features, n_states)
            shape (tuple, optional): shape of the grid, width by height. Defaults to (10, 15).
            self_transitions (bool, optional): Whether to allow . Defaults to True.
        """
        
        self.shape = shape
        self.grid = np.zeros(self.shape)
        self.self_transitions = self_transitions

        # Get state IDs and coordinates
        self.state_ids, self.grid_coords = grid_coords(self.grid)

        # Get SAS
        sas = self._get_sas()

        if not isinstance(sas, np.ndarray):
            raise TypeError("SAS must be a numpy array")
        if not sas.ndim == 3:
            raise AttributeError("SAS must have 3 dimensions - state, action, state")
        if not sas.shape[0] == sas.shape[-1]:
            raise AttributeError("Number of states on first dimension must equal number of"
                                 "states in last dimension. Got {0} and {1}".format(sas.shape[0], sas.shape[1]))

        super().__init__(features, sas)

    @abstractmethod
    def _get_sas(self) -> np.ndarray:
        """
        This method must return the state, action, state transition function

        Returns:
            np.ndarray: State, action, state transition function
        """
        return 

    def get_state_coords(self, state:int):

        assert isinstance(state, int), 'State must be an integer'

        return self.grid_coords[state, :]

    def flat_to_grid(self, values:np.ndarray) -> np.ndarray:
        return values.reshape((self.shape), order='F')

    def features_as_grid(self) -> np.ndarray:
        return self.features.reshape(((self.n_features, ) +  self.shape), order='F')

    def state_to_idx(self, state:int):
        return np.unravel_index(state, self.shape)

    def idx_to_state(self, idx):
        return np.ravel_multi_index(idx, self.shape)


    def plot_trajectory(self, trajectory:List[int], ax:plt.axes, colour:str='black', 
                        head_width:int=0.3, head_length:int=0.3, *args, **kwargs) -> plt.axes:
        """
        Plots a trajectory through the MDP.

        Args:
            trajectory (List[int]): List of visited states
            ax (plt.axes): Axes on which to plot, for example those generated by plot().
            colour (str, optional): Colour of the arrows Defaults to 'black'.
            head_width (int, optional): Arrow head width. Defaults to 0.3.
            head_length (int, optional): Arrow head length. Defaults to 0.3.

        Returns:
            plt.axes: Axes
        """
        
        # Convert trajectory of state IDs to 
        position_trajectory = [self.state_to_position(i) for i in trajectory]

        ax = plot_trajectory(ax=ax, trajectory=position_trajectory, colour=colour, head_width=head_width, 
                                head_length=head_length, *args, **kwargs)

        return ax


class SquareGridMDP(GridMDP):

    def __init__(self, features:np.ndarray, shape:tuple=(10, 15), self_transitions:bool=False):
        """
        Creates a deterministic MDP representing a hexagonal grid with a given shape. Uses offset coordinates ("odd-q").

        Args:
            features (np.ndarray): Array of features in each state, shape (n_features, n_states)
            shape (tuple, optional): Shape of the grid, width by height. Defaults to (10, 15).
            self_transitions (bool, optional): Whether to allow . Defaults to True.
        """

        super().__init__(features=features, shape=shape, self_transitions=self_transitions)

    def _get_sas(self):

        n_actions = 4
        if self.self_transitions:
            n_actions += 1

        # Get adjacency matrix
        _, grid_idx = grid_coords(self.grid)
        adjacency = square_adjacency(grid_idx)
        n_states = np.product(self.grid.shape)

        # Loop over state pairs to get state-action-state info
        sas = np.zeros((n_states, n_actions, n_states))

        for i in range(n_states):
            for j in range(n_states):
                if adjacency[i, j] == 1:
                    action = get_action_from_states_square(i, j, adjacency, self.grid_coords)
                    sas[i, action, j] = 1
            if self.self_transitions:
                sas[i, -1, i] = 1

        return sas

    def plot(self, colours:list=None, alphas:list=None, *args, **kwargs) -> plt.axes:
        """
        Plots the MDP, showing each feature.

        Args:
            colours (list, optional): Colours to use for each feature. Defaults to None.
            alphas (list, optional): Alpha value of each feature. Defaults to None.

        Returns:
            plt.axes: Axes
        """

        ax = plot_grids(self.features_as_grid()[:-self.n_agents, :], colours, alphas, *args, **kwargs)

        return ax

    def plot_trajectory(self, trajectory:List[int], ax:plt.axes, colour:str='black', 
                        head_width:int=0.3, head_length:int=0.3, *args, **kwargs) -> plt.axes:
        """
        Plots a trajectory through the MDP.

        Args:
            trajectory (List[int]): List of visited states
            ax (plt.axes): Axes on which to plot, for example those generated by plot().
            colour (str, optional): Colour of the arrows Defaults to 'black'.
            head_width (int, optional): Arrow head width. Defaults to 0.3.
            head_length (int, optional): Arrow head length. Defaults to 0.3.

        Returns:
            plt.axes: Axes
        """
        
        # Convert trajectory of state IDs to 
        position_trajectory = [self.state_to_position(i) for i in trajectory]

        ax = plot_trajectory(ax=ax, trajectory=position_trajectory, colour=colour, head_width=head_width, 
                                head_length=head_length, *args, **kwargs)

        return ax

    def plot_state_values(self, values, cmap='viridis', *args, **kwargs) -> plt.axes:
        """
        Plots continuous values for all states within the MDP.

        Args:
            values ([type]): 1D array of values for each state
            cmap (str, optional): Colourmap to be used for plotting. Defaults to 'viridis'.

        Returns:
            plt.axes: Figure axes
        """

        if not values.ndim == 1:
            raise AttributeError("Values should be provided as a flat, 1D array with one value per state")

        values = self.flat_to_grid(values)
        ax = plot_grid_values(values, cmap=cmap, *args, **kwargs)

        return ax

    def state_to_position(self, state):

        idx = self.state_to_idx(state)
        
        return idx

    
class HexGridMDP(GridMDP):

    def __init__(self, features:np.ndarray, shape:tuple=(10, 15), self_transitions:bool=False):
        """
        Creates a deterministic MDP representing a hexagonal grid with a given shape. Uses offset coordinates ("odd-q").

        Args:
            features (np.ndarray): Array of features in each state, shape (n_features, n_states)
            shape (tuple, optional): Shape of the grid, width by height. Defaults to (10, 15).
            self_transitions (bool, optional): Whether to allow . Defaults to True.
        """

        super().__init__(features=features, shape=shape, self_transitions=self_transitions)

    def _get_sas(self):

        n_actions = 6
        if self.self_transitions:
            n_actions += 1

        # Get adjacency matrix
        _, grid_idx = grid_coords(self.grid)
        adjacency = square_adjacency(grid_idx)
        n_states = np.product(self.grid.shape)

        # Loop over state pairs to get state-action-state info
        sas = np.zeros((n_states, n_actions, n_states))

        for i in range(n_states):
            for j in range(n_states):
                if adjacency[i, j] == 1:
                    action = get_action_from_states_hex(i, j, adjacency, self.grid_coords)
                    sas[i, action, j] = 1
            if self.self_transitions:
                sas[i, -1, i] = 1

        return sas

    def plot(self, colours:list=None, alphas:list=None, *args, **kwargs) -> plt.axes:
        """
        Plots the MDP, showing each feature.

        Args:
            colours (list, optional): Colours to use for each feature. Defaults to None.
            alphas (list, optional): Alpha value of each feature. Defaults to None.

        Returns:
            plt.axes: Axes
        """

        ax, self.coords = plot_hex_grids(self.features_as_grid()[:-self.n_agents, :], colours, alphas, *args, **kwargs)

        return ax

    def state_to_position(self, state):

        idx = self.state_to_idx(state)
        idx = self.coords[:, idx[0], idx[1]]
        
        return idx

    def plot_state_values(self, values, cmap='viridis', *args, **kwargs) -> plt.axes:
        """
        Plots continuous values for all states within the MDP.

        Args:
            values ([type]): 1D array of values for each state
            cmap (str, optional): Colourmap to be used for plotting. Defaults to 'viridis'.

        Returns:
            plt.axes: Figure axes
        """

        if not values.ndim == 1:
            raise AttributeError("Values should be provided as a flat, 1D array with one value per state")

        values = self.flat_to_grid(values)
        ax, self.coords = plot_hex_grid_values(values, cmap=cmap, *args, **kwargs)

        return ax
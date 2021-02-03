from .base import Algorithm
from numba import njit
import numpy as np
from typing import Dict, Tuple, List, Union
from .dynamic_programming import solve_value_iteration
from ..mdp import MDP
import warnings
import time
from operator import itemgetter 

@njit
def rand_choice_nb(arr, p):
    """
    From https://github.com/numba/numba/issues/2539 
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    prob = p / p.sum()
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]

@njit
def softmax(qs, temperature=1):
    return (np.exp(qs / temperature)) / np.sum(np.exp(qs / temperature), axis=0)

@njit
def get_actions_states(sas, current_node):

    # Get available actions from this state and the resulting next states
    actions_states = np.argwhere(sas[current_node, :, :])
    # Get actions
    actions = actions_states[:, 0] 
    # Get resulting states
    states = actions_states[:, 1]

    return actions_states, actions, states


@njit
def get_opponent_next_state(opponent_policy_method:str, states:list, current_state:int, reward_function:np.ndarray, 
                            features:np.ndarray, sas:np.ndarray, action_selection:str='max', softmax_temperature:float=1, 
                            max_iter:int=500, discount:float=0.9, tol:float=1e-4, q_values=None):

    if opponent_policy_method == 'random':
        next_state = np.random.choice(states)

    elif opponent_policy_method == 'solve':

        # Get q values, if not provided
        if q_values is None:
            _, q_values = solve_value_iteration(reward_function, features, max_iter, discount, sas, tol)

        # Get action
        if action_selection == 'max':
            next_state = np.argmax(sas[current_state, np.argmax(q_values[current_state, :]), :])
        elif action_selection == 'softmax':
            action_p = softmax(q_values[current_state, :], temperature=softmax_temperature)
            action = rand_choice_nb(np.arange(len(action_p)), p=action_p)
            next_state = np.argmax(sas[current_state, action, :])
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    return next_state, q_values

@njit
def UCB(V:np.ndarray, N:np.ndarray, states:List[int], C:float, node:int) -> np.ndarray:
    """ Calculates the upper confidence bound (UCB) for available states - in the context of MCTS,
    referred to as UCT (upper confidence bound for trees).

    Args:
        V (np.ndarray): Array of current state value estimates.
        N (np.ndarray): Array of visitation counts for each state.
        states (List[int]): States to calculate UCB for.
        C (float): Exploration parameter (i.e. how much to explore lesser-visited options), higher = more exploration.
        node (int): Current node.

    Returns:
        np.ndarray: UCB for each state
    """

    ucb = (V[states] / (1 + N[states])) + C * np.sqrt((2 * np.log(N[node])) / (1 + N[states]))

    return ucb

@njit
def MCTS_next_node(expand:bool, V:np.ndarray, N:np.ndarray, states:List[int], C, current_node) -> Union[int, bool]:
    """ Selects the next node for MCTS.

    Args:
        expand (bool): Used to determine whether to expand the tree.
        V (np.ndarray): Current estimate of state values.
        N (np.ndarray): State visitation counts.
        states (List[int]): States from which to choose next node.
        C ([type]): UCB exploration parameter, higher = more exploration.
        current_node ([type]): Current node.

    Returns:
        Union[int, bool]: Next node = the next node in the search tree, expand = whether to expand on next step.
    """

    # Check whether we need to expand - if we haven't already expanded all possible next nodes
    if expand and np.any(N[states] == 0):

        # Identify states that haven't been explored yet
        unexplored = states[N[states] == 0]

        # Select one of these at random
        next_node = np.random.choice(unexplored)
    
        # Each step from now on will be a simulation rather than expansion
        expand = False

    # If we've not yet reached a point where we need to expand, pick next state using UCT
    elif expand:
        
        # Calculate UCB (or UCT)
        ucb = UCB(V, N, states, C, current_node)

        # Pick the node with the highest value
        next_node = states[np.argmax(ucb)]

    # From the expanded node we need to simulate the rest of the moves so we chose nodes at random
    else:
        next_node = np.random.choice(states)

    return next_node, expand

@njit
def check_agent_overlap(consumes_agents:Tuple[Tuple[int]], current_node:np.ndarray, caught_cost:float,
                        agents_active:np.ndarray, agent_values:Tuple[float]) -> Union[float, np.ndarray, bool]:
    """ Checks whether agents are in the same state and determines what to do if they are

    Args:
        consumes_agents (Tuple[Tuple[int]]): [description]
        current_node (np.ndarray): [description]
        caught_cost (float): [description]
        agents_active (np.ndarray): [description]
        agent_values (Tuple[float]): [description]

    Returns:
        Union[float, np.ndarray, bool]: Value acquired, whether each agent is active, whether the primary agent got caught
    """

    added_reward = 0.

    consuming_agents, consumed_agents = np.where(consumes_agents)
    
    # Check whether any agents should be eaten
    for n in range(len(consuming_agents)):
        agent1 = consuming_agents[n]
        agent2 = consumed_agents[n]

        # If agents are in the same state
        if current_node[agent1] == current_node[agent2]:  
            # set the agent inactive
            agents_active[agent2] = False
            # If Agent 2 is the primary agent, end the iteration
            if agent2 == 0:
                added_reward += caught_cost
                return added_reward, agents_active, True
            # If Agent 1 is the primary agent, give it a reward for eating Agent 2
            if agent1 == 0:
                added_reward += agent_values[agent2]
    
    return added_reward, agents_active, False


# @njit
def mcts_iteration(interactive:bool, V:np.ndarray, N:np.ndarray, sas:np.ndarray, features:np.ndarray, 
                   current_node:np.ndarray, cached_q_values:List[Dict[Tuple, np.ndarray]], n_moves:Tuple[int], consumes_agents:np.ndarray,
                   consumes_features:np.ndarray, agent_values:Tuple[float], reward_functions:np.ndarray,
                   n_steps:int, C:float, caught_cost:float=-50, opponent_policy_method:str='solve', opponent_action_selection:str='max', 
                   softmax_temperature:float=1, caching:bool=False, VI_max_iter:int=500, VI_discount:float=0.9, 
                   VI_tol:float=1e-4) -> Union[float, List[int]]:
    """
    Runs a single iteration of the MCTS algorithm, optionally including other agents' actions.

    Args:
        interactive (bool): Whether the MDP is interactive (i.e. whether to include other agents' actions during planning)
        V (np.ndarray): Current estimate of state values.
        N (np.ndarray): Current number of visits for each state.
        sas (np.ndarray): MDP transition function.
        features (np.ndarray): MDP feature array.
        current_node (np.ndarray): Starting node for each agent. 
        cached_q_values (List[Dict[Tuple, np.ndarray]]): Cached Q values, used to save recomputing for other agents
        if not necessary.
        n_moves (Tuple[int]): Number of moves per turn for each agent. K
        consumes_agents (np.ndarray): Which agents consume one another. Boolean array of shape (n agents X n agents), where true
        indicates agent X consumes agent Y.
        consumes_features (np.ndarray): Which agents consume which features. Boolean array of shape (n agents X n features), where true
        indicates agent X consumes feature Y.
        agent_values (Tuple[float]): Value of consuming each agent. Keys = agent names, values = value gained from consuming.
        reward_functions (np.ndarray): Reward function for each agent. Shape = (n_agents x n_features)
        n_steps (int): Number of moves made by the primary agent.
        C (float): Exploration parameter.
        caught_cost (float, optional): Cost incurred by the primary agent getting caught. Defaults to -50.
        opponent_policy_method (str, optional): Strategy used to simulate opponents' actions, one of ['random', 'solve']. If 'random', 
        next state is chosen randomly.
        If set to 'solve', value iteration is used to solve the MDP and provide action values. Defaults to 'solve'.
        opponent_action_selection (str, optional): If using 'solve' as the opponent policy method, specifies the action selection method, 
        can be one of ['max', 'softmax']. Defaults to 'max'.
        softmax_temperature (float, optional). Temperature for softamx action selection. Defaults to 1.
        caching (bool, optional): If true, caches calculated Q values. This speeds things up if the VI procedure takes a long time. Defaults to False.
        VI_max_iter (int, optional): Number of iterations for value iteration. Defaults to 500.
        VI_discount (float, optional): Discount factor for value iteration. Defaults to 0.9
        VI_tol (float, optional): Tolerance for value iteration. Defaults to 1e-4.

    Returns:
        Union[float, List[int], List[Dict[Tuple, np.ndarray]]]: Returns the accumulated value from this iteration and the states that were visited,
        along with cached Q values.
    """

    expand = True  # This determines whether we expand or simulate
    accumulated_reward = 0  # Total reward accumulated across all states
    visited_states = []
    current_node = current_node.copy()
    current_actor = 0  # Keep track of who is moving
    features = features.copy()
    n_agents = len(current_node)
    n_features = features.shape[0] - n_agents

    # Keep track of moves made ??
    current_moves = np.zeros(len(current_node))

    # Keep track of which other agents have been eaten
    agents_active = np.ones(len(current_node), dtype=np.bool_)

    # Track which features have been consumed
    consumed_features = [() for _ in range(features.shape[0])]

    # Which features each agent cares about (i.e. where reward function has zero weight)
    agent_preferences = []
    for agent in range(n_agents):
        agent_preferences.append(np.where(reward_functions[agent, :] != 0)[0].tolist())
    
    total_steps = 0

    cache_used = 0

    # Step through until agent has made as many steps as needed
    while total_steps < n_steps:

        # Set corresponding agent feature
        features[current_actor + n_features, :] = 0
        features[current_actor + n_features, current_node[current_actor]] = 1

        # Check whether any agents should be eaten
        if interactive:
            added_reward, agents_active, caught = check_agent_overlap(consumes_agents, current_node, caught_cost, 
                                                                     agents_active, agent_values)
            accumulated_reward += added_reward      
            if caught:
                return accumulated_reward, visited_states, cached_q_values
        
        # Get actions and resulting states from current node for the current actor
        _, _, states = get_actions_states(sas, current_node[current_actor])

        # PRIMARY AGENT'S TURN
        # Get reward available in current state 
        if current_actor == 0:

            # Recalculate rewards - may have changed as features are consumed
            rewards = np.dot(reward_functions[0, :].astype(np.float64), features)

            # Append to list of visited states 
            visited_states.append(current_node[0])

            # Get next node
            current_node[0], expand = MCTS_next_node(expand, V, N, states, C, current_node[0])

            # Add reward to total
            accumulated_reward += rewards[current_node[0]]

            # Count moves
            total_steps += 1
        
        # OTHER AGENT'S TURN
        else:
            if agents_active[current_actor]:
                other_agent_nodes = current_node[np.arange(len(current_node)) != current_actor]
                feature_consumption = tuple(itemgetter(*agent_preferences[current_actor])(consumed_features))

                if caching and (tuple(other_agent_nodes), feature_consumption) in cached_q_values[current_actor]:
                    cache_used += 1
                    current_node[current_actor], q_values = get_opponent_next_state(opponent_policy_method, states, current_node[current_actor],
                                                                                    reward_functions[current_actor, :], features, sas, opponent_action_selection,
                                                                                    softmax_temperature, max_iter=VI_max_iter, 
                                                                                    discount=VI_discount, tol=VI_tol, 
                                                                                    q_values=cached_q_values[current_actor][(tuple(other_agent_nodes), 
                                                                                                                            feature_consumption)])
                else:
                    current_node[current_actor], q_values = get_opponent_next_state(opponent_policy_method, states, current_node[current_actor],
                                                                                    reward_functions[current_actor, :], features, sas, opponent_action_selection,
                                                                                    softmax_temperature, max_iter=VI_max_iter, 
                                                                                    discount=VI_discount, tol=VI_tol, q_values=None)
                    if caching:
                        cached_q_values[current_actor][(tuple(other_agent_nodes), feature_consumption)] = q_values

        # Consume features in this state
        if agents_active[current_actor]:

            # Keep a record of what's been consumed
            if caching:
                for f in np.where(consumes_features[current_actor, :])[0]:
                    # Check there is something to consume
                    if features[consumes_features[current_actor, :], current_node[current_actor]] != 0:
                        consumed_features[f] = consumed_features[f] + (current_node[current_actor], )

            # Remove consumed feature from the feature array
            features[consumes_features[current_actor, :], current_node[current_actor]] = 0


        current_moves[current_actor] += 1

        if interactive:
            # SET WHO MOVES NEXT
            if current_moves[current_actor] == n_moves[current_actor]:
                # Reset moves to 0
                current_moves[current_actor] = 0
                # Change actor
                if current_actor < len(current_moves) - 1:
                    current_actor += 1
                else:
                    current_actor = 0;
    # print('cache used {0} times'.format(cache_used))
    return accumulated_reward, visited_states, cached_q_values

def extract_agent_info(agent_info:Dict[str, Tuple[int, int, List[int], np.ndarray, int]]):

    names = []
    agent_idx = {}
    current_node = []
    n_moves = []
    consumes_features = []
    reward_functions = []

    for n, agent in enumerate(agent_info.keys()):
        this_agent_info = agent_info[agent]
        names.append(agent)
        agent_idx[this_agent_info[4]] = n
        current_node.append(this_agent_info[0])
        n_moves.append(this_agent_info[1])
        consumes = np.zeros_like(this_agent_info[3])
        consumes[this_agent_info[2]] = 1
        consumes_features.append(consumes)
        reward_functions.append(this_agent_info[3])

    reward_functions = np.stack(reward_functions)
    current_node = np.array(current_node)
    n_moves = tuple(n_moves)
    consumes_features = np.stack(consumes_features).astype(bool)

    assert reward_functions.shape[0] == len(current_node), 'Reward function is wrong shape (shape = {0})'.format(reward_functions.shape)

    return agent_idx, current_node, n_moves, consumes_features, reward_functions

def get_agent_values(agents:List[int], agent_idx:Dict[int, str], 
                     primary_agent_reward_function:np.ndarray) -> Dict[str, float]:
    """ Determines the value of each agent to the primary agent """

    agent_values = np.zeros(len(agents))
    for agent in agents:
        if agent != 0:
            for k, v in agent_idx.items():

                if v == agent:
                    agent_values[agent] = primary_agent_reward_function[k]
    
    agent_values = tuple(agent_values)

    return agent_values

def get_agent_consumes(consumes_features:np.ndarray, agent_idx:Dict[int, str]):
    """ Determines which agents consume each other """

    n_agents = len(consumes_features)
    consumes_agents = np.zeros((n_agents, n_agents))

    for agent in range(consumes_features.shape[0]):
        consumes = consumes_features[agent, :]
        for i in np.where(consumes)[0]:
            if i in agent_idx:
                consumes_agents[agent, agent_idx[i]] = 1
    
    consumes_agents = consumes_agents.astype(bool)

    return consumes_agents

def extract_all_agent_info(agent_info:Dict[str, Tuple[int, int, List[int], np.ndarray, int]], 
                           opponent_info: Dict[str, Tuple[int, int, List[int], np.ndarray, int]]):

    # Extract agent info
    agent_idx, current_node, n_moves, consumes_features, reward_functions = extract_agent_info({**agent_info, **opponent_info})

    # Get value of consuming each agent
    agent_values = get_agent_values(range(len(current_node)), agent_idx, reward_functions[0, :])
            
    # Determine which agent eats which other agent
    consumes_agents = get_agent_consumes(consumes_features, agent_idx)

    return current_node, n_moves, consumes_features,\
           reward_functions, agent_values, consumes_agents


# @njit(parallel=False)
def run_mcts(interactive:bool, n_iter:int, features:np.ndarray, sas:np.ndarray, 
             agent_info:Dict[str, Tuple[int, int, List[int], np.ndarray, int]],
             opponent_info:Dict[str, Tuple[int, int, List[int], np.ndarray, int]], 
             n_steps:int, C:float, caught_cost:float=-50, opponent_policy_method:str='solve', 
             opponent_action_selection:str='max', softmax_temperature:float=1, cache:bool=False, 
             cached_q_values:Dict=None, VI_kwargs:Dict={}) -> Union[np.ndarray, np.ndarray]:
    """     
    Runs the MCTS algorithm to determine the best action to take from the current state.

    The search algorithm incorporates other agents' behaviour in its simulations. Other agents' moves are simulated
    either ranndomly or by solving the MDP using value iteration. 

    This is also accounts for agents eating one another, allowing predator-prey dynamics to be simulated.

    Only implemented for deterministic MDPs.

    Args:
        interactive (bool): If true, the tree search simulates other agents' actions. 
        n_iter (int): Number of iterations to run
        features (np.ndarray): MDP feature array.
        sas (np.ndarray): MDP transition function.
        agent_info (Dict[str, Tuple[int, int, List[int], np.ndarray, int]]): Information about the primary agent
        (i.e. the agent we are planning for). This is given as a single entry dictionary with its key representing the name of the 
        agent, and its value a tuple of the form (start state, n moves per turn, features to consume, reward function, agent index).
        opponent_info (Dict[str, Tuple[int, int, List[int], np.ndarray, int]]): Information about the other agents in the environment.
        This is given as a dictionary with keys representing the name of the agent, and values a tuple of the form 
        (start state, n moves per turn, features to consume, reward function, agent index).
        n_steps (int): Number of steps to run (i.e number of turns the primary agent should take).
        C (float): Exploration parameter.
        caught_cost (float, optional): Cost of the agent getting caught. Defaults to -50.

    Returns:
        Union[np.ndarray, np.ndarray]: Returns the value of each state and the number of times each state has been visited.
    """

    current_node, n_moves, consumes_features, reward_functions, agent_values, consumes_agents = extract_all_agent_info(agent_info, opponent_info)

    # Remove agent features
    features = features.copy()

    V = np.zeros(sas.shape[0])  # Value of each node
    N = np.zeros(sas.shape[0]) # Times each node visited

    # Cached Q values - used to same time solving MDP for other agents
    if cached_q_values is None:
        cached_q_values = [{} for _ in range(len(current_node))]

    # Run MCTS
    for _ in range(n_iter):
        # print('Iteration {0}'.format(i))
        accumulated_reward, visited_states, cached_q_values = mcts_iteration(interactive, V, N, sas, features, current_node, cached_q_values,
                                                                            n_moves, consumes_agents, consumes_features, agent_values,
                                                                            reward_functions, n_steps, C, caught_cost, opponent_policy_method, 
                                                                            opponent_action_selection, softmax_temperature, cache, **VI_kwargs)

        # Backpropogate
        for v in visited_states:
            V[v] += accumulated_reward
            N[v] += 1

    return V, N, cached_q_values

# @njit
def get_MCTS_action_values(node:int, sas:np.ndarray, V:np.ndarray, N:np.ndarray):

        _, actions, states = get_actions_states(sas, node)

        action_values = V[states] / (1 + N[states])

        return actions, action_values, states

@njit
def solve_all_value_iteration(sas, predator_reward_function, features, prey_index, max_iter=None, tol=None, discount=None):

    all_q_values = np.zeros((sas.shape[0], sas.shape[0], sas.shape[1]))  # Prey idx X opponent_states X actions

    for prey_state in range(features.shape[1]):

        # Set prey feature according to the current prey location
        features[prey_index, :] = 0
        features[prey_index, prey_state] = 1

        # Do value iteration
        _, all_q_values[prey_state, ...] = solve_value_iteration(sas.shape[0], sas.shape[1], predator_reward_function, features, max_iter, discount, sas, tol)

    return all_q_values


class MCTS(Algorithm):

    def __init__(self, n_iter:int=1000, n_steps:int=30, C:float=1, caught_cost:float=-50, 
                 interactive:bool=False, opponent_policy_method:str='solve', 
                 opponent_action_selection:str='max', softmax_temperature:float=1, caching:bool=False, reset_cache:bool=False,
                 VI_kwargs:Dict={}):

        self.n_iter = n_iter
        self.n_steps = n_steps
        self.C = C
        self.caught_cost = caught_cost
        self.interactive = interactive
        self.opponent_policy_method = opponent_policy_method
        self.opponent_action_selection = opponent_action_selection
        self.softmax_temperature = softmax_temperature
        self.caching = caching
        self.reset_cache = reset_cache
        self.cache = None
        self.VI_kwargs = VI_kwargs

        super().__init__()

    def _reset(self):

        self.V = None
        self.N = None

    def clear_cache(self):
        self.cache = None

    def create_agent_info(self, reward_function:np.ndarray, interactive:bool, mdp:MDP) -> Union[Dict, Dict]:
        """ Creates information about agents in the environment to be used by the MCTS algorithm.

        Args:
            reward_function (np.ndarray): Primary agent reward function.
            interactive (bool): If true, provides information on other agents.
            mdp (MDP): The MDP being used

        Returns:
            Union[Dict, Dict]: Returns info for the primary agent and all opponents.
        """

        primary_agent_name = self._agent.name
        
        agent_info = {primary_agent_name: (self._agent.position, self._agent.n_moves, self._agent.consumes, 
                                          reward_function, self._agent.agent_idx + (mdp.n_features - self._environment.n_agents))}

        opponent_info = {}

        if interactive:
            for agent_name, agent in self._environment.agents.items():
                if not agent_name == self._agent.name:
                    opponent_info[agent_name] = (agent.position, agent.n_moves, agent.consumes, 
                                                agent.reward_function, agent.agent_idx + (mdp.n_features - self._environment.n_agents))

        return agent_info, opponent_info

    def _fit(self, mdp, reward_function, position):

        # This requires an environment as it needs info about other agents
        if self._environment is None and self.interactive:
            raise NotImplementedError("Interactive MCTS algorithm must be used in the context of an interactive environment")
        
        # Reset things
        self.reset()

        if self.reset_cache:
            self.cache = None

        # Get information about primary and other agents
        agent_info, opponent_info = self.create_agent_info(reward_function, self.interactive, mdp)

        # Run MCTS
        self.V, self.N, self.cache = run_mcts(self.interactive, self.n_iter, mdp.features.copy(), mdp.sas,
                                 agent_info, opponent_info, self.n_steps, self.C, self.caught_cost,
                                 self.opponent_policy_method, self.opponent_action_selection, self.softmax_temperature, self.caching,
                                 self.cache, self.VI_kwargs)

        # Get the value of available actions
        actions, action_values, _ = get_MCTS_action_values(position, mdp.sas, self.V, self.N)

        if np.all(action_values == 0):
            warnings.warn("All action values are zero. The tree search may not be deep enough to reach distant rewards.")

        # Make this look like q values from other algorithms
        q_values = np.zeros((mdp.n_states, mdp.n_actions)) * np.nan

        q_values[position, actions] = action_values

        return self.V, q_values





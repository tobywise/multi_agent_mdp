from .environments import Environment
from .mdp import HexGridMDP
from .agents import Agent
from .algorithms.dynamic_programming import ValueIteration
from .algorithms.action_selection import MaxActionSelector, SoftmaxActionSelector
import numpy as np
from typing import List, Dict
import json
import warnings

# Functions to convert environments to dictionary/JSON representations

def hex_env_to_dict(env:Environment, feature_names:List=None, format:str='index') -> Dict:
    """
    Generates a representation of an environment using a hexagonal grid as a dictionary.

    Args:
        env (Environment): The environment.
        feature_names (List): Names of features. Defaults to None.
        format (str, optional): How to represent agent positions, can be 'coords' or 'index'. Defaults to 'index'.

    Returns:
        Dict: Returns a dictionary specifying shape of the grid, agent positions and properties, MDP properties
    """

    if not (env.mdp.n_features - len(env.agents)) == len(feature_names):
        raise AttributeError('Number of feature names should equal number of features, excluding agents')

    env_dict = dict()

    env_dict['size'] = list(env.mdp.shape)
    env_dict['features'] = {}

    for f in range(env.mdp.n_features - len(env.agents)):
        # print(f, feature_names[f])
        if format == 'index':
            env_dict['features'][feature_names[f]] = np.where(env.mdp.features[f, :])[0].astype(int).tolist()
        elif format == 'coords':
            env_dict['features'][feature_names[f]] = np.stack(env.mdp.state_to_idx(np.where(env.mdp.features[f, :]))).T.squeeze().astype(int).tolist()

    if format == 'index':
        env_dict['walls'] = np.where(env.mdp.walls)[0].astype(int).tolist()
    elif format == 'coords':
        env_dict['walls'] = np.stack(env.mdp.state_to_idx(np.where(env.mdp.walls))).T.squeeze().astype(int).tolist()

    env_dict['type'] = 'hex'
    env_dict['self_transitions'] = env.mdp.self_transitions

    env_dict['agents'] = []

    for agent_name, agent in env.agents.items():
        agent_dict = {agent_name: {}}
        agent_dict[agent_name] = {}
        if format == 'index':
            agent_dict[agent_name]['position'] = agent.position
        elif format == 'coords':
            agent_dict[agent_name]['position']  = [[int(i) for i in list(env.mdp.state_to_idx(agent.position))]]

        agent_dict[agent_name]['n_moves'] = agent.n_moves

        agent_dict[agent_name]['reward_weights'] = [int(i) for i in list(agent.reward_weights)]
        agent_dict[agent_name]['consumes'] = agent.consumes

        agent_dict[agent_name]['algorithm'] = agent.algorithm.name
        agent_dict[agent_name]['algorithm_kwargs'] = agent.algorithm_kwargs
        agent_dict[agent_name]['action_selector'] = agent.action_selector.name
        agent_dict[agent_name]['action_kwargs'] = agent.action_kwargs
        env_dict['agents'].append(agent_dict)

    return env_dict

def hex_env_to_json(env:Environment, feature_names:List, return_string=False, fname=None) -> str:
    """
    Produces a JSON representation of a hexagonal grid environment.

    Args:
        env (Environment): The environment.
        feature_names (List): Names of features
        return_string (bool, optional): Whether to return the JSON string. Defaults to False.
        fname ([type], optional): If a filename is provided, the JSON will be saved to this file. Defaults to None.

    Returns:
        Str: Optionally returns a JSON-format string.
    """

    json_dict = hex_env_to_dict(env, feature_names)

    if fname is not None:
        with open(fname, 'w') as f:
            json.dump(json_dict, f)

    if return_string:
        json_string = json.dumps(json_dict)
        return json_string


def hex_environment_from_dict(env_dict, feature_names=[]):

    if not env_dict['type'] == 'hex':
        raise NotImplementedError()

    else:
        n_features = len(env_dict['features'])
        features = np.zeros((n_features, np.product(env_dict['size'])))
        for n, f in enumerate(feature_names):
            # print("CREATING ENV", n, f)
            states = env_dict['features'][f]
            features[n, states] = 1

        walls = np.zeros(np.product(env_dict['size']))
        walls[env_dict['walls']] = 1

        newMDP = HexGridMDP(features=features, walls=walls, shape=tuple(env_dict['size']), 
                            self_transitions=env_dict['self_transitions'])

        agent_dict = {}

        # This makes sure predator is first in the list of agents
        for agent in env_dict['agents']:
        # for agent_name in env_dict['agents'].keys():
            agent_name = list(agent.keys())[0]
            agent_info = agent[agent_name]

            if agent_info['algorithm'] == 'value_iteration':
                algorithm = ValueIteration
            else:
                raise NotImplementedError

            if agent_info['action_selector'] == 'max':
                action_selector = MaxActionSelector
            elif agent_info['action_selector'] == 'softmax':
                action_selector = SoftmaxActionSelector
            else:
                raise NotImplementedError

            new_agent = Agent(agent_name, n_moves=agent_info['n_moves'], algorithm=algorithm, algorithm_kwargs=agent_info['algorithm_kwargs'],
                              action_selector=action_selector, action_kwargs=agent_info['action_kwargs'])
            
            agent_dict[new_agent] = (agent_info['position'], agent_info['reward_weights'], agent_info['consumes'])
        
        newEnvironment = Environment(newMDP, agent_dict)

        return newEnvironment

def hex_environment_from_json(json_string):
    env_dict = json.loads(json_string)

    return hex_environment_from_dict(env_dict)
from ..algorithms.base import Algorithm
from ..algorithms.action_selection import MaxActionSelector, SoftmaxActionSelector
from .dynamic_programming import ValueIteration, solve_value_iteration
from .mcts import MCTS_next_node, get_actions_states
from ..mdp import MDP
import numpy as np
from numba import njit
from fastprogress import progress_bar
import warnings
from abc import abstractmethod, abstractproperty, ABCMeta
from typing import List, Union
from scipy.optimize import minimize

@njit
def zscore_nb(x):
    xnew = np.zeros_like(x)
    for i in range(x.shape[1]):
        x_i = x[:, i]
        x_i[np.isinf(x_i)] = np.nan
        xnew[:, i] = (x_i - np.nanmean(x_i)) / np.nanstd(x_i)
    return xnew

@njit
def demean(x):
    xnew = np.zeros_like(x)
    for i in range(x.shape[1]):
        xnew[:, i] = x[:, i] - x[:, i].mean()
    return xnew

@njit
def state_visitation_iterator(sas:np.ndarray, n_iter:int, pi:np.ndarray, p_zero:np.ndarray) -> np.ndarray:
    """
    Determines state visitation counts based on a given policy (Algorithm 9.3 in Ziebart's thesis).

    Note: This assumes a deterministic policy (i.e. a single action is chosen in each state with 100% probability)

    Args:
        sas (np.ndarray): State, action, state transition function for the MDP
        n_iter (int): Number of iterations to run
        pi_ (np.ndarray): Policy, 2D array representing probability of taking action A in state S. 
        p_zero (np.ndarray): Probability of starting in a given state (1D array with length equal to number of states)

    Returns:
        np.ndarray: State visitation frequency
    """
    n_states = sas.shape[0]
    n_actions = sas.shape[1]

    D_ = np.zeros(n_states)

    # Checking for convergence
    count = 0
    # Run state visitation algorithm
    while count < n_iter:
        Dprime = p_zero.copy()
        for s in range(n_states):
            for a in range(n_actions):
                for s_prime in range(n_states):
                    Dprime[s_prime] = Dprime[s_prime] + D_[s] * pi[s, a] * sas[s, a, s_prime]
        count += 1
        D_ = Dprime.copy()
    
    return D_

class BaseIRL(metaclass=ABCMeta):

    @abstractmethod
    def fit(self, mdp:MDP, trajectories:List[int], **kwargs):

        pass


class MaxCausalEntIRL(BaseIRL):
    """ 
    Maximum (Causal) Entropy inverse reinforcement learning
    """

    def __init__(self, learning_rate:float=0.3, decay:int=1, max_iter_irl:int=50, theta:np.ndarray=None, 
                 tol:float=1e-8, VI_discount:float=0.9, VI_max_iter:int=500, soft:bool=True):
        """Uses a variant of Maximum Entropy inverse reinforcement learning (MaxEnt IRL) to infer the reward function 
        of an agent based on trajectories within a given MDP. This uses value iteration to solve the MDP 
        on behalf of the agent that is the target of this inference.

        The original Maximum Entropy algorithm (Ziebart et al, 2008) uses a "soft" version of value iteration as it 
        incorporates a softmax function. By setting the value supplied to the `soft` argument to `True`, the algorithm
        will use this "soft" value iteration approach. Alternatively, by setting it to `False` it will use a max policy 
        instead, which means it is identical to standard value iteration (https://apps.dtic.mil/sti/pdfs/AD1090741.pdf), 
        in line with the Maximum Causal Entropy approach (Ziebart et al, 2010).

        The algorithm here is intended for infinite time horizon MDPs (as opposed to those with fixed terminal states), 
        and so technically is an implementation of Maximum Discounted (Causal) Entropy (Bloem & Bambos, 2014). The original
        MaxEnt and MaxCausalEnt approaches assume finite time horizon MDPs. The only difference (to my knowledge) is that
        the value iteration step in the MDCE approach uses discounted future rewards (wuth a discount factor that must) 
        be set.

        Args:
            learning_rate (float, optional): Learning rate for MCE-IRL. Defaults to 0.3.
            decay (int, optional): Learning rate decay. If set to 0, learning rate does not decay. If set to any positive number
            the learning rate decays on each trial according to learning rate * n^-decay.
            max_iter_irl (int, optional): Number of MCE-IRL iterations to run. Defaults to 20.
            theta (np.ndarray, optional): Initial guess at reward function, if None this is set to 1 for each feature. 
            Defaults to None.
            tol (float, optional): Tolerance for convergence - shared across MCE-IRL and underlying value iteration. 
            Defaults to 1e-8.
            VI_discount (float, optional): Value iteration discount factor. Defaults to 0.9.
            VI_max_iter (int, optional): Maximum number of VI iterations. Defaults to 500.
            soft (bool, optional): Whether to use "soft" value iteration (i.e. incorporating a softmax). 
            If false, uses standard value iteration.
        """

        # Settings for value iteration
        self.discount = VI_discount
        self.tol = tol
        self.max_iter = VI_max_iter
        self.soft = soft

        # Settings for IRL
        assert learning_rate > 0, 'Learning rate must be greater than zero'
        assert decay >= 0, 'Decay must be positive'
        self.learning_rate = learning_rate
        self.original_learning_rate = learning_rate
        self.learning_rate_decay = decay
        self.max_iter_irl = max_iter_irl
        self.original_theta = np.array(theta).copy()
        self.theta = theta

        # This is used for feature count estimation
        if soft:
            self.action_selector = SoftmaxActionSelector()
        else:
            self.action_selector = MaxActionSelector()


    def _get_state_visitation_frequencies(self, mdp, trajectories, pi, n_iter):

        # Probability of starting in a given state - estimated from trajectories
        p_zero = np.zeros(mdp.n_states)
        for t in trajectories:
            p_zero[t[0]] += 1
        p_zero /= np.sum(p_zero)

        # Get state visitation frequencies
        self.D_ = state_visitation_iterator(mdp.sas, n_iter, pi, p_zero)

    def _solve_value_iteration(self, reward_function, features, max_iter, discount, sas, tol, soft):

        # Solve using value iteration
        values, q_values = solve_value_iteration(reward_function, features, max_iter, discount, sas, tol, soft)

        return values, q_values

    def _maxcausalent_innerloop(self, mdp, theta, trajectories):

        # Solve MDP using value iteration
        _, q_values = self._solve_value_iteration(theta, mdp.features, self.max_iter, self.discount, mdp.sas, self.tol, self.soft)
        self.q_values = q_values


        # Get policy
        pi = self.action_selector.get_pi_p(q_values)

        # Get state visitation frequencies
        self._get_state_visitation_frequencies(mdp, trajectories, pi, self.max_iter)

        # Get feature counts
        deltaF = (mdp.features * self.D_).sum(axis=1).astype(float)

        return deltaF
            
    def _solve_maxcausalent(self, mdp, trajectories, ignore_features=(), show_progress=True):
        
        # Get observed state visitation counts
        visited_states = np.zeros(mdp.n_states)

        for t in trajectories:
            for s in t:
                visited_states[s] += 1

        # Get observed feature visitation counts
        true_F = (mdp.features * visited_states).sum(axis=1).astype(float)

        # Remove ignored features
        true_F[ignore_features] = 0

        # Initial guess at reward function
        if self.theta is None:
            self.theta = np.ones(mdp.n_features) * 0

        self.theta[ignore_features] = 0

        # Max ent loop
        self.error_history = []

        if show_progress:
            pb = progress_bar(range(self.max_iter_irl))
        else:
            pb = range(self.max_iter_irl)

        for i in pb:
            
            # Get feature visitation counts
            deltaF = self._maxcausalent_innerloop(mdp, self.theta, trajectories)
            deltaF[ignore_features] = 0

            # Prediction error
            error = (true_F - deltaF) #* 0.001 # Scaling helps avoid overflow
            self.error_history.append(error)

            # Increment reward function
            if self.learning_rate_decay == 0:
                self.theta += self.learning_rate * error
            else:
                self.learning_rate = (self.learning_rate * np.power(i+1, -float(self.learning_rate_decay)))
                self.theta += (self.learning_rate * np.power(i+1, -float(self.learning_rate_decay))) * error

            # Update progress bar
            mean_abs_error = np.mean(np.abs(error))
            if show_progress:
                pb.comment = '| Error = {0}'.format(mean_abs_error)
            
            if mean_abs_error < self.tol:
                print("Converged after {0} iterations".format(i))
                break

        if not np.mean(error) < self.tol:
            warnings.warn('Solver did not converge', Warning)

    def reset(self):
        """ Resets the reward function and learning rate to its initial state """
        self.theta = self.original_theta
        self.learning_rate = self.original_learning_rate

    def reset_learning_rate(self):
        """ Resets the learning rate """
        self.learning_rate = self.original_learning_rate

    def fit(self, mdp:Union[MDP, List[MDP]], trajectories:List[List[int]], ignore_features:tuple=(), show_progress:bool=True) -> np.ndarray:
        """
        Uses Maximum Causal Entropy IRL to infer the reward function of an agent based on supplied
        trajectories within a given MDP.

        Args:
            mdp (MDP): MDP in which the agent is acting
            trajectories (list): List of lists of visited states - i.e. a list of trajectories, with each trajectory
            being a list of visited states.
            ignore_features (tuple, optional): Features to ignore. Defaults to ().
            show_progress (bool, optional): If true, show a progress bar representing fitting progress.

        Returns:
            np.ndarray: Inferred reward function
        """

        if not len(trajectories[0]):
            raise TypeError("Trajectories must be a list of lists - i.e. a list of trajectories, with each trajectory \
                            representing a list of states")

        self._solve_maxcausalent(mdp, trajectories, ignore_features=ignore_features, show_progress=show_progress)

        return self.theta

class MFFeatureMatching(BaseIRL):
    """
    Infers a reward function based on feature counts alone.
    """

    def __init__(self):
        super().__init__()

    def _solve_irl(self, theta:np.ndarray, phi:np.ndarray, ignore_features:List[int]):
        theta[ignore_features] = 0
        return -np.dot(theta, phi.T).sum()

    def fit(self, mdp:Union[MDP, List[MDP]], trajectories:List[List[int]], ignore_features:List[int]=[]):

        if not len(trajectories[0]):
            raise TypeError("Trajectories must be a list of lists - i.e. a list of trajectories, with each trajectory \
                            representing a list of states")

        if isinstance(mdp, list):
            if not len(trajectories) == len(mdp):
                raise AttributeError("Length of trajectory list ({0}) and MDP list ({1}) do not match".format(len(trajectories), len(mdp)))
        else:
            mdp = [mdp] * len(trajectories)  # Repeat the MDP to match the number of trajectories

        feature_array = np.zeros((len(trajectories), mdp[0].n_features))

        for n, t in enumerate(trajectories):
            feature_array[n, :] = mdp[n].features[:, t[1:]].sum(axis=1)  # Exclude the first state of the trajectory as it's the start state which wasn't chosen

        res = minimize(self._solve_irl, np.ones(mdp[0].n_features), (feature_array, ignore_features))

        self.theta = res.x

        return self.theta


class HyptestIRL(BaseIRL):
    """ 
    Hypothesis-testing inverse reinforcement learning (HT-IRL)
    """

    def __init__(self, learning_rate:float=0.3, decay:int=1, theta:np.ndarray=None, 
                 tol:float=1e-8, VI_discount:float=0.9, VI_max_iter:int=500, soft:bool=True, normalisation:str='relative'):
        """
        Infers an agent's reward function by testing discrete hypotheses regarding their preference for different
        features.

        Args:
            learning_rate (float, optional): Learning rate for HT-IRL. Defaults to 0.3.
            decay (int, optional): Learning rate decay. If set to 0, learning rate does not decay. If set to any positive number
            the learning rate decays on each trial according to learning rate * n^-decay.
            theta (np.ndarray, optional): Initial guess at reward function, if None this is set to 1 for each feature. 
            Defaults to None.
            tol (float, optional): Tolerance for convergence - shared across MCE-IRL and underlying value iteration. 
            Defaults to 1e-8.
            VI_discount (float, optional): Value iteration discount factor. Defaults to 0.9.
            VI_max_iter (int, optional): Maximum number of VI iterations. Defaults to 500.
            soft (bool, optional): Whether to use "soft" value iteration (i.e. incorporating a softmax). 
            If false, uses standard value iteration. Defaults to True.
            normalisation (str, optional): Method to use for normalising values across hypotheses. Can be one of 'relative' or 
            'z-score'. Defaults to 'relative'.
        """

        # Settings for value iteration
        self.discount = VI_discount
        self.tol = tol
        self.max_iter = VI_max_iter
        self.soft = soft

        # Settings for IRL
        assert learning_rate > 0, 'Learning rate must be greater than zero'
        assert decay >= 0, 'Decay must be positive'
        self.learning_rate = learning_rate
        self.learning_rate_decay = decay
        self.original_theta = np.array(theta).copy()
        self.theta = theta
        self.normalisation = normalisation

    def _single_move(self, start:int, next_state:int, sas:np.ndarray, q_values:Union[np.ndarray, None], 
                    features:np.ndarray, exclude_features:List=[]) -> Union[np.ndarray, np.ndarray]:
        """
        Infers reward weights from a single move (represented by a pair of adjacent states).

        Args:
            start (int): Start state.
            next_state (int): Next state.
            sas (np.ndarray): State-action-state transition matrix.
            q_values (Union[np.ndarray, None]): Q values if already estimated. If None, Q values are estimated using
            value iteration.
            features (np.ndarray): Features in each state.
            exclude_features (List, optional): Features to ignore. Defaults to [].

        Returns:
            Union[np.ndarray, np.ndarray]: Returns estimated reward weights and Q values.
        """

        # Get the action that was taken
        try:
            next_action = np.argwhere(sas[start, :, next_state])[0][0]
        except Exception as e:
            raise ValueError('State {0} and state {1} are not adjacent'.format(start, next_state))

        # Estimate Q values if not provided
        if q_values is None:

            V = np.zeros_like(features.T)
            Q = np.zeros(sas.shape[:2] + (features.shape[0], )) * np.nan

            feature_idx = range(features.shape[0])
            feature_idx = [i for i in feature_idx if not i in exclude_features]

            for f in feature_idx:
                reward_weights = np.zeros(features.shape[0])
                reward_weights[f] = 1
                V[:, f], Q[..., f] = solve_value_iteration(reward_weights, features, max_iter=self.max_iter, 
                                                          discount=self.discount, sas=sas, tol=self.tol, soft=self.soft)
        else:
            Q = q_values

        # Get Q values for the current state
        state_Q = Q[start, ...].copy()

        # Get legal actions in this state
        _, actions, _ = get_actions_states(sas, start)

        # Z score the Q value according to each hypothesis within each action
        if self.normalisation == 'z-score':
            state_Q[actions, :] = zscore_nb(state_Q[actions, :])
        elif self.normalisation == 'relative':
            raise NotImplementedError

        # Get the Q values according to each hypothesis given the action taken
        reward_weights = state_Q[next_action, :]  

        return reward_weights, Q
        
    def _solve_hyptest(self, mdp:List[MDP], trajectories:List[List[int]], ignore_features:List[int]=[], show_progress:bool=True):
        """
        Infers reward weights from a list of trajectories, optionally within different MDPs.

        Args:
            trajectories (List[List[int]]): A list of trajectories, each represented by a list of visited states.
            mdp (List[MDP]): A list of MPDs, one per trajectory.
            ignore_features (List[int], optional): Features to ignore. Defaults to [].
            show_progress (bool, optional): If true, shows a progress bar. Defaults to True.

        """
        
        if not len(trajectories) == len(mdp):
            raise ValueError("Must provide same number of trajectories and MDPs, got {0} and {1}".format(len(trajectories), len(mdp)))

        # Get SAS and features
        sas = [i.sas for i in mdp]
        features = [i.features for i in mdp]

        assert all([i.shape == sas[0].shape for i in sas])
        assert all([i.shape == features[0].shape for i in features])

        # Initialise reward weight estimate and learning rate
        if self.theta is None:
            R_ = np.zeros(features[0].shape[0])
        else:
            R_ = self.theta
        learning_rate = np.ones(features[0].shape[0]) * self.learning_rate
        Rs = []
        observed_Rs = []

        # Progress bar
        if show_progress:
            t = progress_bar(range(len(trajectories)))
        else:
            t = range(len(trajectories))

        # Loop over trajectories
        n_states = 1  # Used for adjusting learning rate

        for n in t:
            trajectory = trajectories[n]

            # Reset Q values
            q_values = None
            
            # Loop over state pairs within trajectory
            for s in range(len(trajectory) - 1):
                
                # Estimate reward function and return estimated Q values
                R, q_values = self._single_move(trajectory[s], trajectory[s+1], sas[n], q_values, features[n], 
                                                ignore_features)

                # Error
                delta = R - R_

                # Adjust learning rate
                if self.learning_rate_decay > 0:
                    R_ += (learning_rate * np.power(n_states, -float(self.learning_rate_decay))) * delta

                # No learning rate adjustment
                else:
                    R_ += learning_rate * delta

                # Keep track of all the reward functions we've estimated
                Rs.append(R_.copy())

                n_states += 1

            # Keep track of reward functions at the end of each trajectory
            observed_Rs.append(R_.copy())

        Rs = np.stack(Rs)
        observed_Rs = np.stack(observed_Rs)

        self.theta = R_  # Final estimate of reward weights
        self.theta_array = Rs  # Estimated reward weights at each step
        self.trajectory_theta = observed_Rs  # Estimated reward weights after each trajectory

    def reset(self):
        """ Resets the reward function to its initial state """
        self.theta = self.original_theta

    def fit(self, mdp:Union[MDP, List[MDP]], trajectories:List[List[int]], ignore_features:tuple=(), show_progress:bool=True) -> np.ndarray:
        """
        Uses hypothesis testing IRL to infer the reward function of an agent based on supplied
        trajectories within a given MDP.

        Args:
            mdp (MDP or List[MDP]): MDP in which the agent is acting. Can also be a list of MDPs, one for each trajectory.
            trajectories (list): List of lists of visited states - i.e. a list of trajectories, with each trajectory
            being a list of visited states.
            ignore_features (tuple, optional): Features to ignore. Defaults to ().
            show_progress (bool, optional): If true, show a progress bar representing fitting progress.

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

        self._solve_hyptest(mdp, trajectories, ignore_features=ignore_features, show_progress=show_progress)

        return self.theta


def state_pair_to_action(mdp, s1, s2):

    if np.max(mdp.sas[s1, :, s2]) > 0:
        return np.argmax(mdp.sas[s1, :, s2])
    else:
        raise AttributeError("States are not adjacent")
    

def get_eyeline_features(mdp, current_state, action):

    complete = False
    states = []

    while not complete:
        states.append(current_state)
        sa_next_states = mdp.sas[current_state, action, :]
        if np.max(sa_next_states) > 0:
            current_state = np.argmax(sa_next_states)
        else:
            complete = True
    observed_features = mdp.features[:, states].sum(axis=1)
    return observed_features

def get_trajectory_eyeline_features(mdp, trajectories, normalise=True):

    feature_counts = np.zeros(mdp.n_features)

    for trajectory in trajectories:

        for n, state in enumerate(trajectory[:-1]):
            action = state_pair_to_action(mdp, state, trajectory[n+1])
            feature_counts += get_eyeline_features(mdp, state, action)

    if normalise and not np.all(feature_counts == 0):
        feature_counts /= feature_counts.sum()

    return feature_counts

def get_trajectory_eyeline_features_diff(mdp, trajectories, normalise=True):

    chosen_feature_counts = np.zeros(mdp.n_features)
    unchosen_feature_counts = np.zeros(mdp.n_features)

    for trajectory in trajectories:

        for n, state in enumerate(trajectory[:-1]):
            chosen_action = state_pair_to_action(mdp, state, trajectory[n+1])
            for action in range(mdp.n_actions-1):
                if action == chosen_action:
                    chosen_feature_counts += get_eyeline_features(mdp, state, action)
                else:
                    unchosen_feature_counts += get_eyeline_features(mdp, state, action)

    if normalise:
        if not np.all(chosen_feature_counts == 0):
            chosen_feature_counts /= chosen_feature_counts.sum()
        if not np.all(unchosen_feature_counts == 0):
            unchosen_feature_counts /= unchosen_feature_counts.sum()

    feature_counts = chosen_feature_counts - unchosen_feature_counts

    return feature_counts

class SimpleActionIRL():

    def __init__(self):
        self.theta = None

    def fit(self, mdp, trajectories, reset=True):

        if reset or self.theta is None:
            self.theta = np.zeros(mdp[0].n_features)

        for n, m in enumerate(mdp):
            observed_feature_counts = get_trajectory_eyeline_features(m, [trajectories[n]])
            self.theta += observed_feature_counts

        return self.theta


class SimpleActionFeatureDiffIRL():

    def __init__(self):
        self.theta = None

    def fit(self, mdp, trajectories, reset=True):
        
        if reset or self.theta is None:
            self.theta = np.zeros(mdp[0].n_features)

        for n, m in enumerate(mdp):
            observed_feature_counts = get_trajectory_eyeline_features_diff(m, [trajectories[n]])

            self.theta += observed_feature_counts
        
        return self.theta


        
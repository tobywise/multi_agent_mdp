from multi_agent_mdp.algorithms.action_selection import MaxActionSelector
from .dynamic_programming import ValueIteration, solve_value_iteration
from ..mdp import MDP
import numpy as np
from numba import njit
from fastprogress import progress_bar
import warnings

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

class MaxCausalEntIRL():
    """ 
    Maximum Causal Entropy inverse reinforcement learning
    """

    def __init__(self, learning_rate:float=0.3, decay:int=1, max_iter_irl:int=50, theta:np.ndarray=None, 
                 tol:float=1e-8, VI_discount:float=0.9, VI_max_iter:int=500):
        """Uses maximum causal entropy inverse reinforcement learning (MCE-IRL) to infer the reward function 
        of an agent based on trajectories within a given MDP. This uses value iteration to solve the MDP 
        on behalf of the agent that is the target of this inference.

        The original maximum causal entropy algorithm uses a "soft" version of value iteration as it 
        incorporates a softmax function. Here we're assuming a max policy which means it's identical to 
        value iteration (https://apps.dtic.mil/sti/pdfs/AD1090741.pdf).

        Args:
            learning_rate (float, optional): Learning rate for MCE-IRL. Defaults to 0.3.
            decay (int, optional): Learning rate decay. If set to 0, learning rate does not decay. If set to any positive number

            max_iter_irl (int, optional): Number of MCE-IRL iterations to run. Defaults to 20.
            theta (np.ndarray, optional): Initial guess at reward function, if None this is set to 1 for each feature. 
            Defaults to None.
            tol (float, optional): Tolerance for convergence - shared across MCE-IRL and underlying value iteration. 
            Defaults to 1e-8.
            VI_discount (float, optional): Value iteration discount factor. Defaults to 0.9.
            VI_max_iter (int, optional): Maximum number of VI iterations. Defaults to 500.
        """

        # Settings for value iteration
        self.discount = VI_discount
        self.tol = tol
        self.max_iter = VI_max_iter

        # Settings for IRL
        assert learning_rate > 0, 'Learning rate must be greater than zero'
        assert decay >= 0, 'Decay must be positive'
        self.learning_rate = learning_rate
        self.learning_rate_decay = decay
        self.max_iter_irl = max_iter_irl
        self.original_theta = theta
        self.theta = theta
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
        _, q_values = self._solve_value_iteration(theta, mdp.features, self.max_iter, self.discount, mdp.sas, self.tol, True)

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

        learning_rate = self.learning_rate

        if show_progress:
            pb = progress_bar(range(self.max_iter_irl))
        else:
            pb = range(self.max_iter_irl)

        for i in pb:
            
            # Get feature visitation counts
            deltaF = self._maxcausalent_innerloop(mdp, self.theta, trajectories)
            deltaF[ignore_features] = 0

            # Prediction error
            error = true_F - deltaF
            self.error_history.append(error)

            # Increment reward function
            if self.learning_rate_decay == 0:
                self.theta += self.learning_rate * error
            else:
                self.theta += (learning_rate / ((i+1) * self.learning_rate_decay)) * error

            # Update progress bar
            mean_abs_error = np.mean(np.abs(error))
            pb.comment = '| Error = {0}'.format(mean_abs_error)
            
            if mean_abs_error < self.tol:
                print("Converged after {0} iterations".format(i))
                break

        if not np.mean(error) < self.tol:
            warnings.warn('Solver did not converge')

    def reset(self):
        """ Resets the reward function to its initial state """
        self.theta = self.original_theta

    def fit(self, mdp:MDP, trajectories:list, ignore_features:tuple=(), show_progress:bool=True) -> np.ndarray:
        """
        Uses Maximum Causal Entropy IRL to infer the reward function of an agent based on supplied
        trajectories within a given MDP.

        Args:
            mdp (MDP): MDP in which the agent is acting.
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
import numpy as np
from .base import MBTransitionAlgorithm, MFTransitionAlgorithm
from ..mdp import MDP, Trajectory


class FullExploration(MBTransitionAlgorithm):
    """
    Takes every action in every state to estimate the transition matrix.
    """

    def __init__(self, n_iter:int=1, seed:int=123):
        """
        Takes every action in every state to estimate the transition matrix.

        Args:
            n_iter (int, optional): Number of iterations to run. For a fully deterministic MDP, a single iteration will 
            capture the full transition dynamics. For probabilistic transitions, more iterations will be required. Defaults to 1.
        """
        
        self.n_iter = n_iter
        self.rng = np.random.RandomState(seed)

        super().__init__()

    
    def _fit(self, mdp:MDP, position:int, n_steps:int):

        est_sas = np.zeros_like(mdp.sas)

        for _ in range(self.n_iter):

            for s in range(mdp.n_states):  # loop through states
                for a in range(mdp.n_actions):  # loop through actions
                    next_states = np.argwhere(mdp.sas[s, a, :])
                    if len(next_states) > 1:
                        # if multiple possible states, select at according to transition probability
                        next_state = self.rng.choice(next_states, p=mdp.sas[s, a, next_states])  
                    elif len(next_states) == 1:
                        next_state = next_states[0]
                    else:
                        next_state = []
                    est_sas[s, a, next_state] += 1

        # Normalise within action
        est_sas = est_sas / est_sas.sum(axis=1)[:, None, :]

        # Remove nans due to dividing by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            est_sas[np.isnan(est_sas)] = 0

        assert ~np.any(np.isnan(est_sas)), 'NaN in estimated transitions'
        assert ~np.any(np.isinf(est_sas)), 'Inf in estimated transitions'

        return est_sas
                
class TransitionLearner(MFTransitionAlgorithm):
    """
    Learns state transitions based on experience.
    """

    def __init__(self, learning_rate:float=0.9):

        self.learning_rate = learning_rate
        
        super().__init__()


    def _fit(self, trajectory:Trajectory, initial_guess:np.ndarray=None) -> np.ndarray:
        
        if initial_guess is None:
            est_sas = np.zeros_like(trajectory.mdp.sas)
        else:
            est_sas = initial_guess.copy()

        if not est_sas.shape == trajectory.mdp.sas.shape:
            raise AttributeError("Initial SAS array shape ({0}) is not the same as expected ({1})".format(est_sas.shape, trajectory.mdp.sas.shape))

        # Loop through observations
        for obs in trajectory:

            # Create a vector that is zero everywhere except the visited state
            next_states = np.zeros_like(est_sas[obs.state_1, obs.action, :])
            next_states[obs.state_2] = 1

            # Calculate prediction error
            pe = next_states - est_sas[obs.state_1, obs.action, :]

            # Update transition matrix
            est_sas[obs.state_1, obs.action, :] += self.learning_rate * pe

        return est_sas
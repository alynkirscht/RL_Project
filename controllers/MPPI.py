import numpy as np
from scipy.interpolate import CubicSpline
from mujoco import rollout

class MPPI():

    def __init__(self, config, model_sim, data_sim):

        self.horizon, self.lambda_, self.n_samples = config['horizon'], config['temperature'], config['n_samples'] 
        self.model_sim = model_sim
        self.data_sim = data_sim

    def sample_delta_u(self):
        if self.sample_type == 'normal':
            size = (self.n_samples, self.horizon, self.act_dim)
            return self.generate_noise(size)
        elif self.sample_type == 'cubic':
            indices = np.arange(self.n_knots)*self.horizon//self.n_knots
            size = (self.n_samples, self.n_knots, self.act_dim)
            knot_points = self.generate_noise(size)
            cubic_spline = CubicSpline(indices, knot_points, axis=1)
            return cubic_spline(np.arange(self.horizon))
        
    def rollout(self, curr_state, action, next_state):
        rollout.rollout(self.model, self.thread_local.data, skip_checks=True,
                        nroll=next_state.shape[0], nstep=next_state.shape[1],
                        initial_state=curr_state, control=action, state=next_state)

    
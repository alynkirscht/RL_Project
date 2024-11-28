import numpy as np
from scipy.interpolate import CubicSpline
from mujoco import rollout
from utils.transforms import batch_world_to_local_velocity, calculate_orientation_quaternion
from utils.tasks import TASKS
import mujoco

class MPPI():

    def __init__(self, config, model_sim, data_sim):

        self.horizon, self.lambda_, self.n_samples = config['horizon'], config['temperature'], config['n_samples'] 
        self.model_sim = model_sim
        self.data_sim = data_sim

        self.goal_pos = TASKS[config['task']]['goal_pos']
        self.goal_ori = TASKS[config['task']]['default_orientation'] 
        self.cmd_vel = TASKS[config['task']]['cmd_vel']

        self.control_dim = 12
        self.control_max = np.array([0.863, 4.501, -0.888] * 4)
        self.control_min = np.array([-0.863, -0.686, -2.818] * 4)

        self.goal_index = 0
        self.body_ref = np.concatenate((self.goal_pos[self.goal_index],
                                        self.goal_ori[self.goal_index],
                                        self.cmd_vel[self.goal_index],
                                        np.zeros(4)))

    def cost_function(self, states, actions, joints_ref, body_ref):
        num_samples = states.shape[0]
        num_pairs = states.shape[1]

        # Repeat body reference for all samples and time steps
        traj_body_ref = np.repeat(body_ref[np.newaxis, :], num_samples * num_pairs, axis=0)

        # Flatten states and actions for batch processing
        states = states.reshape(-1, states.shape[2])
        actions = actions.reshape(-1, actions.shape[2])

        # Repeat and reshape joint references for batch processing
        joints_ref = joints_ref.T
        joints_ref = np.tile(joints_ref, (num_samples, 1, 1))
        joints_ref = joints_ref.reshape(-1, joints_ref.shape[2])

        # Concatenate body and joint references for full reference state
        x_ref = np.concatenate(
            [traj_body_ref[:, :7], joints_ref[:, :12], traj_body_ref[:, 7:], joints_ref[:, 12:]],
            axis=1
        )

        # Rotate velocity vectors to the local frame
        rotated_ref = batch_world_to_local_velocity(states[:, 3:7], states[:, 19:22])
        states[:, 19:22] = rotated_ref

        # Compute cost for each rollout
        costs = self.quadruped_cost_np(states, actions, x_ref)

        # Sum costs across time steps for each sample
        total_costs = costs.reshape(num_samples, num_pairs).sum(axis=1)
        return total_costs
        
    def update(self, model_sim, data_sim):
        control_sequence = np.zeros((self.horizon, self.control_dim))

        sampled_controls = np.random.uniform(
            low=self.control_min, high=self.control_max, size=(self.n_samples, self.horizon, self.control_dim)
        )

        curr_state = np.concatenate((data_sim.qpos[:], data_sim.qvel[:]))

        # Simulate trajectories and calculate costs
        costs = np.zeros(self.n_samples)
        for i in range(self.n_samples):
            temp_state = curr_state.copy()
            for t in range(self.horizon):
                # Apply sampled control
                control = sampled_controls[i, t]
                # next_state, _, _, _, _ = env.step(action)
                mujoco.mj_step(model_sim, data_sim)
                mujoco.mj_forward(model_sim, data_sim)
                next_state = np.concatenate((data_sim.qpos[:], data_sim.qvel[:]))
                
                # Compute cost
                costs[i] += self.cost_function(temp_state, control, next_state)
                temp_state = next_state

        # Compute trajectory weights
        costs -= np.min(costs)  # Normalize costs to avoid overflow
        weights = np.exp(-costs / self.lambda_)
        weights /= np.sum(weights)  # Normalize weights

        # Update the control sequence using weighted averaging
        for t in range(self.horizon):
            control_sequence[t] = np.sum(
                sampled_controls[:, t] * weights[:, None], axis=0
            )

        # Use the first action in the sequence
        return control_sequence[0]


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
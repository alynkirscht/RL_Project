
import gymnasium as gym
import numpy as np
# env = gymnasium.make('Ant-v5', xml_file='./mujoco_menagerie/unitree_go1/scene.xml')

env = gym.make(
    'Ant-v5',
    xml_file='./mujoco_menagerie/unitree_go1/scene.xml',
    forward_reward_weight=1,
    ctrl_cost_weight=0.05,
    contact_cost_weight=5e-4,
    healthy_reward=2,
    main_body=1,
    healthy_z_range=(0.195, 0.75),
    include_cfrc_ext_in_observation=True,
    exclude_current_positions_from_observation=False,
    reset_noise_scale=0.1,
    frame_skip=25,
    max_episode_steps=1000,
    render_mode="human"
)

# Hyperparameters
horizon = 20  # Planning horizon (number of timesteps)
num_samples = 100  # Number of trajectories to sample
lambda_ = 1.0  # Temperature parameter for MPPI
control_dim = env.action_space.shape[0]  # Dimensionality of the control input
action_low, action_high = env.action_space.low, env.action_space.high  # Action bounds

# Initialize control sequence
control_sequence = np.zeros((horizon, control_dim))

# Define a cost function
def cost_function(state, action, next_state):
    """
    Define the cost function for the MPPI controller.
    - Encourage forward progress.
    - Penalize control effort and instability.
    """
    forward_reward = state[0]  # Assuming x-position is state[0]
    ctrl_cost = np.sum(action**2)  # Quadratic control cost
    # healthy_penalty = 0 if env.healthy_reward else 1  # Penalize unhealthy states
    return -forward_reward + 0.05 * ctrl_cost# + healthy_penalty

# MPPI algorithm
def mppi_controller(state):
    global control_sequence

    # Generate random control sequences
    sampled_controls = np.random.uniform(
        low=action_low, high=action_high, size=(num_samples, horizon, control_dim)
    )

    # Simulate trajectories and calculate costs
    costs = np.zeros(num_samples)
    for i in range(num_samples):
        temp_state = state.copy()
        for t in range(horizon):
            # Apply sampled control
            action = sampled_controls[i, t]
            next_state, _, _, _, _ = env.step(action)
            
            # Compute cost
            costs[i] += cost_function(temp_state, action, next_state)
            temp_state = next_state

    # Compute trajectory weights
    costs -= np.min(costs)  # Normalize costs to avoid overflow
    weights = np.exp(-costs / lambda_)
    weights /= np.sum(weights)  # Normalize weights

    # Update the control sequence using weighted averaging
    for t in range(horizon):
        control_sequence[t] = np.sum(
            sampled_controls[:, t] * weights[:, None], axis=0
        )

    # Use the first action in the sequence
    return control_sequence[0]

# # Reset the environment
obs, info = env.reset()

# Run a simulation loop
for _ in range(500):  # Run for 500 timesteps
    # action = env.action_space.sample()  # Random action
    action = mppi_controller(obs)  # Compute the action using MPPI
    import pdb; pdb.set_trace()
    obs, reward, done, truncated, info = env.step(action)  # Take a step
    env.render()  # Render the simulation
    if done or truncated:
        env.reset()  # Reset environment if the episode ends

# # Close the environment when done
env.close()


import gymnasium as gym
import numpy as np
# env = gymnasium.make('Ant-v5', xml_file='./mujoco_menagerie/unitree_go1/scene.xml')

env = gym.make(
    'Ant-v5',
    xml_file='./mujoco_menagerie/unitree_go1/scene.xml',
    forward_reward_weight=1,
    ctrl_cost_weight=0.05,
    contact_cost_weight=5e-4,
    healthy_reward=1,
    main_body=1,
    healthy_z_range=(0.195, 0.75),
    include_cfrc_ext_in_observation=True,
    exclude_current_positions_from_observation=False,
    reset_noise_scale=0.1,
    frame_skip=25,
    max_episode_steps=1000,
    render_mode="human"
)

# # Reset the environment
obs, info = env.reset()

# Run a simulation loop
for _ in range(500):  # Run for 500 timesteps
    action = env.action_space.sample()  # Random action
    obs, reward, done, truncated, info = env.step(action)  # Take a step
    env.render()  # Render the simulation
    if done or truncated:
        env.reset()  # Reset environment if the episode ends

# # Close the environment when done
env.close()

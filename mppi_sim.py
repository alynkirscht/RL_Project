# Imports
import numpy as np
import copy as cp

# Mujoco
import mujoco
import mujoco_viewer

# Controller functions
from controllers.MPPI import MPPI

# from utils.tasks import get_task
# from utils.transforms import batch_world_to_local_velocity

# Visualization
from matplotlib.animation import FuncAnimation
# from IPython.display import HTML
import matplotlib.pyplot as plt

def main():
    T = 100000 # 20 seconds
    TIMECONST = 0.02
    VIEWER = True
    task = 'walk_octagon'

    SIMULATION_STEP = 0.01 #0.002
    CTRL_UPDATE_RATE = 100
    CTRL_HORIZON = 40
    CTRL_LAMBDA = 0.1
    CTRL_N_SAMPLES = 30

    task_data = get_task(task)
    sim_path = task_data["sim_path"]

    path = "models/go1/"
    model_path = "go1_scene_mppi.xml"

    model_sim = mujoco.MjModel.from_xml_path(path+model_path)
    dt_sim = 0.01
    model_sim.opt.timestep = dt_sim
    data_sim = mujoco.MjData(model_sim)
    viewer = mujoco_viewer.MujocoViewer(model_sim, data_sim, 'offscreen')

    mujoco.mj_resetDataKeyframe(model_sim, data_sim, 0) # stand position
    mujoco.mj_forward(model_sim, data_sim)
    q_init = cp.deepcopy(data_sim.qpos) # save reference pose
    v_init = cp.deepcopy(data_sim.qvel) # save reference pose
    print("Configuration: {}".format(q_init)) # save reference pose

    img = viewer.read_pixels()
    plt.imshow(img)

    controller = MPPI(config, model_sim, data_sim)
    controller.internal_ref = True
    controller.reset_planner()

    agent = MPPI(task=task)
    agent.set_params(horizon=CTRL_HORIZON, lambda_=CTRL_LAMBDA, N=CTRL_N_SAMPLES)
    simulator = Simulator(agent=agent, viewer=VIEWER, T=T, dt=SIMULATION_STEP, timeconst=TIMECONST,
                          model_path=sim_path, ctrl_rate=CTRL_UPDATE_RATE)
    
    simulator.run()
    simulator.plot_trajectory()
    pass

if __name__ == "__main__":
    main()
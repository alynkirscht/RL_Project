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
import matplotlib.pyplot as plt

def main():
    config = {
        'task': 'stand',
        'horizon': 40,
        'n_samples': 30,
        'temperature': 0.1
    }

    model_path = './models/go1/go1_scene_mppi.xml'
    model_sim = mujoco.MjModel.from_xml_path(model_path)
    dt_sim = 0.01
    model_sim.opt.timestep = dt_sim
    data_sim = mujoco.MjData(model_sim)
    viewer = mujoco_viewer.MujocoViewer(model_sim, data_sim, 'offscreen')

    mujoco.mj_resetDataKeyframe(model_sim, data_sim, 0) # stand position
    mujoco.mj_forward(model_sim, data_sim)
    q_init = cp.deepcopy(data_sim.qpos) # initial pos from sim
    v_init = cp.deepcopy(data_sim.qvel) # initial vel from sim

    img = viewer.read_pixels()
    plt.imshow(img)
    plt.show()

    controller = MPPI(config, model_sim, data_sim)

    # Run simulation
    anim_imgs = []
    sim_inputs = []
    x_states = []

    tfinal = 8 # 14 for stairs, 30 for walk_octagon
    tvec = np.linspace(0,tfinal,int(np.ceil(tfinal/dt_sim))+1)

    for ticks, ti in enumerate(tvec):
        q_curr = cp.deepcopy(data_sim.qpos) # save reference pose
        v_curr = cp.deepcopy(data_sim.qvel) # save reference pose
        x = np.concatenate([q_curr, v_curr])
        
        if ticks%1 == 0:
            u_joints = controller.update(model_sim, data_sim)  
            
        data_sim.ctrl[:] = u_joints
        mujoco.mj_step(model_sim, data_sim)
        mujoco.mj_forward(model_sim, data_sim)

        error = np.linalg.norm(np.array(controller.body_ref[:3]) - np.array(data_sim.qpos[:3]))

        viewer.add_marker(
            pos=controller.body_ref[:3]*1,         # Position of the marker
            size=[0.15, 0.15, 0.15],     # Size of the sphere
            rgba=[1, 0, 1, 1],           # Color of the sphere (red)
            type=mujoco.mjtGeom.mjGEOM_SPHERE, # Specify that this is a sphere
            label=""
        )

        if error < controller.goal_thresh[controller.goal_index]:
            controller.next_goal()
        
        img = viewer.read_pixels()
        if ticks % 2 == 0:
            anim_imgs.append(img)
        sim_inputs.append(u_joints)
        x_states.append(x)

if __name__ == "__main__":
    main()
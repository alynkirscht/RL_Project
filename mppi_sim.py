# Imports
import numpy as np
import copy as cp

# Mujoco
import mujoco
import mujoco_viewer

# Controller functions
# from control.controllers.mppi_locomotion import MPPI

# from utils.tasks import get_task
# from utils.transforms import batch_world_to_local_velocity

# Visualization
from matplotlib.animation import FuncAnimation
# from IPython.display import HTML
import matplotlib.pyplot as plt

path = "models/go1/"
model_path = "go1_scene_mppi.xml"

model_sim = mujoco.MjModel.from_xml_path(path+model_path)
dt_sim = 0.01
model_sim.opt.timestep = dt_sim
data_sim = mujoco.MjData(model_sim)
viewer = mujoco_viewer.MujocoViewer(model_sim, data_sim, 'offscreen')
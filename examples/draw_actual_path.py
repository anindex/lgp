import pickle
import sys
import argparse
import numpy as np
import pandas as pd
from os.path import join, dirname, abspath, expanduser
from ast import literal_eval as make_tuple
import matplotlib.pyplot as plt
from pyrieef.geometry.workspace import SignedDistanceWorkspaceMap
from pyrieef.geometry.pixel_map import sdf

ROOT_DIR = join(dirname(abspath(__file__)), '..')
DATA_DIR = join(ROOT_DIR, 'data', 'experiments')
MODEL_DIR = join(expanduser("~"), '.qibullet', '1.4.3')
DATASET_DIR = join(ROOT_DIR, 'datasets', 'mogaze')
sys.path.append(ROOT_DIR)
sys.path.append(join(ROOT_DIR, "../humoro"))
robot_model_file = join(MODEL_DIR, 'pepper.urdf')

from lgp.utils.helpers import draw_numpy_trajectory
from lgp.geometry.workspace import HumoroWorkspace
from examples.prediction.hmp_interface import HumanRollout

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python process_data.py data.p')
parser.add_argument('--name', help='The scenario name of the domain, problem file', type=str, default='ground_truth.p')
parser.add_argument('-s', help='The scenario name of the domain, problem file', type=str, default="(\'p7_3\', 29439, 33249)")
args = parser.parse_args()
data_file = join(DATA_DIR, args.name)

segment = make_tuple(args.s)
with open(data_file, 'rb') as f:
    data = pickle.load(f)

hr = HumanRollout(path_to_mogaze=DATASET_DIR)
ws = HumoroWorkspace(hr, robot_model_file=robot_model_file)
ws.initialize_workspace_from_humoro(segment=segment, objects=[])
meshgrid = ws.box.stacked_meshgrid(100)
sdf_map = np.asarray(SignedDistanceWorkspaceMap(ws)(meshgrid))
sdf_map = (sdf_map < 0).astype(float)
signed_dist_field = np.asarray(sdf(sdf_map))
signed_dist_field = np.flip(signed_dist_field, axis=0)
signed_dist_field = np.interp(signed_dist_field, (signed_dist_field.min(), signed_dist_field.max()), (0, max(ws.box.dim)))

fig = plt.figure(figsize=(8, 8))
extents = ws.box.box_extent()
ax = fig.add_subplot(111)
im = ax.imshow(signed_dist_field, cmap='inferno', interpolation='nearest', extent=extents)
fig.colorbar(im)
single_traj = data[segment]['single_actual_path']
dynamic_traj = data[segment]['dynamic_actual_path']
human_traj = data[segment]['human_path']
draw_numpy_trajectory(ax, single_traj, 'green')
draw_numpy_trajectory(ax, dynamic_traj, 'blue')
draw_numpy_trajectory(ax, human_traj, 'red')
plt.show()
import numpy as np
import yaml
import matplotlib.patches as patches


DRAW_MAP = {
    'human': patches.Circle,
    'robot': patches.Circle,
    'box_obj': patches.Rectangle,
    'point_obj': patches.Circle,
}


def frozenset_of_tuples(data):
    return frozenset([tuple(t) for t in data])


def draw_trajectory(ax, traj, color, linewidth=1.0):
    '''
    This only support drawing 2D trajectory
    '''
    assert traj.n() == 2
    x_idx, y_idx = 2 * np.arange(traj.T() + 2), 2 * np.arange(traj.T() + 2) + 1
    ax.plot(traj.x()[x_idx], traj.x()[y_idx], color=color, linewidth=linewidth)


def draw_numpy_trajectory(ax, traj, color, linewidth=1.0):
    '''
    This only support drawing 2D trajectory
    '''
    x, y = [p[0] for p in traj], [p[1] for p in traj]
    ax.plot(x, y, color=color, linewidth=linewidth)


def load_yaml_config(config_file):
        with open(config_file, 'r') as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)
        return config

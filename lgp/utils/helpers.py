import numpy as np
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

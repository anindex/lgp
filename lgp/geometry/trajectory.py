import numpy as np
from pyrieef.motion.trajectory import Trajectory


def linear_interpolation_waypoints_trajectory(waypoints):
    q, T = waypoints[0][0], waypoints[-1][1]
    trajectory = Trajectory(T, q.size)
    t_total = 0
    for i in range(len(waypoints) - 1):
        q_init, q_goal, t_segment = waypoints[i][0], waypoints[i + 1][0], waypoints[i + 1][1] - waypoints[i][1]
        if i == 0 or i == len(waypoints) - 2:  # account for init and goal
            t_segment += 1
        for t in range(t_segment):
            alpha = float(t) / float(t_segment)
            trajectory.configuration(t + t_total)[:] = (1 - alpha) * q_init + alpha * q_goal
        t_total += t_segment
    return trajectory


def compute_path_length(path):
    if path is None or len(path) <= 1:
        return 0.
    length = 0.
    for i in range(len(path) - 1):
        length += np.linalg.norm(path[i + 1] - path[i])
    return length
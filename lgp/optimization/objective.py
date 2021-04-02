import logging
import numpy as np

from pyrieef.geometry.workspace import Circle, Box, Workspace


# temporary importing until complication of install is resolve
import os
import sys
_path_file = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_path_file, "../../../bewego"))
from pybewego import PlanarOptimizer
from pybewego.workspace_viewer_server import WorkspaceViewerServer


class TrajectoryConstraintObjective:
    logger = logging.getLogger(__name__)

    def __init__(self, **kwargs):
        self.verbose = kwargs.get('verbose', False)
        self.T = kwargs.get('T', 0)   # time steps
        self.dt = kwargs.get('dt', 0.1)  # sample rate
        self.n = kwargs.get('n', 2)
        # set parameters
        self.set_parameters(**kwargs)
        self.objective = None
        # ipopt options
        self.ipopt_options = {
            'tol': kwargs.get('tol', 9e-3),
            'acceptable_tol': kwargs.get('tol', 1e-2),
            'acceptable_constr_viol_tol': kwargs.get('acceptable_constr_viol_tol', 5e-1),
            'constr_viol_tol': kwargs.get('constr_viol_tol', 2e-2),
            'max_iter': kwargs.get('max_iter', 200),
            # 'bound_relax_factor': kwargs.get('bound_relax_factor', 0),
            'obj_scaling_factor': kwargs.get('obj_scaling_factor', 1e+2)
        }
        # viewer
        self.enable_viewer = kwargs.get('enable_viewer', False)
        self.delay_viewer = kwargs.get('delay_viewer', 200)

    def set_parameters(self, **kwargs):
        self.workspace = kwargs.get('workspace', None)
        self.trajectory = kwargs.get('trajectory', None)
        if self.trajectory is not None:
            self._q_init = self.trajectory.initial_configuration()
            self._q_goal = self.trajectory.final_configuration()
            self.T = self.trajectory.T()
            self.n = self.trajectory.n()
        self.waypoints = kwargs.get('waypoints', None)
        self.waypoint_manifolds = kwargs.get('waypoint_manifolds', None)
        self.goal_manifold = kwargs.get('goal_manifold', None)
        self.s_velocity_norm = kwargs.get('s_velocity_norm', 0)
        self.s_acceleration_norm = kwargs.get('s_acceleration_norm', 20)
        self.s_obstacles = kwargs.get('s_obstacles', 1e+3)
        self.s_obstacle_alpha = kwargs.get('s_obstacle_alpha', 7)
        self.s_obstacle_gamma = kwargs.get('s_obstacle_gamma', 60)
        self.s_obstacle_margin = kwargs.get('s_obstacle_margin', 0.)
        self.s_obstacle_constraint = kwargs.get('s_obstacle_constraint', 1)
        self.with_smooth_obstacle_constraint = kwargs.get('with_smooth_obstacle_constraint', True)
        self.s_terminal_potential = kwargs.get('s_terminal_potential', 1e+4)
        self.with_goal_constraint = kwargs.get('with_goal_constraint', True)
        self.with_goal_manifold = kwargs.get('with_goal_manifold', True)
        self.s_waypoint_constraint = kwargs.get('s_waypoint_constraint', 1e+4)
        self.with_waypoint_constraint = kwargs.get('with_waypoint_constraint', True)

    def set_problem(self, **kwargs):
        self.set_parameters(**kwargs)
        if self.workspace is None:
            TrajectoryConstraintObjective.logger.error('Workspace is not defined! Cannot set optimization problem.')
            return
        if self.trajectory is None:
            TrajectoryConstraintObjective.logger.error('Init trajectory is not defined! Cannot set optimization problem.')
            return
        self.problem = PlanarOptimizer(self.T, self.dt, self.workspace.box.box_extent())
        # Add workspace obstacles
        for o in self.workspace.obstacles:
            if isinstance(o, Circle):
                self.problem.add_sphere(o.origin, o.radius)
            elif isinstance(o, Box):
                self.problem.add_box(o.origin, o.dim)
            else:
                TrajectoryConstraintObjective.logger.warn('Shape {} not supported by bewego'.format(type(o)))
        # terms
        if self.s_velocity_norm > 0:
            self.problem.add_smoothness_terms(1, self.s_velocity_norm)
        if self.s_acceleration_norm > 0:
            self.problem.add_smoothness_terms(2, self.s_acceleration_norm)
        if self.s_obstacles > 0:
            self.problem.add_obstacle_terms(
                self.s_obstacles,
                self.s_obstacle_alpha,
                self.s_obstacle_margin)
        if self.s_terminal_potential > 0:
            if self.with_goal_constraint:
                if self.with_goal_manifold:
                    self.problem.add_goal_manifold_constraint(self.goal_manifold.origin, self.goal_manifold.radius, self.s_terminal_potential)
                else:
                    self.problem.add_goal_constraint(self.q_goal, self.s_terminal_potential)
            else:
                self.problem.add_terminal_potential_terms(self.q_goal, self.s_terminal_potential)
        if self.s_waypoint_constraint > 0:
            if self.waypoints is not None:  # waypoints take precedent
                for i in range(1, len(self.waypoints) - 1):
                    if self.with_waypoint_constraint:
                        self.problem.add_waypoint_constraint(*self.waypoints[i], self.s_waypoint_constraint)
                    else:
                        self.problem.add_waypoint_terms(*self.waypoints[i], self.s_waypoint_constraint)
            elif self.waypoint_manifolds is not None:
                for i in range(len(self.waypoint_manifolds) - 1):
                    self.problem.add_waypoint_manifold_constraint(self.waypoint_manifolds[i][0].origin, self.waypoint_manifolds[i][1], self.waypoint_manifolds[i][0].radius, self.s_waypoint_constraint)
        if self.s_obstacle_constraint > 0:
            if self.with_smooth_obstacle_constraint:
                self.problem.add_smooth_keypoints_surface_constraints(self.s_obstacle_margin, self.s_obstacle_gamma, self.s_obstacle_constraint)
            else:
                self.problem.add_keypoints_surface_constraints(self.s_obstacle_margin, self.s_obstacle_constraint)
        self.objective = self.problem.objective(self.q_init)
        if self.enable_viewer:
            self.obstacle_potential = self.problem.obstacle_potential()
            self.problem.set_trajectory_publisher(False, self.delay_viewer)

    def cost(self, trajectory=None):
        '''
        Should call set problem first
        '''
        if self.objective is not None:
            if trajectory is None:
                trajectory = self.trajectory
            return self.objective.forward(trajectory.active_segment())[0]
        else:
            return 0.

    def optimize(self, status=None, traj=None, ipopt_options=None):
        if ipopt_options is None:
            ipopt_options = self.ipopt_options
        res = self.problem.optimize(
            self.trajectory.x(),
            self.q_goal,
            ipopt_options
        )
        self.trajectory.active_segment()[:] = res.x
        if self.verbose:
            TrajectoryConstraintObjective.logger.info('Gradient norm : %f' % np.linalg.norm(res.jac))
        if status is not None:  # get out status from multiprocessing
            status.value = res.success
        if traj is not None:
            traj[:] = self.trajectory.x().tolist()
        return res.success, self.trajectory

    @property
    def q_init(self):
        return self._q_init

    @q_init.setter
    def q_init(self, value):
        self._q_init = value

    @property
    def q_goal(self):
        return self._q_goal

    @q_goal.setter
    def q_goal(self, value):
        self._q_goal = value

import logging
import numpy as np
from scipy import optimize

from pyrieef.geometry.workspace import Circle, Box
from pybewego import MotionObjective


class TrajectoryConstraintObjective:
    logger = logging.getLogger(__name__)

    def __init__(self, **kwargs):
        self.verbose = kwargs.get('verbose', False)
        self.config_space_dim = kwargs.get('config_space_dim', 2)
        self.T = kwargs.get('T', 20)   # time steps
        self.dt = kwargs.get('dt', 0.1)  # sample rate
        self.trajectory_space_dim = (self.config_space_dim * (self.T + 2))
        # set parameters
        self.set_parameters(**kwargs)
        self.objective = None

    def set_parameters(self, **kwargs):
        self.workspace = kwargs.get('workspace', None)
        self.trajectory = kwargs.get('trajectory', None)
        if self.trajectory is not None:
            self._q_init = self.trajectory.initial_configuration()
            self._q_goal = self.trajectory.final_configuration()
            self.T = self.trajectory.T()
            self.config_space_dim = self.trajectory.n()
            self.trajectory_space_dim = (self.config_space_dim * (self.T + 2))
        self.waypoints = kwargs.get('waypoints', None)
        self.eq_constraints = kwargs.get('eq_constraints', {})
        self.ineq_constraints = kwargs.get('ineq_constraints', {})
        self.s_velocity_norm = kwargs.get('s_velocity_norm', 1)
        self.s_acceleration_norm = kwargs.get('s_acceleration_norm', 1)
        self.s_obstacles = kwargs.get('s_obstacles', 5)
        self.s_obstacle_alpha = kwargs.get('s_obstacle_alpha', 10)
        self.s_obstacle_margin = kwargs.get('s_obstacle_margin', 1)
        self.s_terminal_potential = kwargs.get('s_terminal_potential', 1e+5)
        self.s_waypoint = kwargs.get('s_waypoint', 1e+5)

    def set_problem(self, **kwargs):
        self.set_parameters(**kwargs)
        if self.workspace is None:
            TrajectoryConstraintObjective.logger.error('Workspace is not defined! Cannot set optimization problem.')
            return
        if self.trajectory is None:
            TrajectoryConstraintObjective.logger.error('Init trajectory is not defined! Cannot set optimization problem.')
            return
        self.problem = MotionObjective(self.T, self.dt, self.config_space_dim)
        # Add workspace obstacles
        for o in self.workspace.obstacles.values():
            if isinstance(o, Circle):
                self.problem.add_sphere(o.origin, o.radius)
            elif isinstance(o, Box):
                self.problem.add_box(o.origin, o.dim)
            else:
                TrajectoryConstraintObjective.logger.warn('Shape {} not supported by bewego'.format(type(o)))
        # Terms
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
            self.problem.add_terminal_potential_terms(self.q_goal, self.s_terminal_potential)
        if self.waypoints is not None:
            for i in range(1, len(self.waypoints) - 1):
                self.problem.add_waypoint_terms(*self.waypoints[i], self.s_waypoint)
        self.objective = self.problem.objective(self.q_init)
    
    def cost(self, trajectory=None):
        if self.objective is not None:
            if trajectory is None:
                trajectory = self.trajectory
            return self.objective.forward(trajectory.active_segment())
        else:
            return 0.

    def optimize(self, nb_steps=100, optimizer='newton'):
        xi = self.trajectory.active_segment()
        if optimizer == 'newton':
            res = optimize.minimize(
                x0=np.array(xi),
                method='Newton-CG',
                fun=self.objective.forward,
                jac=self.objective.gradient,
                hess=self.objective.hessian,
                options={'maxiter': nb_steps, 'disp': self.verbose}
            )
            self.trajectory.active_segment()[:] = res.x
            gradient = res.jac
            delta = res.jac
            dist = np.linalg.norm(
                self.trajectory.final_configuration() - self.q_goal)
            if self.verbose:
                TrajectoryConstraintObjective.logger.info('Gradient norm : ', np.linalg.norm(res.jac))
        elif optimizer == 'ipopt':
            res = minimize_ipopt(
                x0=np.array(xi),
                fun=self.objective.forward,
                jac=self.objective.gradient,
                hess=self.objective.hessian,
                options={'maxiter': nb_steps, 'disp': self.verbose}
            )
            self.trajectory.active_segment()[:] = res.x
            gradient = res.jac
            delta = res.jac
            dist = np.linalg.norm(
                self.trajectory.final_configuration() - self.q_goal)
            if self.verbose:
                TrajectoryConstraintObjective.logger.info('Gradient norm : ', np.linalg.norm(res.jac))
        else:
            TrajectoryConstraintObjective.logger.error('Optimizer %s is not support!' % optimizer)
        return dist < 1.e-3, self.trajectory, gradient, delta

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

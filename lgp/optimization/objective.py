import logging
import numpy as np
from scipy import optimize
from lgp.optimization.cost_terms import LogBarrierFunction

from pyrieef.motion.trajectory import linear_interpolation_trajectory, CliquesFunctionNetwork, TrajectoryObjectiveFunction
from pyrieef.motion.cost_terms import SimplePotential2D, SquaredNormVelocity, SquaredNormAcceleration, BoundBarrier
from pyrieef.geometry.workspace import SignedDistanceWorkspaceMap
from pyrieef.geometry.differentiable_geometry import Pullback, SquaredNorm, Scale, ProductFunction, Compose


class TrajectoryConstraintObjective:
    logger = logging.getLogger(__name__)

    def __init__(self, **kwargs):
        self.verbose = kwargs.get('verbose', False)
        self.config_space_dim = kwargs.get('config_space_dim', 2)
        self.T = kwargs.get('T', 20)   # time steps
        self.dt = kwargs.get('dt', 0.1)  # sample rate
        self.trajectory_space_dim = (self.config_space_dim * (self.T + 2))
        self.workspace = kwargs.get('workspace', None)
        self.signed_distance_field = SignedDistanceWorkspaceMap(self.workspace) if self.workspace is not None else None
        self.objective = None
        # set init trajectory
        self.trajectory = None
        self._q_init = None
        self._q_goal = None
        # setup workspace
        self.box = self.workspace.box if self.workspace is not None else None
        self.extent = self.box.extent if self.box is not None else None
        # constraints  #TODO: extend the constraints when needed
        self.eq_constraints = {}
        self.ineq_constraints = {}
        # params
        self._eta = kwargs.get('eta', 10.)
        self._obstacle_scalar = kwargs.get('obstacle_scalar', 1.)
        self._init_potential_scalar = kwargs.get('init_potential_scalar', 0.)
        self._term_potential_scalar = kwargs.get('term_potential_scalar', 10000000.)
        self._velocity_scalar = kwargs.get('velocity_scalar', 5.)
        self._term_velocity_scalar = kwargs.get('term_velocity_scalar', 100000.)
        self._acceleration_scalar = kwargs.get('acceleration_scalar', 20.)
        self._attractor_stdev = kwargs.get('attractor_stdev', 0.1)

    def set_problem(self, workspace=None, trajectory=None, waypoints=None, eq_constraints={}, ineq_constraints={}):
        if workspace is not None:
            self.workspace = workspace
            self.box = workspace.box
            self.extent = workspace.box.extent()
        if self.workspace is None:
            TrajectoryConstraintObjective.logger.error('Workspace is not defined! Cannot set optimization problem.')
            return
        if trajectory is not None:  # trajectory takes precedented before q_init & q_goal
            self.trajectory = trajectory
            self._q_init = trajectory.initial_configuration()
            self._q_goal = trajectory.final_configuration()
            self.T = trajectory.T()
            self.trajectory_space_dim = (self.config_space_dim * (self.T + 2))
        else:
            if self._q_init is not None and self._q_goal is not None:
                self.trajectory = linear_interpolation_trajectory(self._q_init, self._q_goal, self.T)
            else:
                TrajectoryConstraintObjective.logger.error('Init trajectory or q_init and q_goal are not defined! Cannot set optimization problem.')
                return
        self.eq_constraints = eq_constraints
        self.ineq_constraints = ineq_constraints
        self.signed_distance_field = SignedDistanceWorkspaceMap(self.workspace)
        self.obstacle_potential_from_sdf()
        self.create_clique_network()
        if waypoints is not None:
            for i in range(1, len(waypoints) - 1):
                self.add_waypoint_terms(*waypoints[i], 100000.)
        self.add_all_terms()
        if waypoints is None:
            self.add_attractor(self.trajectory)
        self.create_objective()

    def obstacle_potential_from_sdf(self):
        self.obstacle_potential = SimplePotential2D(self.signed_distance_field)

    def cost(self, trajectory):
        """ compute sum of acceleration """
        return self.objective.forward(trajectory.active_segment())

    def add_attractor(self, trajectory):
        """ Add an attractor to each clique scalled by the distance
            to the goal, it ensures that the trajectory does not slow down
            in time as it progresses towards the goal.
            This is Model Predictive Control grounded scheme.
            TODO check the literature to set this appropriatly. """
        alphas = np.zeros(trajectory.T())
        for t in range(1, trajectory.T()):
            dist = np.linalg.norm(self.q_goal - trajectory.configuration(t))
            alphas[t] = np.exp(-dist / (self._attractor_stdev ** 2))
        alphas /= alphas.sum()
        for t in range(1, trajectory.T()):
            potential = Pullback(SquaredNorm(self.q_goal), self.function_network.center_of_clique_map())
            self.function_network.register_function_for_clique(t, Scale(potential, alphas[t] * self._term_potential_scalar))

    def add_init_and_terminal_terms(self):
        if self._init_potential_scalar > 0.:
            initial_potential = Pullback(SquaredNorm(self.q_init), self.function_network.left_most_of_clique_map())
            self.function_network.register_function_for_clique(0, Scale(initial_potential, self._init_potential_scalar))
        terminal_potential = Pullback(SquaredNorm(self.q_goal), self.function_network.center_of_clique_map())
        self.function_network.register_function_last_clique(Scale(terminal_potential, self._term_potential_scalar))

    def add_waypoint_terms(self, q_waypoint, i, scalar):
        initial_potential = Pullback(SquaredNorm(q_waypoint), self.function_network.left_most_of_clique_map())
        self.function_network.register_function_for_clique(i, Scale(initial_potential, scalar))

    def add_final_velocity_terms(self):
        derivative = Pullback(SquaredNormVelocity(self.config_space_dim, self.dt),
                              self.function_network.left_of_clique_map())
        self.function_network.register_function_last_clique(Scale(derivative, self._term_velocity_scalar))

    def add_smoothness_terms(self, deriv_order=2):
        if deriv_order == 1:
            derivative = Pullback(SquaredNormVelocity(
                self.config_space_dim, self.dt),
                self.function_network.left_of_clique_map())
            self.function_network.register_function_for_all_cliques(Scale(derivative, self._velocity_scalar))
            # TODO change the last clique to have 0 velocity change
            # when linearly interpolating
        elif deriv_order == 2:
            derivative = SquaredNormAcceleration(self.config_space_dim, self.dt)
            self.function_network.register_function_for_all_cliques(Scale(derivative, self._acceleration_scalar))
        else:
            raise ValueError("deriv_order ({}) not suported".format(deriv_order))

    def add_isometric_potential_to_all_cliques(self, potential, scalar):
        """
        Apply the following euqation to all cliques:

                c(x_t) | d/dt x_t |

            The resulting Riemanian metric is isometric. TODO see paper.
            Introduced in CHOMP, Ratliff et al. 2009.
        """
        cost = Pullback(potential, self.function_network.center_of_clique_map())
        squared_norm_vel = Pullback(
            SquaredNormVelocity(self.config_space_dim, self.dt),
            self.function_network.right_of_clique_map())
        self.function_network.register_function_for_all_cliques(Scale(ProductFunction(cost, squared_norm_vel), scalar))

    def add_obstacle_barrier(self):
        """ obstacle barrier function """
        if self.signed_distance_field is None:
            TrajectoryConstraintObjective.logger.error('SDF is not defined! Cannot set obstacle barrier.')
            return
        barrier = LogBarrierFunction()
        barrier.set_mu(20.)
        potential = Compose(barrier, self.signed_distance_field)
        # self.obstacle_potential = potential
        self.function_network.register_function_for_all_cliques(
            Pullback(potential, self.function_network.center_of_clique_map()))

    def add_obstacle_terms(self, geodesic=False):
        """ Takes a matrix and adds an isometric potential term
            to all cliques """
        assert self.obstacle_potential is not None
        self.add_isometric_potential_to_all_cliques(
            self.obstacle_potential, self._obstacle_scalar)

    def add_box_limits(self):
        v_lower = np.array([self.extent.x_min, self.extent.y_min])
        v_upper = np.array([self.extent.x_max, self.extent.y_max])
        box_limits = BoundBarrier(v_lower, v_upper)
        self.function_network.register_function_for_all_cliques(Pullback(
            box_limits, self.function_network.center_of_clique_map()))

    def create_clique_network(self):
        self.function_network = CliquesFunctionNetwork(
            self.trajectory_space_dim,
            self.config_space_dim)

    def create_objective(self):
        """ resets the objective """
        self.objective = TrajectoryObjectiveFunction(self.q_init, self.function_network)

    def add_all_terms(self):
        self.add_final_velocity_terms()
        self.add_smoothness_terms(1)
        self.add_smoothness_terms(2)
        self.add_obstacle_terms()
        self.add_box_limits()
        self.add_init_and_terminal_terms()
        # self.add_obstacle_barrier()

    def optimize(self, nb_steps=100):
        xi = self.trajectory.active_segment()
        res = optimize.minimize(
                x0=np.array(xi),
                method='Newton-CG',
                fun=self.objective.forward,
                jac=self.objective.gradient,
                hess=self.objective.hessian,
                # constraints=[self.eq_constraints, self.ineq_constraints],
                options={'maxiter': nb_steps, 'disp': self.verbose}
            )
        self.trajectory.active_segment()[:] = res.x
        gradient = res.jac
        delta = res.jac
        dist = np.linalg.norm(self.trajectory.final_configuration() - self.q_goal)
        if self.verbose:
            print(("gradient norm: ", np.linalg.norm(res.jac)))
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

    @property
    def eta(self):
        return self._eta

    @eta.setter
    def eta(self, value):
        self._eta = value

    @property
    def obstacle_scalar(self):
        return self._obstacle_scalar

    @obstacle_scalar.setter
    def obstacle_scalar(self, value):
        self._obstacle_scalar = value

    @property
    def init_potential_scalar(self):
        return self._init_potential_scalar

    @init_potential_scalar.setter
    def init_potential_scalar(self, value):
        self._init_potential_scalar = value

    @property
    def term_potential_scalar(self):
        return self._term_potential_scalar

    @term_potential_scalar.setter
    def term_potential_scalar(self, value):
        self._term_potential_scalar = value

    @property
    def velocity_scalar(self):
        return self._velocity_scalar

    @velocity_scalar.setter
    def velocity_scalar(self, value):
        self._velocity_scalar = value

    @property
    def term_velocity_scalar(self):
        return self._term_velocity_scalar

    @term_velocity_scalar.setter
    def term_velocity_scalar(self, value):
        self._term_velocity_scalar = value

    @property
    def acceleration_scalar(self):
        return self._acceleration_scalar

    @acceleration_scalar.setter
    def acceleration_scalar(self, value):
        self._acceleration_scalar = value

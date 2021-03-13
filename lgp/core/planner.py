import logging
import numpy as np
import matplotlib.pyplot as plt
from lgp.logic.planner import LogicPlanner
from lgp.geometry.workspace import YamlWorkspace, HumoroWorkspace
from lgp.geometry.trajectory import linear_interpolation_waypoints_trajectory
from lgp.optimization.objective import TrajectoryConstraintObjective

from pyrieef.geometry.workspace import SignedDistanceWorkspaceMap
from pyrieef.geometry.pixel_map import sdf

from humoro.hmp_interface import HumanRollout


class LGP(object):
    logger = logging.getLogger(__name__)
    SUPPORTED_ACTIONS = ('move', 'pick', 'place')

    def __init__(self, domain, problem, workspace_config, **kwargs):
        self.verbose = kwargs.get('verbose', False)
        self.logic_planner = LogicPlanner(domain, problem)
        self.workspace = YamlWorkspace(workspace_config)
        init_symbols = self.symbol_sanity_check()
        self.workspace.set_init_robot_symbol(init_symbols)
        self.workspace.update_symbolic_state()
        # human motion API # TODO: this is where the wrapper to query human motion comes in
        self.human_predictor = kwargs.get('human_predictor', None)
        if self.human_predictor is None:  # add decoys (just for demo)
            self.human_predictor = self.workspace.humans
        self.objective = TrajectoryConstraintObjective(**kwargs)
        self.action_map = {
            'move': self._move_action,
            'pick': self._pick_action,
            'place': self._place_action
        }

    def action_precondition_check(self, action, update=False):
        if update:
            self.workspace.update_geometric_state()
            self.workspace.update_symbolic_state()
        applied = LogicPlanner.applicable(self.workspace.symbolic_state, action.positive_preconditions, action.negative_preconditions)
        if not applied:
            LGP.logger.error('Sanity check failed! Cannot perform action %s' % action.name)
            LGP.logger.info('Current workspace state: %s' % str(self.workspace.symbolic_state))
            LGP.logger.info('Action parameters: %s' % str(action.parameters))
            LGP.logger.info('Action positive preconditions: %s' % str(action.positive_preconditions))
            LGP.logger.info('Action negative preconditions: %s' % str(action.negative_preconditions))
        return applied

    def act(self, action, sanity_check=True):
        return self.action_map[action.name](action, sanity_check=sanity_check)

    def plan(self):
        '''
        This function will plan a full path conditioned on action skeleton sequence from initial symbolic & geometric states
        '''
        self.workspace.clear_paths()
        # for now, always choose first plan
        plan_idx = 0
        paths, act_seqs = self.logic_planner.plan()
        if self.verbose:
            for i, seq in enumerate(act_seqs):
                LGP.logger.info('Solution %d:' % (i + 1))
                for a in seq:
                    LGP.logger.info(a.name + ' ' + ' '.join(a.parameters))
        # check initial condition
        if not self.action_precondition_check(act_seqs[plan_idx][0]):
            LGP.logger.warn('Preconditions for first action do not satify! Planning may fail.')
        # LGP is highly handcrafted for predicate realization in geometric planning. Somehow a data-driven approach is prefered...
        # this algorithm has no timing coordination (a research question for LGP timing coordination in multi-agent scenario)
        waypoints = {robot_frame: [(self.workspace.geometric_state[robot_frame], 0)] for robot_frame in self.workspace.robots}
        for action in act_seqs[plan_idx]:
            if action.name == 'move':
                robot_frame, location1_frame, location2_frame = action.parameters
                t = len(waypoints[robot_frame]) * self.objective.T
                waypoints[robot_frame].append((self.workspace.geometric_state[location2_frame], t))
            else:
                self.act(action, sanity_check=False)  # sanity check is not needed in planning ahead. This is only a projection of final effective space.
        for robot_frame in self.workspace.robots:
            robot = self.workspace.robots[robot_frame]
            # this is a handcrafted code for setting human as an obstacle.
            if ('avoid_human', robot_frame) in self.workspace.symbolic_state:
                for human in self.human_predictor:
                    self.workspace.obstacles[human] = self.human_predictor[human]
            else:
                for human in self.human_predictor:
                    self.workspace.obstacles.pop(human, None)
            trajectory = linear_interpolation_waypoints_trajectory(waypoints[robot_frame])
            self.objective.set_problem(workspace=self.workspace, trajectory=trajectory, waypoints=waypoints[robot_frame])
            reached, traj, grad, delta = self.objective.optimize()
            if reached:
                robot.paths.append(traj)  # add planned path
            else:
                LGP.logger.warn('Trajectory optim for robot %s failed! Gradients: %s, delta: %s' % (robot_frame, grad, delta))

    def draw_potential_heightmap(self, nb_points=100, show=True):
        fig = plt.figure(figsize=(8, 8))
        extents = self.workspace.box.box_extent()
        ax = fig.add_subplot(111)
        signed_dist_field = self._compute_signed_dist_field(nb_points=nb_points)
        im = ax.imshow(signed_dist_field, cmap='inferno', interpolation='nearest', extent=extents)
        fig.colorbar(im)
        self.workspace.draw_robot_paths(ax, show=False)
        if show:
            plt.show()

    def _compute_signed_dist_field(self, nb_points=100):
        meshgrid = self.workspace.box.stacked_meshgrid(nb_points)
        sdf_map = np.asarray(SignedDistanceWorkspaceMap(self.workspace)(meshgrid))
        sdf_map = (sdf_map < 0).astype(float)
        signed_dist_field = np.asarray(sdf(sdf_map))
        signed_dist_field = np.flip(signed_dist_field, axis=0)
        signed_dist_field = np.interp(signed_dist_field, (signed_dist_field.min(), signed_dist_field.max()), (0, max(self.workspace.box.dim)))
        return signed_dist_field

    def _move_action(self, action, sanity_check=False):
        '''
        This only use for dynamic planning.
        '''
        # geometrically sanity check
        if sanity_check and not self.action_precondition_check(action):
            return
        robot_frame, location1_frame, location2_frame = action.parameters  # location1 is only for symbol checking
        robot = self.workspace.robots[robot_frame]
        # this is a handcrafted code for setting human as an obstacle.
        if ('avoid_human', robot_frame) in self.workspace.symbolic_state:
            for human in self.human_predictor:
                self.workspace.obstacles[human] = self.human_predictor[human]
        else:
            for human in self.human_predictor:
                self.workspace.obstacles.pop(human, None)
        # TODO: add more constrainst if needed
        # for now use location2 state as goal, but it should be specified geometrically
        self.objective.q_init = self.workspace.geometric_state[robot_frame]
        self.objective.q_goal = self.workspace.geometric_state[location2_frame]
        self.objective.set_problem(workspace=self.workspace)
        reached, traj, grad, delta = self.objective.optimize()
        if reached:
            robot.paths.append(traj)  # add planned path
        else:
            LGP.logger.warn('Trajectory optim for action %s failed! Gradients: %s, delta: %s' % (action.parameters, grad, delta))

    def _pick_action(self, action, sanity_check=False):
        # geometrically sanity check
        if sanity_check and not self.action_precondition_check(action):
            return
        robot_frame, obj_frame, location = action.parameters
        robot = self.workspace.robots[robot_frame]
        obj_property = self.workspace.kin_tree.nodes[obj_frame]
        robot.attach_object(obj_frame, obj_property['link_obj'])
        # update kinematic tree (attaching object at agent origin, this could change if needed)
        self.workspace.kin_tree.remove_edge(location, obj_frame)
        obj_property['link_obj'].origin = np.zeros(self.workspace.geometric_state_shape)
        self.workspace.kin_tree.add_edge(robot_frame, obj_frame, **obj_property)

    def _place_action(self, action, sanity_check=False):
        # geometrically sanity check
        if sanity_check and not self.action_precondition_check(action):
            return
        robot_frame, obj_frame, location = action.parameters
        robot = self.workspace.robots[robot_frame]
        obj_property = self.workspace.kin_tree.nodes[obj_frame]
        robot.drop_object(obj_frame)
        # update kinematic tree (attaching object at location origin, this could change if specifying a intermediate goal)
        self.workspace.kin_tree.remove_edge(robot_frame, obj_frame)
        obj_property['link_obj'].origin = np.zeros(self.workspace.geometric_state_shape)
        self.workspace.kin_tree.add_edge(location, obj_frame, **obj_property)

    def symbol_sanity_check(self):
        problem_symbols = self.logic_planner.init_state
        workspace_symbols = self.workspace.symbolic_state
        assert type(problem_symbols) == frozenset and type(workspace_symbols) == frozenset
        adding_symbols = problem_symbols.difference(workspace_symbols)
        for s in adding_symbols:
            if s[0] in self.workspace.DEDUCED_PREDICATES:
                LGP.logger.warn('Adding symbol %s, which is not deduced by workspace. This can be an inconsistence between initial geometric and symbolic states' % str(s))
            if s[0] not in self.workspace.SUPPORTED_PREDICATES:
                LGP.logger.error('Adding symbol %s, which is not in supported predicates of workspace!' % str(s))
        return adding_symbols

    @property
    def supported_predicates(self):
        return self.workspace.SUPPORTED_PREDICATES


class HumoroLGP(LGP):
    logger = logging.getLogger(__name__)
    SUPPORTED_ACTIONS = ('move', 'pick', 'place')

    def __init__(self, domain, problem, config, **kwargs):
        self.verbose = kwargs.get('verbose', False)
        self.path_to_mogaze = kwargs.get('path_to_mogaze', 'datasets/mogaze')
        self.task_name = config.get('name', 'set_table')
        self.segment_id = config['segment_id']
        self.plan_id = config.get('plan_id', 0)  # for now, PDDL problem is designed so that there is only one solution.
        self.window_len = config.get('window_len', 30)
        self.logic_planner = LogicPlanner(domain, problem)  # this will also build feasibility graph
        self.workspace = HumoroWorkspace(hr=HumanRollout(path_to_mogaze=self.path_to_mogaze), 
                                         config=config)
        self.workspace.initialize_workspace_from_humoro(self.segment_id)
        init_symbols = self.symbol_sanity_check()
        self.workspace.set_init_robot_symbol(init_symbols)
        self.objective = TrajectoryConstraintObjective(**kwargs)
        # dynamic parameters
        self.plan = None
        self.workspace_updated = True  # if workspace is updated by outside factors, the plan should be replanned
        self.t = 0  # current environment timestep
        self.elapsed_t = 0  # elapsed time since the last unchanged plan, should be reset to 0 when plan is changed

    def set_current_symbolic_state(self, s):
        self.logic_planner.state = s
    
    def verify_plan(self):
        '''
        For now, only action move relies on predicate predictions.
        This function should be extended to account for other actions that rely on predicate predictions.
        This checks for over all and end time preconditions.
        '''
        if self.plan is None:
            return False
        current_t = 0
        for i, a in enumerate(self.plan[1]):
            if self.check_verifying_action(a):
                # start precondition
                start = True
                if not (i == 0 and self.elapsed_t != 0):  # don't check for start precondition for currently executing first action
                    p = self.workspace.get_prediction_predicates(self.t + current_t)
                    start = LogicPlanner.applicable(p, a.start_positive_preconditions, a.start_negative_preconditions)
                # TODO: implement check for over all precondition when needed 
                # end precondition
                current_t += a.duration - self.elapsed_t if i == 0 else 0
                if current_t > self.window_len:  # don't verify outside window
                    break
                p = self.workspace.get_prediction_predicates(self.t + current_t)
                end = LogicPlanner.applicable(p, a.end_positive_preconditions, a.end_negative_preconditions)
                if not (start and end):
                    return False
            else:
                current_t += a.duration - self.elapsed_t if i == 0 else 0
                if current_t > self.window_len:  # don't verify outside window
                    break
        return True

    def dynamic_plan(self):
        '''
        This function will plan a full path conditioned on action skeleton sequence from initial symbolic & geometric states
        '''
        # for now, always choose first plan
        if self.workspace_updated:
            paths, act_seqs = self.logic_planner.plan()
            self.plan = [paths[self.plan_id], act_seqs[self.plan_id]]
            self.elapsed_t = 0
        if self.verbose:
            for i, seq in enumerate(act_seqs):
                HumoroLGP.logger.info('Solution %d:' % (i + 1))
                for a in seq:
                    HumoroLGP.logger.info(a.name + ' ' + ' '.join(a.parameters))
        # verify path using symbolic traj
        if not self.verify_plan():
            HumoroLGP.logger.warn('Plan is infeasible at current time: %s. Trying replanning at next trigger.' % (self.t))
            return False
        waypoints = {robot_frame: [(self.workspace.geometric_state[robot_frame], 0)] for robot_frame in self.workspace.robots}
        for action in act_seqs[plan_idx]:
            if action.name == 'move':
                robot_frame, location1_frame, location2_frame = action.parameters
                t = len(waypoints[robot_frame]) * self.objective.T
                waypoints[robot_frame].append((self.workspace.geometric_state[location2_frame], t))
            else:
                self.act(action, sanity_check=False)  # sanity check is not needed in planning ahead. This is only a projection of final effective space.
        for robot_frame in self.workspace.robots:
            robot = self.workspace.robots[robot_frame]
            # this is a handcrafted code for setting human as an obstacle.
            if ('avoid_human', robot_frame) in self.workspace.symbolic_state:
                for human in self.human_predictor:
                    self.workspace.obstacles[human] = self.human_predictor[human]
            else:
                for human in self.human_predictor:
                    self.workspace.obstacles.pop(human, None)
            trajectory = linear_interpolation_waypoints_trajectory(waypoints[robot_frame])
            self.objective.set_problem(workspace=self.workspace, trajectory=trajectory, waypoints=waypoints[robot_frame])
            reached, traj, grad, delta = self.objective.optimize()
            if reached:
                robot.paths.append(traj)  # add planned path
            else:
                HumoroLGP.logger.warn('Trajectory optim for robot %s failed! Gradients: %s, delta: %s' % (robot_frame, grad, delta))
        return True

    def check_verifying_action(self, action):
        for p in act.positive_preconditions.union(act.negative_preconditions):
            if p[0] in self.workspace.VERIFY_PREDICATES:
                return True
        return False

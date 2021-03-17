import logging
import numpy as np
import matplotlib.pyplot as plt
from lgp.logic.planner import LogicPlanner
from lgp.geometry.kinematics import Human, Robot, PointObject
from lgp.geometry.workspace import YamlWorkspace, HumoroWorkspace
from lgp.geometry.trajectory import linear_interpolation_waypoints_trajectory
from lgp.optimization.objective import TrajectoryConstraintObjective
from lgp.utils.helpers import frozenset_of_tuples

from pyrieef.geometry.workspace import SignedDistanceWorkspaceMap
from pyrieef.geometry.pixel_map import sdf

# temporary importing until complication of install is resolve
import os
import sys
_path_file = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_path_file, "../../../humoro"))
from examples.prediction.hmp_interface import HumanRollout


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
            'pick': self._pick_action,
            'place': self._place_action
        }

    def act(self, action):
        return self.action_map[action.name](action)

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

    def _pick_action(self, action):
        robot_frame, obj_frame, location = action.parameters
        # check if current action is already executed
        if self.workspace.kin_tree.has_edge(robot_frame, obj_frame):
            return
        robot = self.workspace.robots[robot_frame]
        obj_property = self.workspace.kin_tree.nodes[obj_frame]
        robot.attach_object(obj_frame, obj_property['link_obj'])
        # update kinematic tree (attaching object at agent origin, this could change if needed)
        self.workspace.kin_tree.remove_edge(location, obj_frame)
        obj_property['link_obj'].origin = np.zeros(self.workspace.geometric_state_shape)
        self.workspace.kin_tree.add_edge(robot_frame, obj_frame)

    def _place_action(self, action):
        robot_frame, obj_frame, location = action.parameters
        # check if current action is already executed
        if not self.workspace.kin_tree.has_edge(robot_frame, obj_frame):
            return
        robot = self.workspace.robots[robot_frame]
        obj_property = self.workspace.kin_tree.nodes[obj_frame]
        robot.drop_object(obj_frame)
        # update kinematic tree (attaching object at location origin, this could change if specifying a intermediate goal)
        self.workspace.kin_tree.remove_edge(robot_frame, obj_frame)
        obj_property['link_obj'].origin = np.zeros(self.workspace.geometric_state_shape)
        self.workspace.kin_tree.add_edge(location, obj_frame)

    def symbol_sanity_check(self):
        problem_symbols = self.logic_planner.current_state
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
        self.window_len = config.get('window_len', 30)  # frames, according to this sampling fps
        self.sim_fps = config['sim_fps']  # simulation fps
        self.fps = config['fps']  # sampling fps
        self.ratio = int(self.sim_fps / self.fps)
        self.logic_planner = LogicPlanner(domain, problem)  # this will also build feasibility graph
        self.workspace = HumoroWorkspace(hr=HumanRollout(path_to_mogaze=self.path_to_mogaze, fps=config['sim_fps']), 
                                         config=config, **kwargs)
        self.workspace.initialize_workspace_from_humoro(self.segment_id)
        self.player = self.workspace.hr.p
        init_symbols = self.symbol_sanity_check()
        constant_symbols = [p for p in init_symbols if p[0] not in self.workspace.DEDUCED_PREDICATES]
        self.workspace.set_constant_symbol(constant_symbols)
        self.objective = TrajectoryConstraintObjective(dt=1/self.fps, **kwargs)
        # dynamic parameters
        self.plan = None
        self.t = 0  # current environment timestep
        self.lgp_t = 0  # lgp time 
        self.elapsed_t = 0  # elapsed time since the last unchanged first action, should be reset to 0 when first action in plan is changed
        self.prev_first_action = None
        # action map
        self.action_map = {
            'move': self._move_action,
            'pick': self._pick_action,
            'place': self._place_action
        }

    def clear_plan(self):
        self.workspace.get_robot_link_obj().paths.clear()
        self.plan = None
    
    def check_verifying_action(self, action):
        for p in action.positive_preconditions.union(action.negative_preconditions):
            if p[0] in self.workspace.VERIFY_PREDICATES:
                return True
        return False

    def check_action_precondition(self, action):
        applied = LogicPlanner.applicable(self.logic_planner.current_state, action.positive_preconditions, action.negative_preconditions)
        if not applied:
            LGP.logger.error('Sanity check failed! Cannot perform action %s' % action.name)
            LGP.logger.info('Current workspace state: %s' % str(self.workspace.symbolic_state))
            LGP.logger.info('Action parameters: %s' % str(action.parameters))
            LGP.logger.info('Action positive preconditions: %s' % str(action.positive_preconditions))
            LGP.logger.info('Action negative preconditions: %s' % str(action.negative_preconditions))
        return applied

    def update_current_symbolic_state(self):
        self.workspace.update_symbolic_state()
        self.logic_planner.current_state = self.workspace.symbolic_state

    def update_workspace(self):
        self.workspace.update_workspace(self.t)

    def increase_timestep(self):
        # track elapsed time of the current unchanged first action in plan
        if self.t < self.workspace.duration:
            self.t += 1
        self.lgp_t += 1
        if self.plan is not None:
            if self.lgp_t % self.ratio == 0:
                self.elapsed_t += 1

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
                # print('Action: ', a.name + ' ' + ' '.join(a.parameters))
                # start precondition
                start = True
                if not (i == 0 and self.elapsed_t != 0):  # don't check for start precondition for currently executing first action
                    p = self.workspace.get_prediction_predicates(self.t + current_t * self.ratio)
                    start = LogicPlanner.applicable(p, a.start_positive_preconditions, a.start_negative_preconditions)
                    # print('Start: ', p, a.start_positive_preconditions, a.start_negative_preconditions, self.t + current_t)
                # TODO: implement check for over all precondition when needed 
                # end precondition
                current_t += a.duration - (self.elapsed_t if i == 0 else 0)
                if current_t > self.window_len:  # don't verify outside window
                    break
                p = self.workspace.get_prediction_predicates(self.t + current_t * self.ratio)
                end = LogicPlanner.applicable(p, a.end_positive_preconditions, a.end_negative_preconditions)
                # print('End: ', p, a.end_positive_preconditions, a.end_negative_preconditions, self.t + current_t)
                # print('Result: ', start and end)
                if not (start and end):
                    return False
            else:
                current_t += a.duration - (self.elapsed_t if i == 0 else 0)
                if current_t > self.window_len:  # don't verify outside window
                    break
        return True

    def dynamic_plan(self):
        '''
        This function will plan a full path conditioned on action skeleton sequence from initial symbolic & geometric states
        '''
        robot = self.workspace.get_robot_link_obj()
        self.clear_plan()
        paths, act_seqs = self.logic_planner.plan(shortest_path=True)
        # check logic planning successful
        if paths and act_seqs:
            self.plan = [paths[0], act_seqs[0]]
        else:
            HumoroLGP.logger.warn('Logic planning failed at current time: %s. Trying replanning at next trigger.' % (self.t))
            return False
        if self.verbose:
            for i, seq in enumerate(act_seqs):
                HumoroLGP.logger.info('Solution %d:' % (i + 1))
                for a in seq:
                    HumoroLGP.logger.info(a.name + ' ' + ' '.join(a.parameters))
        # verify path using symbolic traj from human prediction
        if not self.verify_plan():
            HumoroLGP.logger.warn('Plan is infeasible at current time: %s. Trying replanning at next trigger.' % (self.t))
            self.plan = None
            return False
        # mechanism to track elapsed time of unchanged first action
        current_first_action = self.plan[1][0].name + ' ' + ' '.join(self.plan[1][0].parameters)
        if self.prev_first_action != current_first_action:
            self.elapsed_t = 0
            self.prev_first_action = current_first_action
        waypoints = [(self.workspace.get_robot_geometric_state(), 0)]
        t = 0
        for i, action in enumerate(self.plan[1]):
            if action.name == 'move':
                location_frame = action.parameters[0]
            else:
                location_frame = action.parameters[1]
            t += action.duration - (self.elapsed_t if i == 0 else 0)
            if i == 0 and action.name != 'move':  # this if forces robot to stay at current point while executing pick/place
                waypoints.append((self.workspace.get_robot_geometric_state(), t))
            else:
                waypoints.append((self.workspace.geometric_state[location_frame], t))
        # this is a handcrafted code for setting human as an obstacle.
        if ('agent-avoid-human',) in self.logic_planner.current_state:
            segment = self.workspace.segments[self.segment_id]
            human_pos = self.workspace.hr.get_human_pos_2d(segment, self.t)
            self.workspace.obstacles[self.workspace.HUMAN_FRAME] = Human(origin=human_pos, radius=0.3)
        else:
            self.workspace.obstacles.pop(self.workspace.HUMAN_FRAME, None)
        trajectory = linear_interpolation_waypoints_trajectory(waypoints)
        self.objective.set_problem(workspace=self.workspace, trajectory=trajectory, waypoints=waypoints)
        reached, traj, grad, delta = self.objective.optimize()
        # check geometric planning successful
        if reached:
            robot.paths.append(traj)  # add planned path
        else:
            HumoroLGP.logger.warn('Trajectory optim for robot %s failed! Gradients: %s, delta: %s' % (robot_frame, grad, delta))
            self.plan = None
            return False
        return True

    def act(self, action, **kwargs):
        return self.action_map[action.name](action, **kwargs)

    def _move_action(self, action, sanity_check=True):
        '''
        Move the robot to next point in path plan (one timestep).
        '''
        # geometrically sanity check
        if sanity_check and not self.check_action_precondition(action):
            return
        # currently there is only one path
        robot = self.workspace.get_robot_link_obj()
        self.workspace.set_robot_geometric_state(robot.paths[0].configuration(self.elapsed_t))

    def _pick_action(self, action, sanity_check=True):
        # geometrically sanity check
        if sanity_check and not self.check_action_precondition(action):
            return
        obj_frame, location_frame = action.parameters
        # check if current action is already executed
        if self.workspace.kin_tree.has_edge(self.workspace.robot_frame, obj_frame):
            return
        # take control of obj traj on visualization from now (reflecting robot action)
        if obj_frame in self.player._playbackTrajsObj:
            del self.player._playbackTrajsObj[obj_frame]
        robot = self.workspace.get_robot_link_obj()
        obj_property = self.workspace.kin_tree.nodes[obj_frame]
        robot.attach_object(obj_frame, obj_property['link_obj'])
        # update kinematic tree (attaching object at agent origin, this could change if needed)
        if self.workspace.kin_tree.has_edge(location_frame, obj_frame):
            self.workspace.kin_tree.remove_edge(location_frame, obj_frame)
        obj_property['link_obj'] = PointObject(origin=np.zeros(self.workspace.geometric_state_shape))
        self.workspace.kin_tree.add_edge(self.workspace.robot_frame, obj_frame)

    def _place_action(self, action, place_pos=None, sanity_check=True):
        '''
        Place action: place_pos is a global coordinate
        '''
        # geometrically sanity check
        if sanity_check and not self.check_action_precondition(action):
            return
        obj_frame, location_frame = action.parameters
        # check if current action is already executed
        if not self.workspace.kin_tree.has_edge(self.workspace.robot_frame, obj_frame):
            return
        robot = self.workspace.get_robot_link_obj()
        obj_property = self.workspace.kin_tree.nodes[obj_frame]
        robot.drop_object(obj_frame)
        # update kinematic tree (attaching object at location place_pos)
        self.workspace.kin_tree.remove_edge(self.workspace.robot_frame, obj_frame)
        if place_pos is None:
            place_pos = np.zeros(self.workspace.geometric_state_shape)
        obj_property['link_obj'] = PointObject(origin=place_pos)
        self.workspace.kin_tree.add_edge(location_frame, obj_frame)

    def visualize(self):
        self.workspace.visualize_frame(self.t)

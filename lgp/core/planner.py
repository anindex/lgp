import logging
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process
from multiprocessing.sharedctypes import Value
from ctypes import c_bool
import operator
from lgp.logic.planner import LogicPlanner
from lgp.geometry.kinematics import Human, Robot, PointObject
from lgp.geometry.workspace import YamlWorkspace, HumoroWorkspace
from lgp.geometry.trajectory import linear_interpolation_waypoints_trajectory
from lgp.geometry.geometry import get_closest_point_on_circle, get_point_on_circle
from lgp.optimization.objective import TrajectoryConstraintObjective

from pyrieef.geometry.workspace import SignedDistanceWorkspaceMap, Workspace
from pyrieef.motion.trajectory import Trajectory, linear_interpolation_trajectory
from pyrieef.geometry.pixel_map import sdf

# temporary importing until complication of install is resolve
import os
import sys
_path_file = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_path_file, "../../../humoro"))
sys.path.append(os.path.join(_path_file, "../../../bewego"))
from pybewego.workspace_viewer_server import WorkspaceViewerServer
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
        lgp_config = config['lgp']
        self.task_name = lgp_config.get('name', 'set_table')
        self.task_id = lgp_config['task_id']
        self.segment_id = lgp_config['segment_id']
        self.sim_fps = lgp_config['sim_fps']  # simulation fps
        self.fps = lgp_config['fps']  # sampling fps
        self.human_freq = lgp_config.get('human_freq', 40)  # human placement frequency according to fps
        self.traj_init = lgp_config.get('traj_init', 'outer')  # initialization scheme for trajectory
        self.window_len = lgp_config.get('window_len', 'max')  # frames, according to this sampling fps
        self.enable_viewer = lgp_config.get('enable_viewer', False)
        self.ratio = int(self.sim_fps / self.fps)
        self.logic_planner = LogicPlanner(domain, problem, **config['logic'])  # this will also build feasibility graph
        self.workspace = HumoroWorkspace(hr=HumanRollout(path_to_mogaze=self.path_to_mogaze, fps=self.sim_fps), 
                                         config=config['workspace'], **kwargs)
        self.workspace.initialize_workspace_from_humoro(self.task_id, self.segment_id)
        if self.window_len == 'max':
            self.window_len = int(self.workspace.duration / self.ratio)
        self.player = self.workspace.hr.p
        init_symbols = self.symbol_sanity_check()
        constant_symbols = [p for p in init_symbols if p[0] not in self.workspace.DEDUCED_PREDICATES]
        self.workspace.set_constant_symbol(constant_symbols)
        self.human_radius = 0.25
        self._precompute_human_placement()
        self.landmarks = {
            'table': get_point_on_circle(np.pi/2, self.workspace.kin_tree.nodes['table']['limit']),
            'big_shelf': get_point_on_circle(np.pi, self.workspace.kin_tree.nodes['big_shelf']['limit']),
            'small_shelf': get_point_on_circle(0, self.workspace.kin_tree.nodes['small_shelf']['limit'])
        }
        # dynamic parameters
        self.reset()
        # action map
        self.action_map = {
            'move': self._move_action,
            'pick': self._pick_action,
            'place': self._place_action
        }
        # viewers
        if self.enable_viewer == True:
            self.viewer = WorkspaceViewerServer(Workspace(), use_gl=False)

    def reset(self):
        self.clear_plan()
        self.t = 0  # current environment timestep
        self.lgp_t = 0  # lgp time 
        self.symbolic_elapsed_t = 0  # elapsed time since the last unchanged first action, should be reset to 0 when first action in symbolic plan is changed
        self.geometric_elapsed_t = 0  # elapsed time since the last unchanged geometric plan, should be reset to 0 invoking geometric replan
        self.prev_current_action = None

    def clear_plan(self):
        self.workspace.get_robot_link_obj().paths.clear()
        self.plan = None
        self.plans = []
        self.objectives = []

    def get_current_plan_time(self):
        if self.plan is None:
            return 0
        return sum([a.duration for a in self.plan[1]])
    
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
                self.symbolic_elapsed_t += 1
                self.geometric_elapsed_t += 1

    def verify_plan(self, plan=None):
        '''
        For now, only action move relies on predicate predictions.
        This function should be extended to account for other actions that rely on predicate predictions.
        This checks for over all and end time preconditions.
        NOTE: now not used!
        '''
        if plan is None:
            if self.plan is None:
                return False
            plan = self.plan
        plan_t = -self.symbolic_elapsed_t
        for action in plan[1]:
            if plan_t + action.duration > 0:
                if self.check_verifying_action(action):
                    # print('Action: ', action.name + ' ' + ' '.join(action.parameters))
                    # start precondition
                    start = True
                    if not (plan_t < 0 and plan_t + action.duration > 0):  # don't check for start precondition for currently executing first action
                        p = self.workspace.get_prediction_predicates(self.t + plan_t * self.ratio)
                        start = LogicPlanner.applicable(p, action.start_positive_preconditions, action.start_negative_preconditions)
                        # print('Start: ', p, action.start_positive_preconditions, action.start_negative_preconditions, self.t + plan_t)
                    # TODO: implement check for over all precondition when needed 
                    # end precondition
                    plan_t += action.duration
                    if plan_t > self.window_len:  # don't verify outside window
                        break
                    p = self.workspace.get_prediction_predicates(self.t + plan_t * self.ratio)
                    end = LogicPlanner.applicable(p, action.end_positive_preconditions, action.end_negative_preconditions)
                    # print('End: ', p, action.end_positive_preconditions, action.end_negative_preconditions, self.t + plan_t)
                    # print('Result: ', start and end)
                    if not (start and end):
                        return False
                else:
                    plan_t += action.duration
                    if plan_t > self.window_len:  # don't verify outside window
                        break
            else:
                plan_t += action.duration
                if plan_t > self.window_len:  # don't verify outside window
                    break
        return True

    def update_goal(self):
        for t in range(self.window_len):
            symbols = self.workspace.get_prediction_predicates(self.t + t * self.ratio)
            obj = self._get_human_carry_obj(symbols)
            if obj is not None:
                p = self._get_predicate_on_obj(self.logic_planner.problem.positive_goals[0], obj)  # for now there is only + goals                    
                if p is not None and p not in self.logic_planner.current_state:
                    neg_p = self._get_predicate_on_obj(self.logic_planner.current_state, obj)
                    if neg_p is not None:
                        self.logic_planner.current_state = self.logic_planner.current_state.difference(frozenset([neg_p]))
                    self.logic_planner.current_state = self.logic_planner.current_state.union(frozenset([p]))

    def get_waypoints(self, plan=None):
        if plan is None:
            plan = self.plan
            if plan is None:
                return None, None
        prev_pivot = self.workspace.get_robot_geometric_state()
        waypoints = [(prev_pivot, 0)]
        waypoint_manifolds = []
        t = -self.symbolic_elapsed_t
        for action in plan[1]:
            if t + action.duration > 0:
                if action.name == 'move':
                    location_frame = action.parameters[0]
                    limit_circle = self.workspace.kin_tree.nodes[location_frame]['limit']
                    if self.traj_init == 'nearest':
                        p = get_closest_point_on_circle(prev_pivot, limit_circle)
                    elif self.traj_init == 'outer':
                        p = self.landmarks[location_frame]
                    waypoints.append((p, t + action.duration))
                    waypoint_manifolds.append((limit_circle, t + action.duration))
                    prev_pivot = self.workspace.geometric_state[location_frame]
            t += action.duration
        if len(waypoints) == 1 or (not waypoint_manifolds):
            HumoroLGP.logger.warn(f'Elapsed time: {self.symbolic_elapsed_t} is larger than total time: {t} of original plan!')
            return None, None
        return waypoints, waypoint_manifolds

    def place_human(self):
        '''
        Populate human as obstacles
        '''
        # remove all human obstacles
        for name in list(self.workspace.obstacles.keys()):
            if 'human' in name:
                del self.workspace.obstacles[name]
        if ('agent-avoid-human',) in self.logic_planner.current_state:
            if self.human_freq == 'once':
                segment = self.workspace.segments[self.segment_id]
                human_pos = self.workspace.hr.get_human_pos_2d(segment, self.t)
                self.workspace.obstacles[self.workspace.HUMAN_FRAME] = Human(origin=human_pos, radius=self.human_radius)
            elif self.human_freq == 'human-at':
                for t in self.human_placements:
                    if t >= self.t:
                        self.workspace.obstacles[self.workspace.HUMAN_FRAME + str(t)] = self.human_placements[t]
            else:
                for t in range(self.window_len):
                    if t % self.human_freq == 0:
                        sim_t = self.t + t * self.ratio
                        if sim_t > self.workspace.duration:
                            break
                    segment = self.workspace.segments[self.segment_id]
                    human_pos = self.workspace.hr.get_human_pos_2d(segment, sim_t)
                    self.workspace.obstacles[self.workspace.HUMAN_FRAME + str(sim_t)] = Human(origin=human_pos, radius=self.human_radius)

    def symbolic_plan(self, update_goal=True, alternative=True, verify_plan=False):
        '''
        This function plan the feasible symbolic trajectory
        update_goal according to human prediction
        '''
        self.clear_plan()
        if update_goal:  # update goal according to human predictions
            self.update_goal()
        paths, act_seqs = self.logic_planner.plan(alternative=alternative)
        for path, acts in zip(paths, act_seqs):
            plan = (path, acts)
            if verify_plan:
                if self.verify_plan(plan=plan):
                    self.plans.append(plan)
            else:
                self.plans.append(plan)
        if not self.plans:
            return False
        return True

    def geometric_plan(self):
        '''
        This function plans full geometric trajectory at initial
        '''
        if not self.plans:
            HumoroLGP.logger.warn('Symbolic plan is empty. Cannot plan trajectory!')
            return False
        # prepare workspace
        self.place_human()
        workspace = self.workspace.get_pyrieef_ws()
        ranking = []
        # compute plan costs
        for i, plan in enumerate(self.plans):
            waypoints, waypoint_manifolds = self.get_waypoints(plan)
            trajectory = linear_interpolation_waypoints_trajectory(waypoints)
            objective = TrajectoryConstraintObjective(dt=1/self.fps, enable_viewer=self.enable_viewer)
            objective.set_problem(workspace=workspace, trajectory=trajectory, waypoint_manifolds=waypoint_manifolds, goal_manifold=waypoint_manifolds[-1][0])
            self.objectives.append(objective)
            ranking.append((objective.cost(), i))
        # rank the plans
        ranking.sort(key=operator.itemgetter(0))
        # optimize the objective according to ranking
        for r in ranking:
            if self.enable_viewer:
                self.viewer.initialize_viewer(self.objectives[r[1]], self.objectives[r[1]].trajectory)
                status = Value(c_bool, True)
                p = Process(target=self.objectives[r[1]].optimize, args=(status,))
                p.start()
                self.viewer.run()
                p.join()
                success = status.value
                traj = Trajectory(self, q_init=self.viewer.q_init, x=self.viewer.active_x)
            else:
                success, traj = self.objectives[r[1]].optimize()
            if success:  # choose this plan
                self.plan = self.plans[r[1]]
                if self.verbose:
                    for a in self.plan[1]:
                        HumoroLGP.logger.info(a.name + ' ' + ' '.join(a.parameters))
                robot = self.workspace.get_robot_link_obj()
                robot.paths.append(traj)
                return True
        HumoroLGP.logger.warn('All plan geometrical optimization infeasible!')
        return False

    def geometric_replan(self):
        '''
        This function plan partial trajectory upto next symbolic change
        '''
        if not self.plans:
            HumoroLGP.logger.warn('Symbolic plan is empty. Cannot plan geometric trajectory!')
            return False
        if self._check_non_move():
            return True
        # clear previous paths
        robot = self.workspace.get_robot_link_obj()
        robot.paths.clear()
        self.objectives.clear()
        self.geometric_elapsed_t = 0
        # prepare workspace
        self.place_human()
        workspace = self.workspace.get_pyrieef_ws()
        ranking = []
        for plan in self.plans:
            a, t = self._get_next_move(plan)
            location = a.parameters[0]
            current = self.workspace.get_robot_geometric_state()
            goal_manifold = self.workspace.kin_tree.nodes[location_frame]['limit']
            goal = get_closest_point_on_circle(current, goal_manifold)
            trajectory = linear_interpolation_trajectory(current, goal, t)
            objective = TrajectoryConstraintObjective(dt=1/self.fps, enable_viewer=self.enable_viewer)
            objective.set_problem(workspace=workspace, trajectory=trajectory, goal_manifold=goal_manifold)
            self.objectives.append(objective)
            ranking.append((objective.cost(), i))
        # rank the plans
        ranking.sort(key=operator.itemgetter(0))
        # optimize the objective according to ranking
        for r in ranking:
            if self.enable_viewer:
                self.viewer.initialize_viewer(self.objectives[r[1]], self.objectives[r[1]].trajectory)
                status = Value(c_bool, True)
                p = Process(target=self.objectives[r[1]].optimize, args=(status,))
                p.start()
                self.viewer.run()
                p.join()
                success = status.value
                traj = Trajectory(self, q_init=self.viewer.q_init, x=self.viewer.active_x)
            else:
                success, traj = self.objectives[r[1]].optimize()
            if success:  # choose this plan
                self.plan = self.plans[r[1]]
                if self.verbose:
                    for a in self.plan[1]:
                        HumoroLGP.logger.info(a.name + ' ' + ' '.join(a.parameters))
                # mechanism to track elapsed time of unchanged first action
                current_action = self.plan[1][0]
                current_action = current_action.name + ' ' + ' '.join(current_action.parameters)
                if self.prev_current_action != current_action:
                    self.symbolic_elapsed_t = 0
                    self.prev_current_action = current_action
                robot.paths.append(traj)
                return True
        HumoroLGP.logger.warn(f'All replan geometrical optimization infeasible at current time {self.lgp_t}. Trying replanning at next trigger.')
        return False

    def get_current_action(self):
        if self.plan is None:
            HumoroLGP.logger.warn('Symbolic plan is empty. Cannot get current action!')
            return None
        t = 0
        for action in self.plan[1]:
            if t + action.duration > self.symbolic_elapsed_t:
                return action
            t += action.duration
        return None

    def act(self, action=None, **kwargs):
        if action is None: # execute current action
            action = self.get_current_action()
            if action is None:  # if symbolic_elapsed_t is greater than total time of the plan
                return
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
        self.workspace.set_robot_geometric_state(robot.paths[0].configuration(self.geometric_elapsed_t))

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

    def _get_human_carry_obj(self, s):
        '''
        Assuming human carries only one object
        '''
        for p in s:
            if p[0] == 'human-carry':
                return p[1]
        return None
    
    def _get_predicate_on_obj(self, s, obj):
        '''
        Assuming object on only one place
        '''
        for p in s:
            if p[0] == 'on' and p[1] == obj:
                return p
        return None

    def _check_non_move(self):
        if not self.plans:
            return True
        for plan in self.plans:
            for a in plan[1]:
                if a.name == 'move':
                    return False
        return True
    
    def _get_next_move(self, plan):
        if not plan:
            return None, 0
        t = 0
        for a in plan[1]:
            t += a.duration
            if a.name == 'move':
                return a, t
        return None, t

    def _precompute_human_placement(self):
        self.human_placements = {}
        prev_at, at = False, False
        for t in range(self.workspace.duration):
            s = self.workspace.get_prediction_predicates(t)
            at = False
            for p in s:
                if p[0] == 'human-at':
                    at = True
                    break
            if at and not prev_at:
                segment = self.workspace.segments[self.segment_id]
                human_pos = self.workspace.hr.get_human_pos_2d(segment, t)
                self.human_placements[t] = Human(origin=human_pos, radius=self.human_radius)
            prev_at = at

    def visualize(self):
        self.workspace.visualize_frame(self.t)

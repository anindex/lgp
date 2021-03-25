import numpy as np
import pybullet as p
import time
import logging
from lgp.utils.helpers import load_yaml_config
from lgp.logic.parser import PDDLParser
from lgp.core.planner import HumoroLGP
from lgp.geometry.geometry import get_angle

# temporary importing until complication of install is resolve
import os
import sys
_path_file = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_path_file, "../../../humoro"))
from humoro.trajectory import Trajectory

class DynamicLGP(object):
    '''
    General dynamic LGP class
    '''
    def __init__(self, **kwargs):
        domain_file = kwargs.get('domain_file')
        problem_file = kwargs.get('problem_file')
        config_file = kwargs.get('config_file')
        self.domain = PDDLParser.parse_domain(domain_file)
        self.problem = PDDLParser.parse_problem(problem_file)
        self.config = load_yaml_config(config_file)
    
    def run(self):
        raise NotImplementedError()


class HumoroDynamicLGP(DynamicLGP):
    '''
    Humoro environment to interfacing with humoro
    '''
    logger = logging.getLogger(__name__)

    def __init__(self, **kwargs):
        super(HumoroDynamicLGP, self).__init__(**kwargs)
        self.humoro_lgp = HumoroLGP(self.domain, self.problem, self.config, **kwargs)
        # parameters
        self.trigger_period = kwargs.get('trigger_period', 10)  # timesteps
        self.timeout = kwargs.get('timeout', 100)  # seconds
        # useful variables
        self.player = self.humoro_lgp.workspace.hr.p
        self.robot_frame = self.humoro_lgp.workspace.robot_frame
        self.prev_robot_pos = self.humoro_lgp.workspace.get_robot_geometric_state()
        self.prev_robot_q = [0, 0, 0, 1]
        np.seterr(all='raise')

    def check_goal_reached(self):
        return self.humoro_lgp.logic_planner.current_state in self.humoro_lgp.logic_planner.goal_states

    def update_visualization(self):
        '''
        This update currently has no playback (backward in time)
        '''
        # update robot
        robot = self.humoro_lgp.workspace.get_robot_link_obj()
        current_robot_pos = self.humoro_lgp.workspace.get_robot_geometric_state()
        try:
            grad = current_robot_pos - self.prev_robot_pos
            z_angle = get_angle(grad, np.array([0, 1]))  # angle of current path gradient with y axis
            current_q = p.getQuaternionFromEuler([0, 0, (z_angle if grad[0] < 0 else -z_angle) + np.pi / 2])  # + pi/2 due to default orientation of pepper is x-axis
            self.prev_robot_q = current_q
        except:
            current_q = self.prev_robot_q
        self.prev_robot_pos = current_robot_pos
        p.resetBasePositionAndOrientation(self.player._robots[self.robot_frame], [*current_robot_pos, 0], current_q)
        # update object
        if self.humoro_lgp.plan is not None:
            current_action = self.humoro_lgp.get_current_action()
            if current_action is not None and current_action.name == 'place':
                obj, location = current_action.parameters
                obj_pos = self.humoro_lgp.workspace.geometric_state[location]  # TODO: should be desired place_pos on location, or add an animation of placing here
                p.resetBasePositionAndOrientation(self.player._objects[obj], [*obj_pos, 0.8], [0, 0, 0, 1])  # currently ignore object orientation
            elif robot.couplings:
                for obj in robot.couplings:
                    handling_pos = current_robot_pos + np.array([0.3, 0.2])
                    p.resetBasePositionAndOrientation(self.player._objects[obj], [*handling_pos, 1], [0, 0, 0, 1])  # TODO: for now attach object at robot origin

    def run(self, geometric_replan=False):
        self.humoro_lgp.update_current_symbolic_state()
        success = self.humoro_lgp.symbolic_plan(verify_plan=False)
        success = self.humoro_lgp.geometric_plan()
        if not success:
            HumoroDynamicLGP.logger.info('Task failed!')
            return
        max_t = max(self.humoro_lgp.workspace.duration, self.humoro_lgp.ratio * self.humoro_lgp.objective.T)
        while self.humoro_lgp.lgp_t < max_t:
            if geometric_replan and (self.humoro_lgp.lgp_t % (self.trigger_period * self.humoro_lgp.ratio) == 0):
                success = self.humoro_lgp.geometric_plan()
                # self.humoro_lgp.workspace.draw_workspace()
                self.humoro_lgp.draw_potential_heightmap()
            if self.humoro_lgp.lgp_t % self.humoro_lgp.ratio == 0:
                # executing current action in the plan
                if success:
                    self.humoro_lgp.act(sanity_check=False)
                self.humoro_lgp.update_workspace()
                # reflecting changes in PyBullet
                self.update_visualization()
            self.humoro_lgp.visualize()
            self.humoro_lgp.increase_timestep()
            time.sleep(1 / self.humoro_lgp.sim_fps)
        self.humoro_lgp.update_workspace()
        self.humoro_lgp.update_current_symbolic_state()
        if self.check_goal_reached():
            HumoroDynamicLGP.logger.info('Task complete successfully!')
        else:
            HumoroDynamicLGP.logger.info('Task failed!')

    def dynamic_run(self):
        '''
        Main dynamic LGP planning routine
        '''
        success = True
        max_t = self.timeout * self.humoro_lgp.sim_fps
        while self.humoro_lgp.lgp_t < max_t and not self.check_goal_reached():
            self.humoro_lgp.update_current_symbolic_state()
            if self.humoro_lgp.lgp_t % (self.trigger_period * self.humoro_lgp.ratio) == 0:
                success = self.humoro_lgp.dynamic_plan()
                # self.humoro_lgp.workspace.draw_workspace()
                # self.humoro_lgp.draw_potential_heightmap()
            if self.humoro_lgp.lgp_t % self.humoro_lgp.ratio == 0:
                # executing first action in the plan
                if success:
                    self.humoro_lgp.act(self.humoro_lgp.plan[1][0], sanity_check=False)
                self.humoro_lgp.update_workspace()
                # reflecting changes in PyBullet
                self.update_visualization()
            self.humoro_lgp.visualize()
            self.humoro_lgp.increase_timestep()
            time.sleep(1 / self.humoro_lgp.sim_fps)
        if self.check_goal_reached():
            HumoroDynamicLGP.logger.info('Task complete successfully!')
        else:
            HumoroDynamicLGP.logger.info('Task failed!')
            
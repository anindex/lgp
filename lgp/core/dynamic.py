import numpy as np
import pybullet as p
import time
import logging
from lgp.logic.parser import PDDLParser
from lgp.core.planner import HumoroLGP
from lgp.geometry.geometry import get_angle, get_point_on_circle
from lgp.geometry.workspace import Circle

# temporary importing until complication of install is resolve
import os
import sys
_path_file = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_path_file, "../../../humoro"))
from examples.prediction.hmp_interface import HumanRollout


class DynamicLGP(object):
    '''
    General dynamic LGP class
    '''
    def __init__(self, **kwargs):
        domain_file = kwargs.get('domain_file')
        problem_file = kwargs.get('problem_file', None)
        self.domain = PDDLParser.parse_domain(domain_file)
        self.problem = None
        if problem_file is not None:
            self.problem = PDDLParser.parse_problem(problem_file)
    
    def run(self):
        raise NotImplementedError()


class HumoroDynamicLGP(DynamicLGP):
    '''
    Humoro environment to interfacing with humoro
    '''
    logger = logging.getLogger(__name__)

    def __init__(self, **kwargs):
        super(HumoroDynamicLGP, self).__init__(**kwargs)
        path_to_mogaze = kwargs.get('path_to_mogaze', 'datasets/mogaze')
        self.hr = HumanRollout(path_to_mogaze=path_to_mogaze)
        self.humoro_lgp = HumoroLGP(self.domain, self.hr, **kwargs)
        # useful variables
        self.robot_frame = self.humoro_lgp.workspace.robot_frame
        self.handling_circle = Circle(np.zeros(2), radius=0.3)
    
    def init_planner(self, **kwargs):
        if 'problem' not in kwargs:
            kwargs['problem'] = self.problem
        self.humoro_lgp.init_planner(**kwargs)
        self.prev_robot_pos = self.humoro_lgp.workspace.get_robot_geometric_state()
        self.q = [0, 0, 0, 1]
        self.z_angle = 0.

    def check_goal_reached(self):
        return self.humoro_lgp.logic_planner.current_state in self.humoro_lgp.logic_planner.goal_states

    def update_visualization(self):
        '''
        This update currently has no playback (backward in time)
        '''
        # update robot
        robot = self.humoro_lgp.workspace.get_robot_link_obj()
        current_robot_pos = self.humoro_lgp.workspace.get_robot_geometric_state()
        grad = current_robot_pos - self.prev_robot_pos
        if np.linalg.norm(grad) > 0:  # prevent numerical error
            z_angle = get_angle(grad, np.array([1, 0]))  # angle of current path gradient with y axis
            self.z_angle = z_angle if grad[1] > 0 else -z_angle
            self.q = p.getQuaternionFromEuler([0, 0, self.z_angle])  # + pi/2 due to default orientation of pepper is x-axis
        self.prev_robot_pos = current_robot_pos
        p.resetBasePositionAndOrientation(self.humoro_lgp.player._robots[self.robot_frame], [*current_robot_pos, 0], self.q)
        # update object
        if self.humoro_lgp.plan is not None:
            current_action = self.humoro_lgp.get_current_action()
            if current_action is not None and current_action.name == 'place':
                obj, location = current_action.parameters
                box = self.humoro_lgp.workspace.kin_tree.nodes[location]['link_obj']
                x = np.random.uniform(box.origin[0] - box.dim[0] / 2, box.origin[0] + box.dim[0] / 2)  # TODO: should be desired place_pos on location, or add an animation of placing here
                y = np.random.uniform(box.origin[1] - box.dim[1] / 2, box.origin[1] + box.dim[1] / 2)
                p.resetBasePositionAndOrientation(self.humoro_lgp.player._objects[obj], [x, y, 0.735], [0, 0, 0, 1])  # currently ignore object orientation
            elif robot.couplings:
                for obj in robot.couplings:
                    self.handling_circle.origin = current_robot_pos
                    handling_pos = get_point_on_circle(self.z_angle, self.handling_circle)
                    p.resetBasePositionAndOrientation(self.humoro_lgp.player._objects[obj], [*handling_pos, 1], [0, 0, 0, 1])  # TODO: for now attach object at robot origin

    def run(self, replan=False, sleep=False):
        self.humoro_lgp.update_current_symbolic_state()
        success = self.humoro_lgp.symbolic_plan()
        if not replan:
            success = self.humoro_lgp.geometric_plan()
        if not success:
            HumoroDynamicLGP.logger.info('Task failed!')
            return
        max_t = self.humoro_lgp.timeout * self.humoro_lgp.ratio
        while self.humoro_lgp.lgp_t < max_t:
            if replan and (self.humoro_lgp.lgp_t % (self.humoro_lgp.trigger_period * self.humoro_lgp.ratio) == 0):
                self.humoro_lgp.update_current_symbolic_state()
                if self.humoro_lgp.plan is None:
                    success = self.humoro_lgp.symbolic_plan()
                success = self.humoro_lgp.geometric_replan()
            if self.humoro_lgp.lgp_t % self.humoro_lgp.ratio == 0:
                # executing current action in the plan
                if replan:
                    if success:
                        self.humoro_lgp.act(sanity_check=False)
                else:
                    self.humoro_lgp.act(sanity_check=False)
                self.humoro_lgp.update_workspace()
                # reflecting changes in PyBullet
                self.update_visualization()
            self.humoro_lgp.visualize()
            self.humoro_lgp.increase_timestep()
            if self.humoro_lgp.lgp_t > self.humoro_lgp.workspace.duration and self.humoro_lgp.symbolic_elapsed_t > self.humoro_lgp.get_current_plan_time():
                break
            if sleep:
                time.sleep(1 / self.humoro_lgp.sim_fps)
        self.humoro_lgp.update_workspace()
        self.humoro_lgp.update_current_symbolic_state()
        if self.check_goal_reached():
            HumoroDynamicLGP.logger.info('Task complete successfully!')
        else:
            HumoroDynamicLGP.logger.info('Task failed!')
            
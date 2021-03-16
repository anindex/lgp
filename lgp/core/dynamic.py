import numpy as np
import pybullet as p
import time
import logging
from lgp.utils.helpers import load_yaml_config
from lgp.logic.parser import PDDLParser
from lgp.core.planner import HumoroLGP

# temporary importing until complication of install is resolve
import os
import sys
_path_file = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_path_file, "../../../humoro"))
from humoro.trajectory import Trajectory


def convert_to_humoro_traj(traj):
    x_idx, y_idx = 2 * np.arange(traj.T() + 2), 2 * np.arange(traj.T() + 2) + 1
    arr = np.zeros((7, traj.T() + 2))
    arr[0], arr[1] = traj.x()[x_idx], traj.x()[y_idx]
    arr[6] = np.ones(traj.T() + 2)
    return Trajectory(data=arr)


def get_angle(v1, v2):
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def get_quaternion_traj(traj):
    '''
    get orientation trajectory along 2D trajectory
    '''
    q_traj = []
    x = np.array([1, 0])
    for i in range(traj.shape[0] - 1):
        v = traj[i + 1] - traj[i]
        z_angle = get_angle(v, x)
        q = p.getQuaternionFromEuler([0, 0, z_angle])
        q_traj.append(q)
    q_traj.append(q)  # copy last q to match traj len
    assert len(q_traj) == traj.shape[0]
    return np.array(q_traj)


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
        self.humoro_lgp = HumoroLGP(self.domain, self.problem, self.config['workspace'], **kwargs)
        # parameters
        self.trigger_period = kwargs.get('trigger_period', 5)  # timesteps
        self.timeout = kwargs.get('timeout', 20)  # seconds
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
            z_angle = get_angle(current_robot_pos - self.prev_robot_pos, np.array([1, 0]))  # angle of current path gradient with y axis
            current_q = p.getQuaternionFromEuler([0, 0, z_angle])
            self.prev_robot_q = current_q
        except:
            current_q = self.prev_robot_q
        self.prev_robot_pos = current_robot_pos
        p.resetBasePositionAndOrientation(self.player._robots[self.robot_frame], [*current_robot_pos, 0], current_q)
        # update object
        if self.humoro_lgp.plan is not None:
            current_action = self.humoro_lgp.plan[1][0]
            if current_action.name == 'place':
                obj, location = current_action.parameters
                obj_pos = self.humoro_lgp.workspace.geometric_state[location]  # TODO: should be desired place_pos on location, or add an animation of placing here
                p.resetBasePositionAndOrientation(self.player._objects[obj], [*obj_pos, 0.8], [0, 0, 0, 1])  # currently ignore object orientation
            elif robot.couplings:
                for obj in robot.couplings:
                    p.resetBasePositionAndOrientation(self.player._objects[obj], [*current_robot_pos, 0.7], [0, 0, 0, 1])  # TODO: for now attach object at robot origin

    def run(self):
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
            
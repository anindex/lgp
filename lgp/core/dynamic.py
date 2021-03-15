import numpy as np
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


def get_quaternion_traj(traj):
    '''
    get orientation trajectory along 2D trajectory
    '''
    q_traj = []
    x = np.array([1, 0])
    for i in range(traj.shape[0] - 1):
        v = traj[i + 1] - traj[i]
        z_angle = np.arccos(np.dot(v, x) / np.linalg.norm(v))
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
    def __init__(self, **kwargs):
        super(HumoroDynamicLGP, self).__init__(**kwargs)
        self.humoro_lgp = HumoroLGP(self.domain, self.problem, self.config['workspace'], **kwargs)
        self.player = self.humoro_lgp.workspace.hr.p
        # parameters
        self.trigger_period = kwargs.get('trigger_period', 20)  # timesteps

    def check_goal_reached(self):
        return self.humoro_lgp.logic_planner.current_state in self.humoro_lgp.logic_planner.goal_states

    def update_visualization(self):
        # robot traj
        robot = self.humoro_lgp.workspace.get_robot_link_obj()
        robot_traj = convert_to_humoro_traj(robot.paths[0])
        robot_traj.data[2] = np.ones(robot_traj.data.shape[1])  # set height
        robot_traj.data[3:] = get_quaternion_traj(robot_traj.data[:2])  # compute quaternion from 2D traj
        robot_traj.startframe = self.humoro_lgp.t
        self.player.addPlaybackTrajRobot(self.humoro_lgp.workspace.robot_frame, robot_traj)
        # object traj
        current_action = self.humoro_lgp.plan[1][0]
        if current_action.name == 'place':
            obj_traj = np.zeros((7, 1))
            location = current_action.parameters[1]
            obj_traj[:2] = self.humoro_lgp.workspace.geometric_state[location]  # TODO: should be desired place_pos on location
            obj_traj[6] = 1
            traj = Trajectory(data=obj_traj, startframe=self.humoro_lgp.t)
            self.player.addPlaybackTrajObj(traj, obj=current_action.parameters[0])
        elif robot.couplings:
            for obj in robot.couplings:
                self.player.addPlaybackTrajObj(robot_traj, obj=obj)  # TODO: for now attach object at robot origin

    def run(self):
        '''
        Main dynamic LGP planning routine
        '''
        prev_first_action = None
        # while not self.check_goal_reached():
        #     self.humoro_lgp.update_current_symbolic_state()
        #     if self.humoro_lgp.t % self.trigger_period == 0:
        #         self.humoro_lgp.dynamic_plan()
        #         # reflecting changes in PyBullet
        #         self.update_visualization()
        #     # executing first action in the plan
        #     self.humoro_lgp.act(self.humoro_lgp.plan[1][0])
        #     self.humoro_lgp.visualize()
        #     # track elapsed time of the current unchanged first action in plan
        #     self.humoro_lgp.t += 1
        #     current_first_action = self.humoro_lgp.plan[1][0].name + ' ' + ' '.join(self.humoro_lgp.plan[1][0].parameters)
        #     if prev_first_action != current_first_action:
        #         self.humoro_lgp.elapsed_t = 0
        #         prev_first_action = current_first_action
        #     else:
        #         self.humoro_lgp.elapsed_t += 1

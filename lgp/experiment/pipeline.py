import logging
import sys
import os
from os.path import dirname, realpath, join, expanduser
import pickle
from datetime import datetime

from lgp.logic.problem import Problem
from lgp.utils.helpers import frozenset_of_tuples
from lgp.core.dynamic import HumoroDynamicLGP

_path_file = dirname(realpath(__file__))
_domain_dir = join(_path_file, '../../data', 'scenarios')
_dataset_dir = join(_path_file, '../../datasets', 'mogaze')
_data_dir = join(_path_file, '../../data', 'experiments')
_model_dir = join(expanduser("~"), '.qibullet', '1.4.3')
robot_model_file = join(_model_dir, 'pepper.urdf')


class Experiment(object):

    logger = logging.getLogger(__name__)
    def __init__(self, **kwargs):
        self.verbose = kwargs.get('verbose', False)
        self.task = kwargs.get('task', 'set_table')
        domain_file = kwargs.get('domain_file', join(_domain_dir, 'domain_' + self.task + '.pddl'))
        robot_model_file= kwargs.get('robot_model_file', join(_model_dir, 'pepper.urdf'))
        mogaze_dir = kwargs.get('mogaze_dir', _dataset_dir)
        self.data_dir = kwargs.get('data_dir', _data_dir)
        self.data_name = join(self.data_dir, self.task + str(datetime.now()))
        os.makedirs(self.data_dir, exist_ok=True)
        self.engine = HumoroDynamicLGP(domain_file=domain_file, robot_model_file=robot_model_file, path_to_mogaze=mogaze_dir, verbose=self.verbose)
        # experiment params
        self.total_pnp = kwargs.get('total_pnp', [6, 7])
        self.taskid = kwargs.get('taskid', [2, 3])  # set table for 2, 3 people
        self.human_carry = kwargs.get('human_carry', 3)
        self.trigger_period = kwargs.get('trigger_period', 10)
        self.start_agent_symbols = frozenset([('agent-avoid-human',), ('agent-free',)])
        self.end_agent_symbols = frozenset([('agent-at', 'table')])
        self.get_segments()
        # experiment storing
        self.segment_data = {}

    def get_segments(self):
        self.segments = {}
        domain = self.engine.humoro_lgp.logic_planner.domain
        for i in self.taskid:
            segments = self.engine.hr.get_data_segments(taskid=i)
            for segment in segments:
                objects = self.engine.hr.get_object_carries(segment)
                n_carries = len(objects)
                if n_carries in self.total_pnp:
                    init_pred = self.engine.hr.get_object_predicates(segment, 0)
                    init_pred = [p for p in init_pred if p[1] in objects]
                    final_pred = self.engine.hr.get_object_predicates(segment, segment[2] - segment[1] - 1)
                    final_pred = [p for p in final_pred if p[1] in objects]
                    # init problem
                    problem = Problem()
                    problem.name = str(segment)
                    problem.domain_name = domain.name
                    problem.objects = {'object': objects,  
                                       'location': domain.constants['location']}
                    problem.state = frozenset_of_tuples(init_pred).union(self.start_agent_symbols)
                    problem.positive_goals = [frozenset_of_tuples(final_pred).union(self.end_agent_symbols)]
                    problem.negative_goals = [frozenset()]
                    self.segments[segment] = problem
    
    def save_data(self):
        with open(self.data_name, 'wb') as f:
            pickle.dump(self.segment_data, f)
    
    def run(self):
        for segment, problem in self.segments.items():
            # single plan
            self.engine.init_planner(segment=segment, problem=problem, 
                                     human_carry=self.human_carry, trigger_period=self.trigger_period,
                                     human_freq='human-at', traj_init='outer')
            single_success = self.engine.run(replan=False, sleep=False)
            # dynamic plan
            self.engine.init_planner(segment=segment, problem=problem, 
                                     human_carry=self.human_carry, trigger_period=self.trigger_period,
                                     human_freq='once', traj_init='nearest')
            dynamic_success = self.engine.run(replan=True, sleep=False)
            data = self.engine.get_experiment_data()
            data['single_success'] = single_success
            data['dynamic_success'] = dynamic_success
            self.segment_data[segment] = data
            self.engine.reset_experiment()

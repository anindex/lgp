import sys
import argparse
import time
import yaml
import logging
from ast import literal_eval as make_tuple
from os.path import join, dirname, abspath, expanduser

logging.basicConfig(level=logging.INFO)

ROOT_DIR = join(dirname(abspath(__file__)), '..')
DATA_DIR = join(ROOT_DIR, 'data', 'scenarios')
MODEL_DIR = join(expanduser("~"), '.qibullet', '1.4.3')
DATASET_DIR = join(ROOT_DIR, 'datasets', 'mogaze')
sys.path.append(ROOT_DIR)

from lgp.core.dynamic import HumoroDynamicLGP
from lgp.experiment.pipeline import Experiment


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python test_lgp.py \"(\'p7_3\', 29439, 33249)\"')
parser.add_argument('segment', help='The scenario name of the domain, problem file', type=str)
parser.add_argument('-v', help='verbose', type=bool, default=False)
args = parser.parse_args()

domain_file = join(DATA_DIR, 'domain_set_table.pddl')
robot_model_file = join(MODEL_DIR, 'pepper.urdf')
start_time = time.time()
segment = make_tuple(args.segment)
engine = HumoroDynamicLGP(domain_file=domain_file, robot_model_file=robot_model_file, path_to_mogaze=DATASET_DIR,
                          enable_viewer=args.v, verbose=args.v)
objects = engine.hr.get_object_carries(segment)
start_agent_symbols = frozenset([('agent-avoid-human',), ('agent-free',)])
end_agent_symbols = frozenset([('agent-at', 'table')])
problem = Experiment.get_problem_from_segment(engine.hr, segment, engine.domain, objects, start_agent_symbols, end_agent_symbols)
engine.init_planner(segment=segment, problem=problem, 
                    human_carry=3, trigger_period=10,
                    human_freq='once', traj_init='nearest')
init_time = time.time()
print('Init time: ' + str(init_time - start_time) + 's')
engine.run(replan=True, sleep=False)
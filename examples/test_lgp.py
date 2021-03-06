import sys
import argparse
import time
import yaml
import logging
from ast import literal_eval as make_tuple
from os.path import join, dirname, abspath, expanduser
import os
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

logging.basicConfig(level=logging.INFO)

ROOT_DIR = join(dirname(abspath(__file__)), '..')
DATA_DIR = join(ROOT_DIR, 'data', 'scenarios')
MODEL_DIR = join(expanduser("~"), '.qibullet', '1.4.3')
DATASET_DIR = join(ROOT_DIR, 'datasets', 'mogaze')
sys.path.append(ROOT_DIR)

from lgp.core.dynamic import HumoroDynamicLGP
from lgp.experiment.pipeline import Experiment


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python test_lgp.py --segment \"(\'p7_3\', 29439, 33249)\"')
parser.add_argument('--segment', help='The scenario name of the domain, problem file', type=str, default="(\'p5_1\', 100648, 108344)")
parser.add_argument('-d', help='dynamic', type=bool, default=False)
parser.add_argument('-p', help='prediction', type=bool, default=False)
parser.add_argument('-v', help='verbose', type=bool, default=False)
args = parser.parse_args()

domain_file = join(DATA_DIR, 'domain_set_table.pddl')
robot_model_file = join(MODEL_DIR, 'pepper.urdf')
start_time = time.time()
segment = make_tuple(args.segment)
sim_fps = 30 if args.p else 120 
engine = HumoroDynamicLGP(domain_file=domain_file, robot_model_file=robot_model_file, path_to_mogaze=DATASET_DIR,
                          sim_fps=sim_fps, prediction=args.p, enable_viewer=args.v, verbose=args.v)
objects = engine.hr.get_object_carries(segment, predicting=False)
start_agent_symbols = frozenset([('agent-avoid-human',), ('agent-free',)])
end_agent_symbols = frozenset([('agent-at', 'table')])
problem = Experiment.get_problem_from_segment(engine.hr, segment, engine.domain, objects, start_agent_symbols, end_agent_symbols)
human_freq = 'once' if args.d else 'human-at'
traj_init = 'nearest' if args.d else 'outer'
engine.init_planner(segment=segment, problem=problem, 
                    human_carry=3, trigger_period=10,
                    human_freq=human_freq, traj_init=traj_init)
init_time = time.time()
print('Init time: ' + str(init_time - start_time) + 's')
engine.run(replan=args.d, sleep=True, save_frame=False)
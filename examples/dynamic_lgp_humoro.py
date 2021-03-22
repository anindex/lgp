import sys
import argparse
import time
import yaml
import logging
from os.path import join, dirname, abspath, expanduser

logging.basicConfig(level=logging.INFO)

ROOT_DIR = join(dirname(abspath(__file__)), '..')
DATA_DIR = join(ROOT_DIR, 'data', 'scenarios')
MODEL_DIR = join(expanduser("~"), '.qibullet', '1.4.3')
DATASET_DIR = join(ROOT_DIR, 'datasets', 'mogaze')
sys.path.append(ROOT_DIR)

from lgp.core.dynamic import HumoroDynamicLGP

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python lgp_planner_humoro.py set_table')
parser.add_argument('scenario', help='The scenario name of the domain, problem file', type=str)
parser.add_argument('-p', help='problem number', type=str, default='1')
parser.add_argument('-v', help='verbose', type=bool, default=False)
args = parser.parse_args()

domain_file = join(DATA_DIR, 'domain_' + args.scenario + '.pddl')
problem_file = join(DATA_DIR, 'problem_' + args.scenario + args.p + '.pddl')
config_file = join(DATA_DIR, args.scenario + args.p + '.yaml')
robot_model_file = join(MODEL_DIR, 'pepper.urdf')

start_time = time.time()
lgp = HumoroDynamicLGP(domain_file=domain_file, problem_file=problem_file, config_file=config_file, 
                       robot_model_file=robot_model_file, path_to_mogaze=DATASET_DIR, verbose=args.v)
init_time = time.time()
print('Init time: ' + str(init_time - start_time) + 's')
lgp.run()
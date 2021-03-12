import sys
import argparse
import time
import yaml
import logging
from os.path import join, dirname, abspath

logging.basicConfig(level=logging.INFO)

ROOT_DIR = join(dirname(abspath(__file__)), '..')
DATA_DIR = join(ROOT_DIR, 'data', 'scenarios')
DATASET_DIR = join(ROOT_DIR, 'datasets', 'mogaze')
sys.path.append(ROOT_DIR)

from lgp.logic.parser import PDDLParser  # noqa
from lgp.core.planner import HumoroLGP  # noqa

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python lgp_planner_humoro.py set_table')
parser.add_argument('scenario', help='The scenario name of the domain, problem file', type=str)
parser.add_argument('-id', help='Segment id in the MoGaze dataset', type=int, default=1)
parser.add_argument('-v', help='verbose', type=bool, default=False)
args = parser.parse_args()

domain_file = join(DATA_DIR, 'domain_' + args.scenario + '.pddl')
problem_file = join(DATA_DIR, 'problem_' + args.scenario + '.pddl')
start_time = time.time()
domain = PDDLParser.parse_domain(domain_file)
problem = PDDLParser.parse_problem(problem_file)
parse_time = time.time()
print('Parsing time: ' + str(parse_time - start_time) + 's')
lgp = HumoroLGP(domain, problem, verbose=args.v, path_to_mogaze=DATASET_DIR, task_name=args.scenario, segment_id=args.id)
init_time = time.time()
print('Init time: ' + str(init_time - parse_time) + 's')

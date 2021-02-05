import sys
import argparse
import time
import yaml
import logging
from os.path import join, dirname, abspath

logging.basicConfig(level=logging.INFO)

ROOT_DIR = join(dirname(abspath(__file__)), '..')
DATA_DIR = join(ROOT_DIR, 'data', 'scenarios')
sys.path.append(ROOT_DIR)

from lgp.logic.parser import PDDLParser  # noqa
from lgp.core.planner import LGP  # noqa

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python lgp_planner.py prepare_meal')
parser.add_argument('scenario', help='The scenario name of the domain, problem and data file', type=str)
parser.add_argument('-v', help='verbose', type=bool, default=False)
args = parser.parse_args()

domain_file = join(DATA_DIR, 'domain_' + args.scenario + '.pddl')
problem_file = join(DATA_DIR, 'problem_' + args.scenario + '.pddl')
data_file = join(DATA_DIR, args.scenario + '.yaml')
with open(data_file, 'r') as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)
start_time = time.time()
domain = PDDLParser.parse_domain(domain_file)
problem = PDDLParser.parse_problem(problem_file)
parse_time = time.time()
print('Parsing time: ' + str(parse_time - start_time) + 's')
lgp = LGP(domain, problem, config['workspace'], verbose=args.v)
init_time = time.time()
print('Init time: ' + str(init_time - parse_time) + 's')
lgp.plan()
plan_time = time.time()
print('Planning time: ' + str(plan_time - init_time) + 's')
if args.v:
    lgp.lgp_tree.draw_tree()
    lgp.workspace.draw_kinematic_tree()
lgp.workspace.draw_workspace()
lgp.draw_potential_heightmap()

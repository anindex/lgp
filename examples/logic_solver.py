import sys
import argparse
import time
from os.path import join, dirname, abspath

ROOT_DIR = join(dirname(abspath(__file__)), '..')
DATA_DIR = join(ROOT_DIR, 'data', 'scenarios')
sys.path.append(ROOT_DIR)

from lgp.logic.parser import PDDLParser  # noqa
from lgp.logic.planner import PDDLPlanner  # noqa

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python logic_solver.py prepare_meal')
parser.add_argument('scenario', help='The scenario name of the domain and problem file', type=str)
parser.add_argument('-v', help='verbose', type=bool, default=False)
args = parser.parse_args()

domain_file = join(DATA_DIR, 'domain_' + args.scenario + '.pddl')
problem_file = join(DATA_DIR, 'problem_' + args.scenario + '.pddl')

start_time = time.time()
domain = PDDLParser.parse_domain(domain_file)
problem = PDDLParser.parse_problem(problem_file)
parse_time = time.time()
print('Parsing time: ' + str(parse_time - start_time) + 's')
plan = PDDLPlanner.plan(domain, problem)
print('Planning time: ' + str(time.time() - parse_time) + 's')
print('Plan: ')
for act in plan:
    print(act if args.v else act.name + ' ' + ' '.join(act.parameters))

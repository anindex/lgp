import sys
import argparse
from os.path import join, dirname, abspath

ROOT_DIR = join(dirname(abspath(__file__)), '..')
DATA_DIR = join(ROOT_DIR, 'data', 'scenarios')
sys.path.append(ROOT_DIR)

from lgp.logic.parser import PDDLParser  # noqa

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python logic_solver.py prepare_meal')
parser.add_argument('scenario', help='The scenario name of the domain and problem file', type=str)
args = parser.parse_args()

domain_file = join(DATA_DIR, 'domain_' + args.scenario + '.pddl')
problem_file = join(DATA_DIR, 'problem_' + args.scenario + '.pddl')
pddl_parser = PDDLParser()
print('----------------------------')
print(pddl_parser.parse_domain(domain_file))
print('----------------------------')
print(pddl_parser.parse_problem(problem_file))
print('----------------------------')

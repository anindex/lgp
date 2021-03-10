import sys
import argparse
import time
from os.path import join, dirname, abspath

ROOT_DIR = join(dirname(abspath(__file__)), '..')
DATA_DIR = join(ROOT_DIR, 'data', 'scenarios')
sys.path.append(ROOT_DIR)

from lgp.logic.parser import PDDLParser  # noqa
from lgp.logic.tree import LGPTree  # noqa

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python logic_planner.py set_table')
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
# planner = LGPTree(domain, problem)
# build_time = time.time()
# print('Build LGP tree time: ' + str(build_time - parse_time) + 's')
# paths, act_seqs = planner.plan()
# print('Planning time: ' + str(time.time() - build_time) + 's')
# for i, seq in enumerate(act_seqs):
#     print('Solution %d:' % (i + 1))
#     for act in seq:
#         print(act if args.v else act.name + ' ' + ' '.join(act.parameters))
# planner.draw_tree(paths=paths, label=True)

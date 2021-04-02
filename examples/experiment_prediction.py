import sys
import argparse
import time
import yaml
import logging
from os.path import join, dirname, abspath, expanduser

logging.basicConfig(level=logging.INFO)

ROOT_DIR = join(dirname(abspath(__file__)), '..')
sys.path.append(ROOT_DIR)
from lgp.experiment.pipeline import Experiment


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python lgp_planner_humoro.py set_table')
parser.add_argument('-s', help='The scenario name of the domain, problem file', type=str, default='set_table')
parser.add_argument('-v', help='verbose', type=bool, default=False)
args = parser.parse_args()

test_segments = [
    ('p4_1', 2050, 4531),
    ('p4_1', 18373, 20586),
    ('p4_1', 27431, 29344),
    ('p6_1', 33598, 36110),
    ('p2_1', 1272, 3155),
    ('p2_1', 41369, 43303),
    ('p2_1', 137536, 139256),
    ('p6_1', 131335, 135158)
]
start_time = time.time()
experiment = Experiment(task=args.s, test_segments=test_segments, sim_fps=30, prediction=True, verbose=args.v)
init_time = time.time()
print('Init time: ' + str(init_time - start_time) + 's')
experiment.run()
experiment.save_data()

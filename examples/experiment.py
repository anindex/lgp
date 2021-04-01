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
parser.add_argument('-c', help='number of human carry', type=int, default=3)
parser.add_argument('-v', help='verbose', type=bool, default=False)
args = parser.parse_args()

start_time = time.time()
experiment = Experiment(task=args.s, human_carry=args.c, verbose=args.v)
init_time = time.time()
print('Init time: ' + str(init_time - start_time) + 's')
experiment.run()
experiment.save_data()

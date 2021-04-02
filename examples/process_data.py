import pickle
import sys
import argparse
from os.path import join, dirname, abspath, expanduser

ROOT_DIR = join(dirname(abspath(__file__)), '..')
DATA_DIR = join(ROOT_DIR, 'data', 'experiments')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python process_data.py data.p')
parser.add_argument('--name', help='The scenario name of the domain, problem file', type=str, default='main_data.p')
args = parser.parse_args()
data_file = join(DATA_DIR, args.name)

with open(data_file, 'rb') as f:
    data = pickle.load(f)

for segment in data:
    if not data[segment]['single_success'] or not data[segment]['dynamic_success']:
        print(segment)
        

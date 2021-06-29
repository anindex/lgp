import sys
import argparse
from os.path import join, dirname, abspath, realpath
_path_file = dirname(realpath(__file__))
sys.path.append(join(_path_file, "../../humoro"))
from examples.prediction.hmp_interface import HumanRollout

ROOT_DIR = join(dirname(abspath(__file__)), '..')
DATASET_DIR = join(ROOT_DIR, 'datasets', 'mogaze')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python list_segments.py -i 1')
parser.add_argument('-i', help='Task id', type=int)
args = parser.parse_args()

hr = HumanRollout(path_to_mogaze=DATASET_DIR)
segments = hr.get_data_segments(taskid=args.i)
for i, seg in enumerate(segments):
    print(f'segment_id: {i}, segment: {seg}')
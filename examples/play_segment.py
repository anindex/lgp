import sys
import argparse
from os.path import join, dirname, abspath, realpath
_path_file = dirname(realpath(__file__))
sys.path.append(join(_path_file, "../../humoro"))
from examples.prediction.hmp_interface import HumanRollout

ROOT_DIR = join(dirname(abspath(__file__)), '..')
DATASET_DIR = join(ROOT_DIR, 'datasets', 'mogaze')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python play_segment.py -tid 2 -sid 17')
parser.add_argument('-tid', help='Task id', type=int)
parser.add_argument('-sid', help='Segment id', type=int)
args = parser.parse_args()

hr = HumanRollout(path_to_mogaze=DATASET_DIR)
segments = hr.get_data_segments(taskid=args.tid)
segment = segments[args.sid]
hr.load_for_playback(segment)
hr.visualize_frame(segment, 0)
hr.visualize_fullbody(segment)
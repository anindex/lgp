import sys
import time
from os.path import join, dirname, abspath

from humoro.hmp_interface import HumanRollout

ROOT_DIR = join(dirname(abspath(__file__)), '..')
DATA_DIR = join(ROOT_DIR, 'datasets', 'mogaze')
sys.path.append(ROOT_DIR)
from lgp.geometry.workspace import LGPWorkspace  # noqa

start_time = time.time()
segment_id = 1
workspace = LGPWorkspace()
hr = HumanRollout(path_to_mogaze=DATA_DIR)
segments = hr.get_data_segments(taskid=0)
hr.load_for_playback(segments[segment_id])
workspace.initialize_workspace_from_humoro(hr, segments[segment_id])
print('Build workspace tree time: ' + str(time.time() - start_time) + 's')
workspace.draw_kinematic_tree()

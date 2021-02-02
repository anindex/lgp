import logging
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from pyrieef.geometry.workspace import Workspace

from .kinematics import OBJECT_MAP
from lgp.utils.helpers import DRAW_MAP


class LGPWorkspace(Workspace):
    '''
    NOTE: Ideally, we should adopt URDF format to define the workspace. Currently, we use self-defined yaml config for this.
    '''
    logger = logging.getLogger(__name__)
    GLOBAL_FRAME = 'world'

    def __init__(self, config):
        self.name = config['workspace']['name']
        self.kin_tree = LGPWorkspace.build_kinematic_tree(config)
        super(LGPWorkspace, self).__init__(box=self.kin_tree.nodes[LGPWorkspace.GLOBAL_FRAME]['link_obj'])
        self.update_geometric_state()

    def get_global_coordinate(self, frame, x=None):
        if frame not in self.kin_tree:
            LGPWorkspace.logger.warn('Object %s is not in workspace!' % frame)
        if x is None:
            x = np.zeros(self.geometric_state_shape)
        while frame != LGPWorkspace.GLOBAL_FRAME:
            x = self.kin_tree.nodes[frame]['link_obj'].kinematic_map.forward(x)
            frame = list(self.kin_tree.predecessors(frame))[0]
        return x

    def update_geometric_state(self):
        '''
        Update the dict containing objects global coordinates
        '''
        self._geometric_state = {}
        fringe = deque()
        fringe.append(LGPWorkspace.GLOBAL_FRAME)
        while fringe:
            frame = fringe.popleft()
            self._geometric_state[frame] = self.get_global_coordinate(frame)
            for child in self.kin_tree.successors(frame):
                fringe.append(child)

    def draw_workspace(self, show=True):
        fig = plt.figure(figsize=(8, 8))
        extents = self.box.box_extent()
        ax = fig.add_subplot(111, autoscale_on=False, xlim=extents[:2], ylim=extents[2:])
        ax.set_aspect('equal')
        ax.grid()
        for frame in self.geometric_state:
            if frame == LGPWorkspace.GLOBAL_FRAME:
                continue
            typ, color = self.kin_tree.nodes[frame]['type_obj'], self.kin_tree.nodes[frame]['color']
            extents = self.kin_tree.nodes[frame]['link_obj'].extents
            origin = self.geometric_state[frame]
            draw = DRAW_MAP[typ](origin, *extents, facecolor=color)
            ax.add_patch(draw)
            ax.text(*origin, frame, fontsize=10)
        if show:
            plt.show()

    def draw_kinematic_tree(self, show=True):
        node_color = [self.kin_tree.nodes[n]['color'] for n in self.kin_tree]
        nx.draw(self.kin_tree, with_labels=True, node_color=node_color, font_size=10)
        if show:
            plt.show()

    @staticmethod
    def build_kinematic_tree(config):
        tree = nx.DiGraph()
        fringe = deque()
        fringe.append(config['workspace']['tree'])
        while fringe:
            node = fringe.popleft()
            for link in node:
                link_obj = OBJECT_MAP[node[link]['type']](origin=np.array(node[link]['origin']), **node[link]['geometry'])
                tree.add_node(link, link_obj=link_obj, type_obj=node[link]['type'], color=node[link]['color'])
                if node[link]['children'] is not None:
                    for child in node[link]['children']:
                        childname = list(child.keys())[0]
                        tree.add_edge(link, childname)
                        fringe.append(child)
        return tree

    @property
    def symbolic_state(self):
        pass

    @property
    def geometric_state(self):
        return self._geometric_state

    @property
    def geometric_state_shape(self):
        return self.box.origin.shape

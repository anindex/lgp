import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from pyrieef.geometry.workspace import Workspace

from .kinematics import OBJECT_MAP


class LGPWorkspace(Workspace):
    '''
    NOTE: Ideally, we should adopt URDF format to define the workspace. Currently, we use self-defined yaml config for this.
    '''
    def __init__(self, config):
        self.name = config['workspace']['name']
        self.kin_tree = LGPWorkspace.build_kinematic_tree(config)
        super(LGPWorkspace, self).__init__(box=self.kin_tree.nodes['world']['link_obj'])

    @staticmethod
    def build_kinematic_tree(config):
        tree = nx.DiGraph()
        fringe = deque()
        fringe.append(config['workspace']['tree'])
        while fringe:
            node = fringe.popleft()
            for link in node:
                link_obj = OBJECT_MAP[node[link]['type']](origin=np.array(node[link]['origin']), **node[link]['geometry'])
                tree.add_node(link, link_obj=link_obj)
                if node[link]['children'] is not None:
                    for child in node[link]['children']:
                        childname = list(child.keys())[0]
                        tree.add_edge(link, childname)
                        fringe.append(child)
        return tree

    def draw_kinematic_tree(self, show=True):
        node_color = self._color_states()
        nx.draw(self.kin_tree, with_labels=True, node_color=node_color, font_size=10)
        if show:
            plt.show()

    def _color_states(self):
        color_map = []
        for n in self.kin_tree:
            if n == 'robot':
                color_map.append('green')
            else:
                color_map.append('skyblue')
        return color_map

    @property
    def symbolic_state(self):
        pass

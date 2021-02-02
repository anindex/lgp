import logging
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from pyrieef.geometry.workspace import Workspace

from lgp.geometry.kinematics import OBJECT_MAP
from lgp.geometry.transform import LinearTranslation
from lgp.utils.helpers import DRAW_MAP, frozenset_of_tuples


class LGPWorkspace(Workspace):
    '''
    NOTE: Ideally, we should adopt URDF format to define the workspace. Currently, we use self-defined yaml config for this.
    '''
    logger = logging.getLogger(__name__)
    SUPPORTED_PREDICATES = ['at', 'on', 'carry', 'free', 'avoid_human']
    GLOBAL_FRAME = 'world'

    def __init__(self, config, init_symbol=None):
        self.name = config['workspace']['name']
        self.kin_tree = LGPWorkspace.build_kinematic_tree(config)
        super(LGPWorkspace, self).__init__(box=self.kin_tree.nodes[LGPWorkspace.GLOBAL_FRAME]['link_obj'])
        self.update_geometric_state()
        self.update_symbolic_state(init_symbol=init_symbol)

    def get_global_coordinate(self, frame, x=None):
        if frame not in self.kin_tree:
            LGPWorkspace.logger.warn('Object %s is not in workspace!' % frame)
        if x is None:
            x = np.zeros(self.geometric_state_shape)
        while frame != LGPWorkspace.GLOBAL_FRAME:
            x = self.kin_tree.nodes[frame]['link_obj'].kinematic_map.forward(x)
            frame = list(self.kin_tree.predecessors(frame))[0]
        return x

    def get_global_map(self, frame):
        if frame not in self.kin_tree:
            LGPWorkspace.logger.warn('Object %s is not in workspace!' % frame)
        return LinearTranslation(self.geometric_state[frame])

    def transform(self, source_frame, target_frame, x):
        if source_frame not in self.kin_tree or target_frame not in self.kin_tree:
            LGPWorkspace.logger.warn('Frame %s or frame %s is not in workspace!' % (source_frame, target_frame))
        target_global_map = self.get_global_map(target_frame)
        source_local_map = LinearTranslation(target_global_map.backward(self.geometric_state[source_frame]))
        return source_local_map.forward(x)

    def update_symbolic_state(self, init_symbol=None):
        '''
        Update the frozenset of grounded predicates
        NOTE: This function is handcrafted for deducing symbolic state from kinematic tree
        TODO: Extend this function by a more general deduction (could be a research question)
        '''
        symbolic_state = []
        robot = self.kin_tree.nodes['robot']['link_obj']
        fringe = deque()
        locations = list(self.kin_tree.successors(LGPWorkspace.GLOBAL_FRAME))
        locations.remove('robot')
        locations = tuple(locations)
        fringe.extend(locations)
        while fringe:
            frame = fringe.popleft()
            frame_property = self.kin_tree.nodes[frame]
            if frame in locations and frame_property['link_obj'].is_inside(self.geometric_state['robot']):  # this assume locations list
                symbolic_state.append(['at', 'robot', frame])
            for child in self.kin_tree.successors(frame):
                child_property = self.kin_tree.nodes[child]
                if frame in locations and child_property['movable'] and frame_property['link_obj'].is_inside(self.geometric_state[child]):
                    symbolic_state.append(['on', child, frame])
                fringe.append(child)
        # get predicates from robot
        if init_symbol is not None:
            robot.set_init_symbol(init_symbol)
        symbolic_state = frozenset_of_tuples(symbolic_state)
        self._symbolic_state = symbolic_state.union(robot.symbolic_state)

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
            frame_property = self.kin_tree.nodes[frame]
            origin = self.geometric_state[frame]
            extents = frame_property['link_obj'].extents
            if frame_property['type_obj'] == 'box_obj':  # shift origin
                origin = origin - np.array(extents) / 2
            draw = DRAW_MAP[frame_property['type_obj']](origin, *extents, facecolor=frame_property['color'])
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
                link_obj = OBJECT_MAP[node[link]['property']['type_obj']](origin=np.array(node[link]['origin']), **node[link]['geometry'])
                tree.add_node(link, link_obj=link_obj, **node[link]['property'])
                if node[link]['children'] is not None:
                    for child in node[link]['children']:
                        childname = list(child.keys())[0]
                        tree.add_edge(link, childname)
                        fringe.append(child)
        return tree

    @property
    def symbolic_state(self):
        return self._symbolic_state

    @property
    def geometric_state(self):
        return self._geometric_state

    @property
    def geometric_state_shape(self):
        return self.box.origin.shape

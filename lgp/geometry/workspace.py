import logging
import numpy as np
import pybullet as p
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from pyrieef.geometry.pixel_map import PixelMap

from lgp.geometry.kinematics import OBJECT_MAP, EnvBox
from lgp.geometry.transform import LinearTranslation
from lgp.utils.helpers import DRAW_MAP, frozenset_of_tuples, draw_trajectory


class Workspace:
    """
       Contains obstacles.
    """

    def __init__(self, **kwargs):
        self.box = kwargs.get('box', EnvBox())
        self.obstacles = kwargs.get('obstacles', {})

    def in_collision(self, pt):
        for k, obst in self.obstacles.items():
            if obst.dist_from_border(pt) < 0.:
                return True
        return False

    def min_dist(self, pt):
        if len(pt.shape) == 1:
            d_m = float("inf")
        else:
            d_m = np.full((pt.shape[1], pt.shape[2]), np.inf)
        obj = None
        for k, obst in self.obstacles.items():
            d = obst.dist_from_border(pt)
            closer_to_k = d < d_m
            d_m = np.where(closer_to_k, d, d_m)
            obj = np.where(closer_to_k, k, obj)
        return [d_m.tolist(), obj.tolist()]

    def min_dist_gradient(self, pt):
        """ Warning: this gradient is ill defined
            it has a kink when two objects are at the same distance """
        [d_m, obj] = self.min_dist(pt)
        return self.obstacles[obj].dist_gradient(pt)

    def all_points(self):
        points = []
        for k, o in self.obstacles.items():
            points += o.sampled_points()
        return points

    def pixel_map(self, nb_points=100):
        extent = self.box.extent()
        assert extent.x() == extent.y()
        resolution = extent.x() / nb_points
        return PixelMap(resolution, extent)


class YamlWorkspace(Workspace):
    '''
    NOTE: Ideally, we should adopt URDF format to define the workspace. Currently, we use self-defined yaml config for this.
    For now this only supports one agent (robot).
    '''
    logger = logging.getLogger(__name__)
    SUPPORTED_PREDICATES = ('at', 'on', 'carry', 'free', 'avoid_human')
    DEDUCED_PREDICATES = ('at', 'on', 'carry', 'free')
    GLOBAL_FRAME = 'world'

    def __init__(self, config, init_symbol=None, init=True):
        self.name = config['name']
        self.robots = {}
        self.humans = {}
        self.kin_tree = self.build_kinematic_tree(config)
        super(YamlWorkspace, self).__init__(box=self.kin_tree.nodes[YamlWorkspace.GLOBAL_FRAME]['link_obj'])
        # init locations
        self.locations = tuple(location for location in self.kin_tree.successors(YamlWorkspace.GLOBAL_FRAME)
                               if not self.kin_tree.nodes[location]['movable'])
        # init geometric & symbolic states
        if init:
            self.update_geometric_state()
            self.update_symbolic_state(init_symbol=init_symbol)

    def set_init_robot_symbol(self, init_symbol):
        if not self.robots:
            YamlWorkspace.logger.warn('There is no robot frame in workspace!')
            return
        for symbol in init_symbol:
            robot_frame = symbol[1]  # assuming agent argument is always at second place by convention
            if symbol[0] not in YamlWorkspace.SUPPORTED_PREDICATES:
                continue
            if robot_frame in self.robots:
                self.robots[robot_frame].add_symbol(frozenset_of_tuples([symbol]))
            else:
                YamlWorkspace.logger.warn('This symbol %s is not associated with any robot!' % str(symbol))

    def build_kinematic_tree(self, config):
        tree = nx.DiGraph()
        fringe = deque()
        fringe.append(config['tree'])
        while fringe:
            node = fringe.popleft()
            for link in node:
                typ = node[link]['property']['type_obj']
                link_obj = OBJECT_MAP[typ](origin=np.array(node[link]['origin']), **node[link]['geometry'])
                if typ == 'robot':
                    link_obj.name = link
                    self.robots[link] = link_obj
                elif typ == 'human':
                    link_obj.name = link
                    self.humans[link] = link_obj
                tree.add_node(link, link_obj=link_obj, **node[link]['property'])
                if node[link]['children'] is not None:
                    for child in node[link]['children']:
                        childname = list(child.keys())[0]
                        tree.add_edge(link, childname)
                        fringe.append(child)
        return tree

    def clear_paths(self):
        for robot in self.robots.values():
            robot.paths.clear()

    def get_global_coordinate(self, frame, x=None):
        if frame not in self.kin_tree:
            YamlWorkspace.logger.error('Object %s is not in workspace!' % frame)
            return
        if x is None:
            x = np.zeros(self.geometric_state_shape)
        while frame != YamlWorkspace.GLOBAL_FRAME:
            x = self.kin_tree.nodes[frame]['link_obj'].kinematic_map.forward(x)
            frame = list(self.kin_tree.predecessors(frame))[0]
        return x

    def get_global_map(self, frame):
        if frame not in self.kin_tree:
            YamlWorkspace.logger.error('Object %s is not in workspace!' % frame)
            return
        return LinearTranslation(self.geometric_state[frame])

    def transform(self, source_frame, target_frame, x):
        if source_frame not in self.kin_tree or target_frame not in self.kin_tree:
            YamlWorkspace.logger.warn('Frame %s or frame %s is not in workspace!' % (source_frame, target_frame))
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
        fringe = deque()
        fringe.extend(self.locations)
        while fringe:
            frame = fringe.popleft()
            frame_property = self.kin_tree.nodes[frame]
            if frame in self.locations:
                for robot_frame in self.robots:
                    if frame_property['link_obj'].is_inside(self.geometric_state[robot_frame]):  # this assume locations list
                        symbolic_state.append(['at', robot_frame, frame])
            for child in self.kin_tree.successors(frame):
                child_property = self.kin_tree.nodes[child]
                if frame in self.locations and child_property['movable'] and frame_property['link_obj'].is_inside(self.geometric_state[child]):
                    symbolic_state.append(['on', child, frame])
                fringe.append(child)
        self._symbolic_state = frozenset_of_tuples(symbolic_state)
        # get predicates from robot
        if init_symbol is not None:
            self.set_init_robot_symbol(init_symbol)
        for robot in self.robots.values():
            self._symbolic_state = self._symbolic_state.union(robot.symbolic_state)

    def update_geometric_state(self):
        '''
        Update the dict containing objects global coordinates
        '''
        self._geometric_state = {}
        fringe = deque()
        fringe.append(YamlWorkspace.GLOBAL_FRAME)
        while fringe:
            frame = fringe.popleft()
            self._geometric_state[frame] = self.get_global_coordinate(frame)
            for child in self.kin_tree.successors(frame):
                fringe.append(child)

    def draw_workspace(self, show=True):
        self.update_geometric_state()
        fig = plt.figure(figsize=(8, 8))
        extents = self.box.box_extent()
        ax = fig.add_subplot(111, autoscale_on=False, xlim=extents[:2], ylim=extents[2:])
        ax.set_aspect('equal')
        ax.grid()
        for frame in self.geometric_state:
            if frame == YamlWorkspace.GLOBAL_FRAME:
                continue
            frame_property = self.kin_tree.nodes[frame]
            origin = self.geometric_state[frame]
            extents = frame_property['link_obj'].extents
            if frame_property['type_obj'] == 'box_obj':  # shift origin
                origin = origin - np.array(extents) / 2
            draw = DRAW_MAP[frame_property['type_obj']](origin, *extents, facecolor=frame_property['color'])
            ax.add_patch(draw)
            ax.text(*origin, frame, fontsize=10)
        # draw paths
        self.draw_robot_paths(ax, show=False)
        if show:
            plt.show()

    def draw_robot_paths(self, ax, show=True):
        for frame, robot in self.robots.items():
            color = self.kin_tree.nodes[frame]['color']
            for path in robot.paths:
                draw_trajectory(ax, path, color)
        if show:
            plt.show()

    def draw_kinematic_tree(self, show=True):
        node_color = [self.kin_tree.nodes[n]['color'] for n in self.kin_tree]
        nx.draw(self.kin_tree, with_labels=True, node_color=node_color, font_size=10)
        if show:
            plt.show()

    @property
    def symbolic_state(self):
        return self._symbolic_state

    @property
    def geometric_state(self):
        return self._geometric_state

    @property
    def geometric_state_shape(self):
        return self.box.origin.shape


class HumoroWorkspace(YamlWorkspace):
    logger = logging.getLogger(__name__)
    SUPPORTED_PREDICATES = ('agent-at', 'agent-avoid-human', 'agent-carry', 'agent-free', 'agent-avoid-human', 'on', 'human-at', 'human-carry')
    DEDUCED_PREDICATES = ('on', 'human-at', 'human-carry')
    VERIFY_PREDICATES = ('human-at', 'human-carry')
    HUMAN_FRAME = 'human'

    def __init__(self, hr, config):
        super(HumoroWorkspace, self).__init__(config, init=False)
        self.hr = hr
        self.segments = hr.get_data_segments()
        self.segment_id = config['segment_id']
        self.robot_frame = list(self.robots.keys())[0]   # for now only support one robot
        self._symbolic_state = frozenset()

    def set_init_robot_symbol(self, init_symbol):
        self._symbolic_state = self._symbolic_state.union(init_symbol)
    
    def set_robot_geometric_state(self, state):
        self.kin_tree.nodes[self.robot_frame]['link_obj'].origin = np.array(state)
        self.geometric_state[self.robot_frame] = np.array(state)

    def get_robot_geometric_state(self):
        return self.geometric_state[self.robot_frame]
    
    def get_robot_link_obj(self):
        return self.robots[self.robot_frame]

    def get_prediction_predicates(self, t):
        return self.hr.get_predicates(self.segment_id, t)

    def clear_workspace(self):
        for n in self.kin_tree.nodes():
            if n != self.robot_frame and n != YamlWorkspace.GLOBAL_FRAME:
                self.kin_tree.remove_node(n)

    def initialize_workspace_from_humoro(self, segment_id):
        '''
        Initialize workspace using interface from humoro
        '''
        global_frame = YamlWorkspace.GLOBAL_FRAME
        self.hr.load_for_playback(self.segments[segment_id])
        self.segment_id = segment_id
        self.clear_workspace()
        # obstables
        for obj in self.hr.obstacles:
            obj_id = self.hr.p._objects[obj]
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            origin = [pos[0] - .10, pos[1] - .175]
            link_obj = OBJECT_MAP['box_obj'](origin=np.array(origin), dim=np.array([.20, .35]))
            self.kin_tree.add_node(obj, link_obj=link_obj, type_obj='box_obj', movable=False, color=[1., 1., .3, .1])
            self.kin_tree.add_edge(global_frame, obj)
            self.workspace.obstables[obj] = link_obj
        # table
        pos, _ = p.getBasePositionAndOrientation(self.hr.p._objects['table'])
        origin = [pos[0] - .4, pos[1] - .4]
        link_obj = OBJECT_MAP['box_obj'](origin=np.array(origin), dim=np.array([.8, .8]))
        self.kin_tree.add_node('table', link_obj=link_obj, type_obj='box_obj', movable=False, color=[1., .5, .25, 1.])
        self.kin_tree.add_edge(global_frame, 'table')
        # small_shelf
        pos, _ = p.getBasePositionAndOrientation(self.hr.p._objects['vesken_shelf'])
        origin = [pos[0] - .18, pos[1] - .11]
        link_obj = OBJECT_MAP['box_obj'](origin=np.array(origin), dim=np.array([.36, .22]))
        self.kin_tree.add_node('small_shelf', link_obj=link_obj, type_obj='box_obj', movable=False, color=[1., .5, .25, 1.])
        self.kin_tree.add_edge(global_frame, 'small_shelf')
        # small_shelf
        pos, _ = p.getBasePositionAndOrientation(self.hr.p._objects['laiva_shelf'])
        origin = [pos[0] - .30, pos[1] - .12]
        link_obj = OBJECT_MAP['box_obj'](origin=np.array(origin), dim=np.array([.59, .24]))
        self.kin_tree.add_node('big_shelf', link_obj=link_obj, type_obj='box_obj', movable=False, color=[1., .5, .25, 1.])
        self.kin_tree.add_edge(global_frame, 'big_shelf')
        # objects
        for obj in self.hr.obj_names:
            obj_id = self.hr.p._objects[obj]
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            link_obj = OBJECT_MAP['point_obj'](origin=np.array(pos[:2]))
            self.kin_tree.add_node(obj, link_obj=link_obj, type_obj='point_obj', movable=True, color=[0, 1, 1, 0.9])
            self.kin_tree.add_edge(global_frame, obj)
        # init symbolic state
        self._symbolic_state = frozenset_of_tuples(self.hr.get_predicates(self.segments[segment_id], 0))
        # init geometric state
        self.update_geometric_state()

    @property
    def symbolic_state(self):
        return self._symbolic_state

    @property
    def geometric_state(self):
        return self._geometric_state
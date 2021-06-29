import logging
import numpy as np
import pybullet as p
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from collections import deque
from pyrieef.geometry.pixel_map import PixelMap
from pyrieef.geometry.workspace import Circle, Workspace

from lgp.geometry.kinematics import OBJECT_MAP, EnvBox
from lgp.geometry.transform import LinearTranslation
from lgp.utils.helpers import DRAW_MAP, frozenset_of_tuples, draw_trajectory


class LGPWorkspace:
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

    def get_pyrieef_ws(self):
        pyrieef_ws = Workspace(box=self.box)
        pyrieef_ws.obstacles = list(self.obstacles.values())
        return pyrieef_ws


class YamlWorkspace(LGPWorkspace):
    '''
    NOTE: Ideally, we should adopt URDF format to define the workspace. Currently, we use self-defined yaml config for this.
    For now this only supports one agent (robot).
    '''
    logger = logging.getLogger(__name__)
    SUPPORTED_PREDICATES = ('at', 'on', 'carry', 'free', 'avoid_human')
    DEDUCED_PREDICATES = ('at', 'on', 'carry', 'free')
    GLOBAL_FRAME = 'world'
    INIT_ROBOT_POSE = np.array([0, -1.5])

    def __init__(self, config=None, init_symbol=None, init=True, **kwargs):
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

    def build_kinematic_tree(self, config=None):
        tree = nx.DiGraph()
        fringe = deque()
        if config is not None:
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
        else:
            env_obj = OBJECT_MAP['env'](origin=np.zeros(2), dim=np.array([7., 7.]))
            robot_obj = OBJECT_MAP['robot'](origin=self.INIT_ROBOT_POSE, radius=0.1)  # robot default init
            tree.add_node(self.GLOBAL_FRAME, link_obj=env_obj, type_obj='env', color=[1, 1, 1, 1], movable=False)
            tree.add_node('robot', link_obj=robot_obj, type_obj='robot', color=[0, 1, 0, 1], movable=True)
            tree.add_edge(self.GLOBAL_FRAME, 'robot')
            self.robots['robot'] = robot_obj
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
        return ax

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
    DEDUCED_PREDICATES = ('on', 'human-at', 'human-carry', 'agent-at', 'agent-carry', 'agent-free')
    VERIFY_PREDICATES = ('human-at', 'human-carry')
    HUMAN_FRAME = 'Human1'
    HUMAN_RADIUS = 0.2

    def __init__(self, hr, **kwargs):
        super(HumoroWorkspace, self).__init__(init=False, **kwargs)
        self.robot_model_file = kwargs.get('robot_model_file', 'data/models/cube.urdf')
        self.robot_frame = list(self.robots.keys())[0]   # for now only support one robot
        self.hr = hr
        self._symbolic_state = frozenset()
        self.robot_spawned = False

    def set_parameters(self, **kwargs):
        self.segment = kwargs.get('segment')
        self.human_carry = kwargs.get('human_carry', 0)
        self.prediction = kwargs.get('prediction', False)
        self.objects = set(kwargs['objects'])
        if self.prediction or self.human_carry == 'all':
            fraction = 1.0
        else:
            fraction = self.hr.get_fraction_duration(self.segment, self.human_carry)
        self.duration = int(self.hr.get_segment_timesteps(self.segment) * fraction) + (1 if fraction != 1.0 else 0)
        self.set_robot_geometric_state(self.INIT_ROBOT_POSE)  # reset initial robot pose
        self.constant_symbols = frozenset()

    def set_constant_symbol(self, symbols):
        self.constant_symbols = frozenset_of_tuples(symbols)
    
    def set_robot_geometric_state(self, state):
        self.kin_tree.nodes[self.robot_frame]['link_obj'] = OBJECT_MAP['robot'](origin=np.array(state), radius=0.1)

    def get_robot_geometric_state(self):
        return self.geometric_state[self.robot_frame]
    
    def get_robot_link_obj(self):
        return self.robots[self.robot_frame]
    
    def get_human_geometric_state(self):
        return self.geometric_state[self.HUMAN_FRAME]

    def get_prediction_predicates(self, t):
        if t >= self.duration:
            return []
        return self.hr.get_predicates(self.segment, t)

    def get_location(self, x):
        for loc in self.locations:
            if self.kin_tree.nodes[loc]['link_obj'].is_inside(x):
                return loc
        return 'unknown'

    def get_area(self, x):
        for loc in self.locations:
            if self.kin_tree.nodes[loc]['area'].is_inside(x):
                return loc
        return 'unknown'

    def clear_workspace(self):
        for n in list(self.kin_tree.nodes()):
            if n != self.robot_frame and n != YamlWorkspace.GLOBAL_FRAME:
                self.kin_tree.remove_node(n)

    def initialize_workspace_from_humoro(self, **kwargs):
        '''
        Initialize workspace using interface from humoro
        '''
        self.set_parameters(**kwargs)
        global_frame = self.GLOBAL_FRAME
        self.hr.load_for_playback(self.segment)
        self.hr.visualize_frame(self.segment, 0)
        self.clear_workspace()
        # obstables
        i = 0
        for obj in self.hr.obstacles:
            i += 1
            obj_id = self.hr.p._objects[obj]
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            link_obj = OBJECT_MAP['box_obj'](origin=np.array(pos[:2]), dim=np.array([.20, .35]))
            self.kin_tree.add_node('chair' + str(i), link_obj=link_obj, type_obj='box_obj', movable=False, color=[1., .5, .25, 1.])
            self.kin_tree.add_edge(global_frame, 'chair'+ str(i))
            self.obstacles['chair' + str(i)] = Circle(origin=np.array(pos[:2]), radius=0.25)
        # table
        pos, _ = p.getBasePositionAndOrientation(self.hr.p._objects['table'])
        link_obj = OBJECT_MAP['box_obj'](origin=np.array(pos[:2]), dim=np.array([.8, .8]))
        area = Circle(origin=np.array(pos[:2]), radius=1.2)
        limit = Circle(origin=np.array(pos[:2]), radius=1.0)
        self.kin_tree.add_node('table', link_obj=link_obj, area=area, limit=limit, type_obj='box_obj', movable=False, color=[1., .5, .25, 1.])
        self.kin_tree.add_edge(global_frame, 'table')
        self.obstacles['table'] = Circle(origin=np.array(pos[:2]), radius=0.8)
        # small_shelf
        pos, _ = p.getBasePositionAndOrientation(self.hr.p._objects['vesken_shelf'])
        link_obj = OBJECT_MAP['box_obj'](origin=np.array(pos[:2]), dim=np.array([.36, .22]))
        area = Circle(origin=np.array(pos[:2]), radius=0.75)
        limit = Circle(origin=np.array(pos[:2]), radius=0.7)
        self.kin_tree.add_node('small_shelf', link_obj=link_obj, area=area, limit=limit, type_obj='box_obj', movable=False, color=[1., .5, .25, 1.])
        self.kin_tree.add_edge(global_frame, 'small_shelf')
        self.obstacles['small_shelf'] = Circle(origin=np.array(pos[:2]), radius=0.4)
        # big_shelf
        pos, _ = p.getBasePositionAndOrientation(self.hr.p._objects['laiva_shelf'])
        link_obj = OBJECT_MAP['box_obj'](origin=np.array(pos[:2]), dim=np.array([.24, .59]))
        area = Circle(origin=np.array(pos[:2]), radius=0.85)
        limit = Circle(origin=np.array(pos[:2]), radius=0.8)
        self.kin_tree.add_node('big_shelf', link_obj=link_obj, area=area, limit=limit, type_obj='box_obj', movable=False, color=[1., .5, .25, 1.])
        self.kin_tree.add_edge(global_frame, 'big_shelf')
        self.obstacles['big_shelf'] = Circle(origin=np.array(pos[:2]), radius=0.5)
        self.locations = set(['table', 'small_shelf', 'big_shelf'])
        self.update_geometric_state()
        # objects
        for obj in self.objects:
            if obj in self.hr.p._objects:
                obj_id = self.hr.p._objects[obj]
                pos, _ = p.getBasePositionAndOrientation(obj_id)
                pos = np.array(pos[:2])
                location = self.get_location(pos)
                if location != 'unknown':
                    origin = self.kin_tree.nodes[location]['link_obj'].kinematic_map.backward(pos)
                    link_obj = OBJECT_MAP['point_obj'](origin=origin)
                    self.kin_tree.add_edge(location, obj)
                else:
                    link_obj = OBJECT_MAP['point_obj'](origin=pos)
                    self.kin_tree.add_edge(global_frame, obj)
                self.kin_tree.add_node(obj, link_obj=link_obj, type_obj='point_obj', movable=True, color=[0, 1, 1, 0.9])
        # human
        human_pos = self.hr.get_human_pos_2d(self.segment, 0)
        link_obj = OBJECT_MAP['human'](origin=np.array(human_pos))
        self.kin_tree.add_node(self.HUMAN_FRAME, link_obj=link_obj, type_obj='human', movable=True, color=[1, 0, 0, 0.9])
        self.kin_tree.add_edge(global_frame, self.HUMAN_FRAME)
        # init geometric state
        self.update_geometric_state()
        # init symbolic state
        self.update_symbolic_state()
        # init robot pos
        if not self.robot_spawned:
            self.hr.p.spawnRobot(self.robot_frame, urdf=self.robot_model_file)
            self.robot_spawned = True
        p.resetBasePositionAndOrientation(self.hr.p._robots[self.robot_frame], [*self.get_robot_geometric_state(), 0], [0, 0, 0, 1])

    def update_workspace(self, t):
        '''
        Update workspace with human pos and movable objects (for now all are global coordinate)
        '''
        if t < self.duration:
            human_pos = self.hr.get_human_pos_2d(self.segment, t)
            self.kin_tree.nodes[self.HUMAN_FRAME]['link_obj'] = OBJECT_MAP['human'](origin=np.array(human_pos))
        for obj in self.objects:
            if obj in self.hr.p._objects:
                if self.kin_tree.has_edge(self.robot_frame, obj):  # ignore carrying objects
                    continue
                loc = list(self.kin_tree.predecessors(obj))[0]
                self.kin_tree.remove_edge(loc, obj)
                obj_id = self.hr.p._objects[obj]
                pos, _ = p.getBasePositionAndOrientation(obj_id)
                pos = np.array(pos[:2])
                location = self.get_location(pos)
                if location != 'unknown':
                    origin = self.kin_tree.nodes[location]['link_obj'].kinematic_map.backward(pos)
                    link_obj = OBJECT_MAP['point_obj'](origin=origin)
                    self.kin_tree.add_edge(location, obj)
                    self.kin_tree.nodes[obj]['link_obj'] = link_obj
                else:
                    link_obj = OBJECT_MAP['point_obj'](origin=pos)
                    self.kin_tree.add_edge(self.GLOBAL_FRAME, obj)
                    self.kin_tree.nodes[obj]['link_obj'] = link_obj
        self.update_geometric_state()

    def update_symbolic_state(self):
        '''
        Should call update_workspace(t) first.
        '''
        preds = []
        for obj in self.objects:
            for n in self.kin_tree.predecessors(obj):
                if n != self.robot_frame and n != self.GLOBAL_FRAME:
                    preds.append(('on', obj, n))
        # deduce agent-carry
        successors = self.kin_tree.successors(self.robot_frame)
        agent_preds = [('agent-carry', obj) for obj in successors]
        if agent_preds:
            preds.extend(agent_preds)
        else:
            preds.append(('agent-free',))
        location = self.get_area(self.get_robot_geometric_state())
        if location != 'unknown':
            preds.append(('agent-at', location))
        self._symbolic_state = frozenset_of_tuples(preds).union(self.constant_symbols)

    def visualize_frame(self, t):
        self.hr.visualize_frame(self.segment, t)

    @property
    def symbolic_state(self):
        return self._symbolic_state

    @property
    def geometric_state(self):
        return self._geometric_state
import logging
import numpy as np

from lgp.geometry.transform import LinearTranslation
from lgp.utils.helpers import frozenset_of_tuples
from pyrieef.geometry.workspace import Shape, Circle, Box, EnvBox, point_distance_gradient, point_distance_hessian


# TODO: extend these classes when needed
# NOTE: now only support translation from geometrical level to symbolic level of predicates: at, carry, on, free
#       It is defined that only agents (e.g. humans, robots) have symbolic state.
class Human(Circle):
    '''
    For now assuming human is 2D circle agent.
    '''
    SUPPORTED_PREDICATES = []

    def __init__(self, **kwargs):
        origin = np.asarray(kwargs.get('origin', np.zeros(2)))
        radius = np.asarray(kwargs.get('radius', 0.2))
        self.kinematic_map = LinearTranslation(origin)
        super(Human, self).__init__(origin=origin, radius=radius)

    @property
    def symbolic_state(self):  # modify this if there is some predicates defined for human
        return None

    @property
    def geometric_state(self):
        return self.origin

    @property
    def extents(self):
        return None


class Robot(Circle):
    '''
    For now assuming robot is 2D circle agent, and the robot is always children of root world.
    Ideally, we should define the kinematic tree of the robot from URDF.
    However, currently this is a circle robot so it would a simple tree with one depth describing holding objects.
    '''
    logger = logging.getLogger(__name__)
    SUPPORTED_PREDICATES = ['carry', 'free']

    def __init__(self, **kwargs):
        origin = np.asarray(kwargs.get('origin', np.zeros(2)))
        radius = np.asarray(kwargs.get('radius', 0.2))
        self.init_symbol = frozenset_of_tuples(kwargs.get('init_symbol', []))
        self.kinematic_map = LinearTranslation(origin)
        self.couplings = {}  # hold objects. TODO: replace by a kinematic tree later if robot is more complex.
        super(Robot, self).__init__(origin=origin, radius=radius)

    def attach_object(self, x):
        self.couplings[x.name] = x

    def drop_object(self, name):
        if name not in self.couplings:
            Robot.logger.warn('No object name %s holding!' % name)
            return
        self.couplings.pop(name)

    def set_init_symbol(self, init_symbol):
        self.init_symbol = frozenset_of_tuples(init_symbol)

    @property
    def symbolic_state(self):
        symbols = []
        if not self.couplings:
            symbols.append(['free', 'robot'])
        else:
            for obj in self.couplings:
                symbols.append(['carry', 'robot', obj.name])
        return self.init_symbol.union(frozenset_of_tuples(symbols))

    @property
    def geometric_state(self):
        states = [self.origin]
        states.extend([x.origin for n, x in self.couplings])
        return np.concatenate(states, axis=None)  # 1 dim array

    @property
    def extents(self):
        return (self.radius,)


class BoxObject(Box):
    def __init__(self, **kwargs):
        origin = np.asarray(kwargs.get('origin', np.zeros(2)))
        dim = np.asarray(kwargs.get('dim', np.ones(2)))
        self.kinematic_map = LinearTranslation(origin)
        super(BoxObject, self).__init__(origin=origin, dim=dim)

    @property
    def extents(self):
        return tuple(self.dim.tolist())


class PointObject(Shape):
    def __init__(self, **kwargs):
        self.origin = np.asarray(kwargs.get('origin', np.zeros(2)))
        self.kinematic_map = LinearTranslation(self.origin)
        super(PointObject, self).__init__()

    def closest_point(self, x):
        return point_distance_gradient(x, self.origin)

    def dist_gradient(self, x):
        return self.closest_point(x)

    def dist_hessian(self, x):
        return point_distance_hessian(x, self.origin)

    @property
    def extents(self):  # for visible drawing
        return (0.05,)


OBJECT_MAP = {
    'env': EnvBox,
    'human': Human,
    'robot': Robot,
    'box_obj': BoxObject,
    'point_obj': PointObject,
}

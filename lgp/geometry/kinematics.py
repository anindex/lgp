import numpy as np

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
        super(Human, self).__init__(**kwargs)

    @property
    def symbolic_state(self):  # modify this if there is some predicates defined for human
        return None

    @property
    def geometric_state(self):
        return self.origin


class Robot(Circle):
    '''
    For now assuming robot is 2D circle agent.
    Ideally, we should define the kinematic tree of the robot from URDF.
    However, currently this is a circle robot so it would a simple tree with one depth describing holding objects.
    '''
    SUPPORTED_PREDICATES = ['carry', 'free']

    def __init__(self, **kwargs):
        super(Robot, self).__init__(**kwargs)
        self.init_symbol = kwargs.get('init_symbol', frozenset())
        self.couplings = {}  # hold objects. TODO: replace by a kinematic tree later if robot is more complex.

    def attach_object(self, x):
        self.couplings[x.name] = x

    def drop_object(self, x):
        self.couplings.pop(x.name)

    @property
    def symbolic_state(self):
        symbols = []
        if not self.couplings:
            symbols.append(['free', self.name])
        else:
            for obj in self.couplings:
                symbols.append(['carry', self.name, obj.name])
        return self.init_symbol.union(frozenset_of_tuples(symbols))

    @property
    def geometric_state(self):
        states = [self.origin]
        states.extend([x.origin for n, x in self.couplings])
        return np.concatenate(states, axis=None)  # 1 dim array


class PointObject(Shape):
    def __init__(self, **kwargs):
        super(PointObject, self).__init__()
        self.origin = kwargs.get('origin', np.zeros(2))

    def closest_point(self, x):
        return point_distance_gradient(x, self.origin)

    def dist_gradient(self, x):
        return self.closest_point(x)

    def dist_hessian(self, x):
        return point_distance_hessian(x, self.origin)


OBJECT_MAP = {
    'env': EnvBox,
    'human': Human,
    'robot': Robot,
    'box_obj': Box,
    'point_obj': PointObject,
}

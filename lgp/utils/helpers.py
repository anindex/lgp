import matplotlib.patches as patches


DRAW_MAP = {
    'human': patches.Circle,
    'robot': patches.Circle,
    'box_obj': patches.Rectangle,
    'point_obj': patches.Circle,
}


def frozenset_of_tuples(data):
    return frozenset([tuple(t) for t in data])

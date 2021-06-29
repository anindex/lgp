import numpy as np


def get_angle(v1, v2):
    '''
    Get angle from 2 vectors
    '''
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def get_point_on_circle(angle, circle):
    x = circle.origin[0] + (circle.radius * np.cos(angle))
    y = circle.origin[1] + (circle.radius * np.sin(angle))
    return np.array([x, y])


def get_closest_point_on_circle(p, circle):
    '''
    Finding the closest x,y coordinates on circle, based on given point
    '''
    x_vec = np.array([1, 0])
    v = p - circle.origin
    angle = get_angle(v, x_vec) 
    angle = angle if v[1] > 0 else -angle
    return get_point_on_circle(angle, circle)


if __name__ == '__main__':
    from pyrieef.geometry.workspace import Circle 
    circle = Circle(origin=np.zeros(2), radius=1.0)
    p = np.array([0, 2])
    print(get_closest_point_on_circle(p, circle))
    p = np.array([-1, 1])
    print(get_closest_point_on_circle(p, circle))
    p = np.array([-1, 0])
    print(get_closest_point_on_circle(p, circle))
    p = np.array([0, -1])
    print(get_closest_point_on_circle(p, circle))
    p = np.array([1, -1])
    print(get_closest_point_on_circle(p, circle))
    p = np.array([1, 0])
    print(get_closest_point_on_circle(p, circle))
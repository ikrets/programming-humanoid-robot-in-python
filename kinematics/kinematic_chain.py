import numpy as np
from collections import namedtuple
from numpy import sin, cos

Link = namedtuple('Link', ['name', 'translation', 'q_to_roll_pitch_yaw'])


def transform(translation, roll_pitch_yaw):
    def rotate(axis, angle):
        rotation_matrix = np.array([[cos(angle), -sin(angle), 0],
                                    [sin(angle), cos(angle), 0],
                                    [0, 0, 1]])
        start = {'x': 2, 'y': 1, 'z': 0}
        order = [(start[axis] + i) % 3 for i in range(3)]
        rotation_matrix = rotation_matrix[order][:, order]
        return rotation_matrix

    rotation = np.array(np.eye(3))
    roll, pitch, yaw = roll_pitch_yaw
    if roll:
        rotation.dot(rotate('x', roll), rotation)
    if pitch:
        rotation.dot(rotate('y', pitch), rotation)
    if yaw:
        rotation.dot(rotate('z', yaw), rotation)

    tf = np.array(np.zeros((4, 4), dtype=np.float32))
    tf[:3, :3] = rotation
    tf[:3, 3] = np.asarray(translation)
    tf[3, 3] = 1

    return tf


def translation_part(transform):
    return transform[:3, 3].reshape(-1)


class KinematicChain:
    def __init__(self):
        self.links = []

    def append(self, name, translation, q_to_roll_pitch_yaw):
        self.links.append(Link(name=name, translation=np.array(translation),
                               q_to_roll_pitch_yaw=q_to_roll_pitch_yaw))

    def calculate_transforms(self, values):
        transforms = {}

        T = np.eye(4, dtype=np.float64)
        for link in self.links:
            value = values[link.name] if link.name in values else 0
            roll_pitch_yaw = link.q_to_roll_pitch_yaw(value)
            local_transform = transform(link.translation, roll_pitch_yaw)

            T = T.dot(local_transform)
            transforms[link.name] = T

        return transforms

    def transform(self, transform_link):
        flipped = KinematicChain()
        flipped.links = [transform_link(l) for l in self.links]
        return flipped

from numpy.testing import *
from kinematic_chain import transform, KinematicChain, translation_part
import numpy as np
from numpy import sin, cos, pi

atol = 1e-9


def array_from_values(values):
    return np.array([values[k] for k in sorted(values.keys())])


def test_transform():
    assert_allclose(transform([0, 0, 0], [0, 0, 0]), np.eye(4))
    assert_allclose(transform([0, 0, 0], [0, 0, pi / 2]),
                    [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0],
                     [0, 0, 0, 1]], atol=1e-6)
    assert_allclose(transform([1, 3, 0], [0, pi / 4, 0]),
                    [[cos(pi / 4), 0, sin(pi / 4), cos(pi / 4)],
                     [0, 1, 0, 3],
                     [-sin(pi / 4), 0, cos(pi / 4), -sin(pi / 4)],
                     [0, 0, 0, 1]], atol=1e-6)


def test_plane_chain():
    chain = KinematicChain()
    chain.append('link1', [1, 0, 0], lambda q: [0, 0, q])
    chain.append('link2', [2, 0, 0], lambda q: [0, 0, q])

    transforms = chain.calculate_transforms({'link1': 0, 'link2': 0})
    assert_allclose(translation_part(transforms['link1']), [1, 0, 0], atol=atol)
    assert_allclose(translation_part(transforms['link2']), [3, 0, 0], atol=atol)

    transforms = chain.calculate_transforms({'link1': pi / 2, 'link2': 0})
    assert_allclose(translation_part(transforms['link1']), [0, 1, 0], atol=atol)
    assert_allclose(translation_part(transforms['link2']), [0, 3, 0], atol=atol)

    transforms = chain.calculate_transforms({'link1': pi / 2, 'link2': -pi / 2})
    assert_allclose(translation_part(transforms['link1']), [0, 1, 0], atol=atol)
    assert_allclose(translation_part(transforms['link2']), [2, 1, 0], atol=atol)

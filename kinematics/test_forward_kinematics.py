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
                    [[cos(pi / 4), 0, sin(pi / 4), 1],
                     [0, 1, 0, 3],
                     [-sin(pi / 4), 0, cos(pi / 4), 0],
                     [0, 0, 0, 1]], atol=1e-6)


def test_plane_xonly_chain():
    chain = KinematicChain()
    chain.append('link0', [0, 0, 0], lambda q: [0, 0, q])
    chain.append('link1', [1, 0, 0], lambda q: [0, 0, q])
    chain.append('link2', [2, 0, 0], lambda q: [0, 0, q])

    transforms = chain.calculate_transforms(
        {'link0': 0, 'link1': 0, 'link2': 0})
    assert_allclose(translation_part(transforms['link1']), [1, 0, 0], atol=atol)
    assert_allclose(translation_part(transforms['link2']), [3, 0, 0], atol=atol)

    transforms = chain.calculate_transforms(
        {'link0': pi / 2, 'link1': 0, 'link2': 0})
    assert_allclose(translation_part(transforms['link1']), [0, 1, 0], atol=atol)
    assert_allclose(translation_part(transforms['link2']), [0, 3, 0], atol=atol)

    transforms = chain.calculate_transforms(
        {'link0': pi / 2, 'link1': -pi / 2, 'link2': 0})
    assert_allclose(translation_part(transforms['link1']), [0, 1, 0], atol=atol)
    assert_allclose(translation_part(transforms['link2']), [2, 1, 0], atol=atol)

    transforms = chain.calculate_transforms({'link0': pi / 4, 'link1': pi / 6, 'link2': 0})
    assert_allclose(translation_part(transforms['link1']),
                    [0.7071067811865, 0.7071067811865, 0], atol=atol)
    assert_allclose(translation_part(transforms['link2']),
                    [1.2247448713916, 2.6389584337647, 0], atol=atol)

    transforms = chain.calculate_transforms(
        {'link0': -pi / 3, 'link1': -pi / 12, 'link2': 0})
    assert_allclose(translation_part(transforms['link1']),
                    [0.5, -0.8660254037844, 0], atol=atol)
    assert_allclose(translation_part(transforms['link2']),
                    [1.017638090205, -2.7978770563626, 0], atol=atol)


def test_plane_xy_chain():
    chain = KinematicChain()
    chain.append('link0', [0, 0, 0], lambda q: [0, 0, q])
    chain.append('link1', [1, 2, 0], lambda q: [0, 0, q])
    chain.append('link2', [3, -0.5, 0], lambda q: [0, 0, q])

    transforms = chain.calculate_transforms({'link0': 0, 'link1': 0, 'link2': 0})
    assert_allclose(translation_part(transforms['link1']), [1, 2, 0])
    assert_allclose(translation_part(transforms['link2']), [4, 1.5, 0])

    transforms = chain.calculate_transforms({'link0': -pi / 6, 'link1': pi / 3, 'link2': 0})
    assert_allclose(translation_part(transforms['link1']),
                    [1.8660254037844, 1.2320508075689, 0])
    assert_allclose(translation_part(transforms['link2']),
                    [4.7141016151378, 2.2990381056767, 0])


def test_3d_chain():
    chain = KinematicChain()
    chain.append('link0', [0, 0, 0], lambda q: [0, q, 0])
    chain.append('link1', [2, 0, 0], lambda q: [q, 0, 0])
    chain.append('link2', [0, 0, 4], lambda q: [q, 0, 0])

    transforms = chain.calculate_transforms({'link0': 0, 'link1': 0, 'link2': 0})
    assert_allclose(translation_part(transforms['link1']), [2, 0, 0])
    assert_allclose(translation_part(transforms['link2']), [2, 0, 4])

    transforms = chain.calculate_transforms({'link0': -pi / 6, 'link1': pi / 3, 'link2': 0})
    assert_allclose(translation_part(transforms['link1']), [1.7320508076, 0, 1])
    assert_allclose(translation_part(transforms['link2']),
                    [0.7320508076, -3.4641016151, 2.7320508076])

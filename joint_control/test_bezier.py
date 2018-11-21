from numpy.testing import *
from bezier_interpolators import BezierInterpolators


def test_linear_segment():
    names = ["foo1"]
    times = [[0.5, 1]]
    keys = [[[1, [3, 0, 0], [3, 0.1, 0.2]],
             [2, [3, -0.1, -0.2], [3, 0, 0]]]]

    interpolators = BezierInterpolators((names, times, keys))
    assert_allclose(interpolators.compute(0.5).values(), [1],
                    err_msg='keypoint 0')
    assert_allclose(interpolators.compute(1).values(), [2],
                    err_msg='keypoint 1')
    assert_allclose(interpolators.compute(0.75).values(), [1.5],
                    err_msg='value between keypoint 0 and 1')


def test_two_joints_linear_segments():
    names = ["foo1", "foo2"]
    times = [[0.5, 1, 2], [1, 2]]
    keys = [[[1, [3, 0, 0], [3, 0.1, 0.2]],
             [2, [3, -0.1, -0.2], [3, 0.1, 0.2]],
             [3, [3, -0.1, -0.2], [3, 0, 0]]],
            [[2, [3, 0, 0], [3, 1, 3]],
             [5, [3, -1, -3], [3, 0, 0]]]]

    interpolators = BezierInterpolators((names, times, keys))

    assert_allclose(interpolators.compute(1).values(), [2, 2],
                    err_msg='first on keypoint, second middle of section')
    assert_allclose(interpolators.compute(1.5).values(), [2.5, 3.5])
    assert_allclose(interpolators.compute(2).values(), [3, 5],
                    err_msg='both on end keypoint')

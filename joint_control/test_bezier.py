from numpy.testing import *
from bezier_interpolators import BezierInterpolators


def test_linear_segment():
    names = ["foo1", "foo2"]
    times = [[0.5, 1]]
    keys = [[[1, [3, 0.4, 0.9], [3, 0.6, 1.1]],
             [2, [3, 0.9, 1.9], [3, 1.1, 2.1]]]]

    interpolators = BezierInterpolators(times, keys)
    assert_allclose(interpolators.compute(0), [0],
                    err_msg='implicit 0th keypoint')
    assert_allclose(interpolators.compute(0.5), [1],
                    err_msg='value in 1st keypoint')
    assert_allclose(interpolators.compute(1), [2],
                    err_msg='value in 2nd keypoint')
    assert_allclose(interpolators.compute(0.75), [1.5],
                    err_msg='value between 1st and 2nd keypoint')


def test_two_joints_linear_segments():
    names = ["foo1", "foo2"]
    times = [[0.5, 1, 2], [2]]
    keys = [[[1, [3, 0.4, 0.9], [3, 0.6, 1.1]],
             [2, [3, 0.9, 1.9], [3, 1.1, 2.1]],
             [3, [3, 1.9, 2.9], [3, 2.1, 3.1]]],
            [[5, [3, 0, 5], [3, 2.1, 5.1]]]]

    interpolators = BezierInterpolators(times, keys)
    # modify initial curvature
    interpolators.bezier_sections[1, 0, 1] = [2, 0]

    assert_allclose(interpolators.compute(0), [0, 0],
                    err_msg='implicit 0ths keypoints')
    assert_allclose(interpolators.compute(1), [2, 2.5],
                    err_msg='first on keypoint, second middle of section')
    assert_allclose(interpolators.compute(2), [3, 5],
                    err_msg='both on end keypoint')

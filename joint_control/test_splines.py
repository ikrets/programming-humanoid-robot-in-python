from numpy.testing import *
from spline_interpolators import SplineInterpolators
import numpy as np


# key in choregraph format, only the key value is set
def keys_to_choregraph(keys):
    result = []
    for i in range(len(keys)):
        joint = []
        for j in range(len(keys[i])):
            joint.append([keys[i][j], [0, 0, 0], [0, 0, 0]])
        result.append(joint)
    return result


def test_two_joints_linear_segments():
    names = ["foo1", "foo2"]
    # first and last segments cannot be linear so we need 3 segments at least
    times = [[0.5, 1, 1.5], [0.25, 0.5, 0.75, 1]]
    keys = keys_to_choregraph([[1, 2, 3],
                               [3, 4, 8, 10]])

    interpolators = SplineInterpolators((names, times, keys),
                                        {"foo1": 0, "foo2": 1})

    assert_allclose(interpolators.compute(0).values(), [0, 1],
                    err_msg="initial values")
    assert_allclose(interpolators.compute(1.5).values(), [3],
                    err_msg="first on end, second after end not returned")
    assert_allclose(interpolators.compute(0.75).values(), [1.5, 8],
                    err_msg='first on linear section, second on keypoint')

def test_do_not_overshoot():
    names = ["foo"]
    times = [[1, 2]]
    keys = keys_to_choregraph([[-0.00873, -0.35278]])

    interpolators = SplineInterpolators((names, times, keys), {"foo": 0})
    first_section_t = np.linspace(0, 1)
    values = [interpolators.compute(t)['foo'] for t in first_section_t]
    assert_array_equal(np.asarray(values) <= 0, True)

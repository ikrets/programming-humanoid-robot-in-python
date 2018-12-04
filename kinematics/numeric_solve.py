from kinematic_chain import KinematicChain
from scipy.optimize import fmin
import matplotlib.pyplot as plt
import numpy as np


def numeric_solve(chain, end_transform):
    last_link_name = chain.links[-1].name
    names = sorted(link.name for link in chain.links)

    def loss(joints):
        transforms = chain.calculate_transforms(
            {names[i]: joints[i] for i in range(len(names))})

        # the Frobenius norm
        return np.linalg.norm(end_transform - transforms[last_link_name])

    joint_array = np.zeros(len(names))

    func_values = []
    optimization_result = fmin(loss, joint_array,
                               maxiter=1000,
                               callback=lambda x: func_values.append(loss(x)))

    plt.plot(func_values)
    plt.show()

    return names, optimization_result

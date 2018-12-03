'''In this exercise you need to implement forward kinematics for NAO robot

* Tasks:
    1. complete the kinematics chain definition (self.chains in class ForwardKinematicsAgent)
       The documentation from Aldebaran is here:
       http://doc.aldebaran.com/2-1/family/robots/bodyparts.html#effector-chain
    2. implement the calculation of local transformation for one joint in function
       ForwardKinematicsAgent.local_trans. The necessary documentation are:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    3. complete function ForwardKinematicsAgent.forward_kinematics, save the transforms of all body parts in torso
       coordinate into self.transforms of class ForwardKinematicsAgent

* Hints:
    the local_trans has to consider different joint axes and link parameters for different joints
'''

# add PYTHONPATH
import os
import sys
import json
from kinematic_chain import KinematicChain, Link

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..',
                             'joint_control'))

from math import sin, cos, pi
import numpy as np

from angle_interpolation import AngleInterpolationAgent


class ForwardKinematicsAgent(AngleInterpolationAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(ForwardKinematicsAgent, self).__init__(simspark_ip, simspark_port,
                                                     teamname, player_id,
                                                     sync_mode)

        self.transforms = {}

        self.chains = {name: KinematicChain() for name in
                       ['Head', 'LArm', 'LLeg', 'RArm', 'RLeg']}

        x_rotation = lambda q: [q, 0, 0]
        y_rotation = lambda q: [0, q, 0]
        z_rotation = lambda q: [0, 0, q]

        self.chains['Head'].append('HeadYaw', [0, 0, 126.50], z_rotation)
        self.chains['Head'].append('HeadPitch', [0, 0, 0], y_rotation)

        self.chains['LArm'].append('LShoulderPitch', [0, 98, 100], y_rotation)
        self.chains['LArm'].append('LShoulderRoll', [0, 0, 0], z_rotation)
        self.chains['LArm'].append('LElbowYaw', [105, 15, 0], x_rotation)
        self.chains['LArm'].append('LElbowRoll', [0, 0, 0], z_rotation)
        self.chains['LArm'].append('LWristYaw', [55.95, 0, 0],
                                   lambda q: [0, 0, 0])

        self.chains['LLeg'].append('LHipYawPitch', [0, 50, -85],
                                   lambda q: [0, q, pi / 2])
        self.chains['LLeg'].append('LHipRoll', [0, 0, 0], x_rotation)
        self.chains['LLeg'].append('LHipPitch', [0, 0, 0], y_rotation)
        self.chains['LLeg'].append('LKneePitch', [0, 0, -100], y_rotation)
        self.chains['LLeg'].append('LAnklePitch', [0, 0, -102.9], y_rotation)
        self.chains['LLeg'].append('LAnkleRoll', [0, 0, 0], x_rotation)

        def flip_l_to_r(link):
            return Link(name='R' + link.name[1:],
                        translation=link.translation * [1, -1, 1],
                        q_to_roll_pitch_yaw=link.q_to_roll_pitch_yaw)

        self.chains['RArm'] = self.chains['LArm'].transform(flip_l_to_r)
        self.chains['RLeg'] = self.chains['LLeg'].transform(flip_l_to_r)

    def think(self, perception):
        self.forward_kinematics(perception.joint)
        return super(ForwardKinematicsAgent, self).think(perception)

    def local_trans(self, joint_name, joint_angle):
        '''calculate local transformation of one joint

        :param str joint_name: the name of joint
        :param float joint_angle: the angle of joint in radians
        :return: transformation
        :rtype: 4x4 matrix
        '''

        return self.local_transforms[joint_name](joint_angle)

    def forward_kinematics(self, joints):
        '''forward kinematics

        :param joints: {joint_name: joint_angle}
        '''
        for chain in self.chains.values():
            self.transforms.update(chain.calculate_transforms(joints))


if __name__ == '__main__':
    agent = ForwardKinematicsAgent()
    with open('fk_samples.json', 'r') as fp:
        samples = json.load(fp)

    names = samples['joint_names']
    times = np.empty((len(names), 2))
    times[:, 0] = 2
    times[:, 1] = 4
    keys = []

    for i in range(len(samples['joint_names'])):
        keys.append([[samples['init']['angles'][i], [0, 0, 0], [0, 0, 0]],
                     [samples['rest']['angles'][i], [0, 0, 0], [0, 0, 0]]])

    agent.set_keyframes((names, times, keys))
    agent.run()

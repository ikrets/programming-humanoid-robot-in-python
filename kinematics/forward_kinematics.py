'''In this exercise you need to implement forward kinematics for NAO robot

* Tasks:
    1. complete the kinematics chain definition (self.chains in class ForwardKinematicsAgent)
       The documentation from Aldebaran is here:
       http://doc.aldebaran.com/2-1/family/robots/bodyparts.html#effector-chain
    2. implement the calculation of local transformation for one joint in function
       ForwardKinematicsAgent.local_trans. The necessary documentation are:
       http://doc.aldebaran.com/2-1/family/nao_h25/joints_h25.html
       http://doc.aldebaran.com/2-1/family/nao_h25/links_h25.html
    3. complete function ForwardKinematicsAgent.forward_kinematics, save the transforms of all body parts in torso
       coordinate into self.transforms of class ForwardKinematicsAgent

* Hints:
    the local_trans has to consider different joint axes and link parameters for differnt joints
'''

# add PYTHONPATH
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'joint_control'))

import numpy as np

from angle_interpolation import AngleInterpolationAgent

#joints with x-axis rotation:
xJoints = ['LElbowYaw', 'RElbowYaw', 'LHipRoll', 'LAnkleRoll', 'RHipRoll', 'RAnkleRoll']
#joints with y-axis rotation:
yJoints = ['HeadPitch', 'LShoulderPitch', 'RShoulderPitch', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'RHipPitch', 'RKneePitch', 'RAnklePitch']
#joints with z-axis rotation:
zJoints = ['HeadYaw', 'LShoulderRoll', 'LElbowRoll',  'RShoulderRoll', 'RElbowRoll']
#special joints:
wristJoints = ['LWristYaw', 'RWristYaw']

class ForwardKinematicsAgent(AngleInterpolationAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(ForwardKinematicsAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.transforms = {n: np.identity(4) for n in self.joint_names}
        #adding the wrist joints for proper modeling
        self.transforms.update({'LWristYaw': np.identity(4), 'RWristYaw': np.identity(4)})

        # chains defines the name of chain and joints of the chain
        # YOUR CODE HERE
        self.chains = {'Head': ['HeadYaw', 'HeadPitch'],
		       'LArm': ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw'],
		       'LLeg': ['LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll'],
		       'RLeg': ['RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll'],
		       'RArm': ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw']}
	
	self.jointOffsets = { 'LShoulderPitch':  [0.00, 98.00, 100.00], 'LShoulderRoll':  [0.00, 0.00, 0.00], 'LElbowYaw':  [105.00, 15.00, 0.00], 'LElbowRoll':  [0.00, 0.00, 0.00],
			      'HeadYaw':  [0.00, 0.00, 126.50], 'HeadPitch':  [0.00, 0.00, 0.00],'LHipYawPitch':  [0.00, 50.00, -85.00], 'LHipRoll':  [0.00, 0.00, 0.00],
			      'LHipPitch':  [0.00, 0.00, 0.00], 'LKneePitch':  [0.00, 0.00, -100.00], 'LAnklePitch':  [0.00, 0.00, -102.90], 'LAnkleRoll':  [0.00, 0.00, 0.00],
			      'RHipYawPitch':  [0.00, -50, -85.00], 'RHipRoll':  [0.00, 0, 0.00], 'RHipPitch':  [0.00, 0, 0.00], 'RKneePitch':  [0.00, 0, -100.00], 'RAnklePitch':  [0.00, 0, -102.90],
			      'RAnkleRoll':  [0.00, 0, 0.00], 'RShoulderPitch':  [0, -98, 100], 'RShoulderRoll':  [0, 0, 0], 'RElbowYaw':  [105, -15, 0], 'RElbowRoll':  [0, 0, 0], 'RWristYaw':  [55.95, 0, 0] , 'LWristYaw':  [55.95, 0.00, 0.00]}

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
        # YOUR CODE HERE
        T = np.identity(4)
        sine = np.sin(joint_angle)
        cosine = np.cos(joint_angle)
          
	# Apply corresponding rotation matrix
        if joint_name in xJoints:
	  Rx = np.array([[1, 0, 0, 0], [0, cosine, -sine, 0], [0, sine, cosine, 0], [0, 0, 0, 1]])
	  T = np.dot(T, Rx)
        if joint_name in yJoints:
	  Ry = np.array([[cosine, 0, sine, 0], [0, 1, 0, 0], [-sine, 0, cosine, 0], [0, 0, 0, 1]])
	  T = np.dot(T, Ry)
	if joint_name in zJoints:
	  Rz = np.array([[cosine, -sine, 0, 0], [sine, cosine, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
	  T = np.dot(T, Rz)
	  
        #Apply basis changes before rotation for the HipYawPitch joints
        if joint_name == 'RHipYawPitch':
	  #Change of basis matrix:
	  CoB = np.array([[1, 0, 0, 0], [0, np.cos(np.pi/4), -np.sin(np.pi/4), 0], [0, np.sin(np.pi/4), np.cos(np.pi/4), 0], [0, 0, 0, 1]])
	  #Rotation matrix:
	  Rz = np.array([[cosine, sine, 0, 0], [-sine, cosine, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
	  T = np.dot(T, CoB.T)
	  T = np.dot(T, Rz.T)	
	  T = np.dot(T, CoB)
	if joint_name == 'LHipYawPitch':
	  #Change of basis matrix:
	  CoB = np.array([[1, 0, 0, 0], [0, np.cos(np.pi/4), -np.sin(np.pi/4), 0], [0, np.sin(np.pi/4), np.cos(np.pi/4), 0], [0, 0, 0, 1]])
	  #Rotation matrix:
	  Rz = np.array([[cosine, sine, 0, 0], [-sine, cosine, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
	  T = np.dot(T, CoB)
	  T = np.dot(T, Rz)
	  T = np.dot(T, CoB.T)
	
	# Add translation
        T[:,-1][:-1] = self.jointOffsets[joint_name]
  
        return T

    def forward_kinematics(self, joints):
        '''forward kinematics

        :param joints: {joint_name: joint_angle}
        '''
        for chain_joints in self.chains.values():
            T = np.identity(4)
            for joint in chain_joints:
		if joint in wristJoints:
		  angle = 0
		else:
		  angle = joints[joint]
                Tl = self.local_trans(joint, angle)
                # YOUR CODE HERE
                T = np.dot(T, Tl)

                self.transforms[joint] = T

if __name__ == '__main__':
    agent = ForwardKinematicsAgent()
    agent.run()

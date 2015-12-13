'''In this exercise you need to implemente inverse kinematics for NAO's legs

* Tasks:
    1. solve inverse kinemtatics for NAO's legs by using analytical or numerical method.
       You may need documentation of NAO's leg:
       http://doc.aldebaran.com/2-1/family/nao_h25/joints_h25.html
       http://doc.aldebaran.com/2-1/family/nao_h25/links_h25.html
    2. use the results of inverse kinemtatics to control NAO's legs (in InverseKinematicsAgent.set_transforms)
       and test your inverse kinemtatics implementation.
'''


#using Efficient computation of the jacobian for robot manipulators (Orin and Schrader)
#and Introduction to Inverse Kinematics with Jacobian Transpose...

from forward_kinematics import ForwardKinematicsAgent
import numpy as np
from collections import deque

lambda_factor = 0.1
posMargin = 0.01
angleMargin = 1

class InverseKinematicsAgent(ForwardKinematicsAgent):
  
  
    def getPosition(self, joint_name, transform):
	
	#calculate axisOfRotation
	rotationMatrix = transform[:-1,:-1]
	#hipyawpitch rotation missing
	if joint_name in self.xJoints:
	  axisOfRotation = [1, 0, 0]
	if joint_name in self.yJoints:
	  axisOfRotation = [0, 1, 0]
	if joint_name in self.zJoints:
	  axisOfRotation = [0, 0, 1]
	if joint_name == 'LHipYawPitch':
	  axisOfRotation = [0, np.sin(np.pi/4), np.sin(np.pi/4)]
	axisOfRotation = np.dot(rotationMatrix, axisOfRotation)
	
	#get position
	coordinates = transform[:,-1][:-1]
	
	#return concatenation
	return (coordinates, axisOfRotation)
      
    def getEulerAngles(self, transform):
	theta_x = np.arctan2(transform[2,1], transform[2,2])
	theta_y = np.arctan2(-transform[2,0], np.sqrt((transform[2,1] * transform[2,1]) + (transform[2,2] * transform[2,2])))
	theta_z = np.arctan2(transform[0,1], transform[0,0])
	
	return [theta_x, theta_y, theta_z]
	
    def inverse_kinematics(self, effector_name, transform):
        '''solve the inverse kinematics

        :param str effector_name: name of end effector, e.g. LLeg, RLeg
        :param transform: 4x4 transform matrix
        :return: list of joint angles
        '''
        joint_angles = [] #thetas
        # YOUR CODE HERE
        
        #getting current angles
	for joint in self.chains[effector_name]:
	      joint_angles.append(self.perception.joint[joint])
	print joint_angles
	#target position
        t = self.getPosition(self.chains[effector_name][-1], transform)
        t = np.r_[t[0], self.getEulerAngles(transform)]
        
        #starting inversed kinematics loop
        while True:
	  #running forward_kinematics
	  transformations = [np.identity(4)]
	  p = []
	  for i, joint in enumerate(self.chains[effector_name]):
                Tl = self.local_trans(joint, joint_angles[i])
                transformations.append(np.dot(transformations[-1], Tl))
                p.append(self.getPosition(joint, transformations[-1]))
                
			   
	  #end effectors position
	  s = p[-1]
	  if(effector_name == 'Lleg'):
	    s[0] += np.dot(transformations[-1][:-1,:-1], [0,0,-self.mainLength['FootHeight']])
	  s = np.r_[s[0], self.getEulerAngles(transformations[-1])]
	  #get the error
	  e = t - s
	  #print s, p[-1]
	  #stop if close enough
	  if (np.allclose(e[:-3], 0, 1, posMargin) and np.allclose(e[-3:], 0, 1, angleMargin)):
	    break
	  
	  #print "error: ", e
	  
	  #calculating jacobian matrix
	  jcolumns = []
	  for (position, rotAxis) in p:
	    d_position = np.cross(rotAxis, s[0] - position)
	    jcolumns.append(np.r_[d_position, rotAxis])
	  
	  J = np.column_stack(jcolumns)
	  
	  
	  #calculating pseudoinverse
	  JI = np.linalg.pinv(J)
	  #calculating angle correction
	  d_theta = np.dot((lambda_factor * JI), np.asarray(e).T)
	  #adding angle correction
	  joint_angles += d_theta #turn matrix into vector
	  joint_angles = np.mod(joint_angles, 2*np.pi)
	print joint_angles
        return joint_angles

    def set_transforms(self, effector_name, transform):
        '''solve the inverse kinematics and control joints use the results
        '''
        # YOUR CODE HERE
        angles = self.inverse_kinematics(effector_name, transform)
        for i, joint in enumerate(self.chains[effector_name]):
	      self.target_joints[joint] = angles[i]
        #self.keyframes = ([], [], [])  # the result joint angles have to fill in

if __name__ == '__main__':
    agent = InverseKinematicsAgent()
    # test inverse kinematics
    T = np.identity(4)
    T[1, -1] = 50.0
    T[2, -1] = -260.
    print T
    agent.set_transforms('LLeg', T)
    agent.run()

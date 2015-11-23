'''In this exercise you need to implemente inverse kinematics for NAO's legs

* Tasks:
    1. solve inverse kinemtatics for NAO's legs by using analytical or numerical method.
       You may need documentation of NAO's leg:
       http://doc.aldebaran.com/2-1/family/nao_h25/joints_h25.html
       http://doc.aldebaran.com/2-1/family/nao_h25/links_h25.html
    2. use the results of inverse kinemtatics to control NAO's legs (in InverseKinematicsAgent.set_transforms)
       and test your inverse kinemtatics implementation.
'''


from forward_kinematics import ForwardKinematicsAgent
import numpy as np

lambda_factor = 0.001
margin = 0.05
class InverseKinematicsAgent(ForwardKinematicsAgent):
    def inverse_kinematics(self, effector_name, transform):
        '''solve the inverse kinematics

        :param str effector_name: name of end effector, e.g. LLeg, RLeg
        :param transform: 4x4 transform matrix
        :return: list of joint angles
        '''
        joint_angles = []
        # YOUR CODE HERE
        #getting current angles
	for joint in self.chains[effector_name]:
	      joint_angles.append(self.perception.joint[joint])
        #naivly applying 2d to 3d:
        #getting target  TODO: currently not caring about angle
        target = transform[:,-1]
        while True:
	  #running forward_kinematics
	  
	  T = [np.identity(4)]
	  for i, joint in enumerate(self.chains[effector_name]):
                Tl = self.local_trans(joint, joint_angles[i])
                T.append(np.dot(T[-1], Tl))
			   
	  if (np.allclose(target, T, 1, margin)):
	    break
	  print target - T
	  #get last column of final joint TODO: currently not caring about angle
	  Te = T[-1][:,-1]
	  #getting error towards target
	  e = target - Te
	  #getting coordinates for each joint
	  T = [m[:,-1] for m in T[:-1]]
	  #calculating jacobian matrix
	  J = Te - T
	  #filling last row with 1
	  J[-1,:] = 1
	  #calculating pseudoinverse
	  JI = np.dot(J,np.dot(np.asarray(J).T, J)) 
	  #calculating angle correction
	  d_theta = np.dot((lambda_factor * JI), np.asarray(e).T)
	  #adding angle correction
	  joint_angles += d_theta[0] #turn matrix into vector
	  
        return joint_angles

    def set_transforms(self, effector_name, transform):
        '''solve the inverse kinematics and control joints use the results
        '''
        # YOUR CODE HERE
        angles = self.inverse_kinematics(effector_name, transform)
        for i, joint in enumerate(self.chains[effector_name]):
	      self.perception.joint[joint] = angles[i]
        #self.keyframes = ([], [], [])  # the result joint angles have to fill in

if __name__ == '__main__':
    agent = InverseKinematicsAgent()
    # test inverse kinematics
    T = np.identity(4)
    T[-1, 1] = 0.05
    T[-1, 2] = 0.26
    agent.set_transforms('LLeg', T)
    agent.run()

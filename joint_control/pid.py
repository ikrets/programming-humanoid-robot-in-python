'''In this exercise you need to implement the PID controller for joints of robot.

* Task:
    1. complete the control function in PIDController with prediction
    2. adjust PID parameters for NAO in simulation

* Hints:
    1. the motor in simulation can simple modelled by angle(t) = angle(t-1) + speed * dt
    2. use self.y to buffer model prediction
'''

# add PYTHONPATH
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'introduction'))

import numpy as np
from collections import deque
from spark_agent import SparkAgent, JOINT_CMD_NAMES



class PIDController(object):
    '''a discretized PID controller, it controls an array of servos,
       e.g. input is an array and output is also an array
    '''
    def __init__(self, dt, size):
        '''
        @param dt: step time
        @param size: number of control values
        @param delay: delay in number of steps
        '''
        self.dt = dt
        self.e1 = np.zeros(size)
        self.e2 = np.zeros(size)
        # ADJUST PARAMETERS BELOW
        delay = 1
        self.Kp = 77
        self.Ki = 0
        self.Kd = 0
        self.t = deque([np.zeros(size)], maxlen=delay + 1) # queue for old targets
        self.y = deque([np.zeros(size)], maxlen=delay + 1) #queue for predictions (to calculate prediction error)
        self.u = deque([np.zeros(size)], maxlen=delay + 1) #queue for sent signals (to account for sent signals in prediction with delay > 1)
	self.test = np.zeros(size)
    def set_delay(self, delay):
        '''
        @param delay: delay in number of steps
        '''
        self.t = deque(self.t, delay + 1)
        while len(self.t) < delay + 1 :
	  self.t.appendleft(self.t[0])
        self.y = deque(self.y, delay + 1)
        while len(self.y) < delay + 1 :
	  self.y.appendleft(self.y[0])
        self.u = deque(self.u, delay + 1)
        while len(self.u) < delay + 1 :
	  self.u.appendleft(self.u[0])


    def control(self, target, sensor):
        '''apply PID control
        @param target: reference values
        @param sensor: current values from sensor
        @return control signal
        '''
        # prediction calculation
        prediction = sensor
        vOld = self.u.popleft() #ignore the oldest signal for prediction calculation
        for v in self.u: 
	  #print v, self.dt
	  prediction += (v * self.dt) #account for already sent signals
        self.u.appendleft(vOld)
        
        ''' # does not work!
        #prediction error calculation
        self.y.popleft() #delete oldest prediciton
        self.t.popleft() #delete oldest target
        predictionCorrection = 1
        
        if(self.y):
	  ep = self.t[0] - self.y[0] #get error prediction for this sensor data
	  ae = self.t[0] - sensor # get actual error for this sensor data
	  predictionCorrection = abs(ae - ep) * np.sign(ae)
	  ep[ep == 0] = 1
	  predictionCorrection /= ep
	  
        self.y.append(prediction) #queue prediction without error correction so it won't accumulate
        self.t.append(target)
        prediction *= predictionCorrection
        ''' 
        
        # speed calculation
        e = target - prediction
        
        #notice that while self.u is a list of vectors, u is just one vector
	u = self.u[-1] + (self.Kp + self.Ki * self.dt + self.Kd / self.dt) * e - (self.Kp + 2*self.Kd/self.dt) * self.e1 + (self.Kd/self.dt) * self.e2
	self.u.popleft() #delete oldest signal
	
	self.u.append(u) # queing sent signal (speed) for better prediction with delays > 1
	self.e2 = self.e1
	self.e1 = e
        return u


class PIDAgent(SparkAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(PIDAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.joint_names = JOINT_CMD_NAMES.keys()
        number_of_joints = len(self.joint_names)
        self.joint_controller = PIDController(dt=0.01, size=number_of_joints)
        self.target_joints = {k: 0 for k in self.joint_names}

    def think(self, perception):
        action = super(PIDAgent, self).think(perception)
        '''calculate control vector (speeds) from
        perception.joint:   current joints' positions (dict: joint_id -> position (current))
        self.target_joints: target positions (dict: joint_id -> position (target)) '''
        joint_angles = np.asarray(
            [perception.joint[joint_id]  for joint_id in JOINT_CMD_NAMES])
        target_angles = np.asarray([self.target_joints.get(joint_id, 
            perception.joint[joint_id]) for joint_id in JOINT_CMD_NAMES])
        u = self.joint_controller.control(target_angles, joint_angles)
        action.speed = dict(zip(JOINT_CMD_NAMES.iterkeys(), u))  # dict: joint_id -> speed
        return action


if __name__ == '__main__':
    agent = PIDAgent()
    agent.target_joints['HeadYaw'] = 1.0
    agent.run()

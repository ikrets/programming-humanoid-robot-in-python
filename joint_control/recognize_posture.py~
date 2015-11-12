'''In this exercise you need to use the learned classifier to recognize current posture of robot

* Tasks:
    1. load learned classifier in `PostureRecognitionAgent.__init__`
    2. recognize current posture in `PostureRecognitionAgent.recognize_posture`

* Hints:
    Let the robot execute different keyframes, and recognize these postures.

'''


from angle_interpolation import AngleInterpolationAgent
from keyframes import hello
from keyframes import leftBackToStand
import numpy as np
import pickle
from sklearn import svm
featureNames = ['AngleX', 'AngleY', 'LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch']
classes = ['Knee', 'Back', 'Right', 'Crouch', 'Sit', 'Belly', 'StandInit', 'Stand', 'HeadBack', 'Frog', 'Left']
ROBOT_POSE_CLF = 'robot_pose.pkl'
class PostureRecognitionAgent(AngleInterpolationAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(PostureRecognitionAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.posture = 'unknown'
        self.posture_classifier = pickle.load(open(ROBOT_POSE_CLF))  # LOAD YOUR CLASSIFIER

    def think(self, perception):
        self.posture = self.recognize_posture(perception)
        return super(PostureRecognitionAgent, self).think(perception)

    def recognize_posture(self, perception):
        posture = 'unknown'
        # YOUR CODE HERE
        data = np.empty(len(featureNames))
        data[0] = perception.imu[0]
        data[1] = perception.imu[1]
        i = 2
        for feature in featureNames[i:]:
	  data[i] = perception.joint[feature]
	  i += 1
        
        prediction = self.posture_classifier.predict(data)
        posture = classes[prediction[0]]
        print posture
        return posture

if __name__ == '__main__':
    agent = PostureRecognitionAgent()
    agent.keyframes = leftBackToStand()  # CHANGE DIFFERENT KEYFRAMES
    agent.run()

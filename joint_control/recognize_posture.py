'''In this exercise you need to use the learned classifier to recognize current posture of robot

* Tasks:
    1. load learned classifier in `PostureRecognitionAgent.__init__`
    2. recognize current posture in `PostureRecognitionAgent.recognize_posture`

* Hints:
    Let the robot execute different keyframes, and recognize these postures.

'''

import pickle
import numpy as np
from angle_interpolation import AngleInterpolationAgent
from keyframes import *


class PostureRecognitionAgent(AngleInterpolationAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(PostureRecognitionAgent, self).__init__(simspark_ip,
                                                      simspark_port, teamname,
                                                      player_id, sync_mode)
        with open('robot_pose.pkl', 'rb') as fp:
            self.posture_classifier = pickle.load(fp)
        with open('robot_pose_classes.pickle', 'rb') as fp:
            self.classes = pickle.load(fp)

        self.feature_names = ['LHipYawPitch', 'LHipRoll', 'LHipPitch',
                              'LKneePitch', 'RHipYawPitch', 'RHipRoll',
                              'RHipPitch', 'RKneePitch', 'AngleX', 'AngleY']
        self.postures = []

    def think(self, perception):
        self.postures.append(self.recognize_posture(perception))
        if len(self.postures) > 1 and self.postures[-1] != self.postures[-2]:
            print(self.postures[-2] + ' -> ' + self.postures[-1])
        return super(PostureRecognitionAgent, self).think(perception)

    def recognize_posture(self, perception):
        feature_vector = np.empty(len(self.feature_names), dtype=np.float32)
        for i in range(len(self.feature_names) - 2):
            feature_vector[i] = self.perception.joint[self.feature_names[i]]
        feature_vector[-2:] = self.perception.imu

        prediction = self.posture_classifier.predict(feature_vector[np.newaxis, :])
        return self.classes[np.squeeze(prediction)]


if __name__ == '__main__':
    agent = PostureRecognitionAgent()
    agent.set_keyframes(leftBellyToStand())
    agent.run()

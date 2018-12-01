'''In this exercise you need to implement an angle interploation function which makes NAO executes keyframe motion

* Tasks:
    1. complete the code in `AngleInterpolationAgent.angle_interpolation`,
       you are free to use splines interploation or Bezier interploation,
       but the keyframes provided are for Bezier curves, you can simply ignore some data for splines interploation,
       please refer data format below for details.
    2. try different keyframes from `keyframes` folder

* Keyframe data format:
    keyframe := (names, times, keys)
    names := [str, ...]  # list of joint names
    times := [[float, float, ...], [float, float, ...], ...]
    # times is a matrix of floats: Each line corresponding to a joint, and column element to a key.
    keys := [[float, [int, float, float], [int, float, float]], ...]
    # keys is a list of angles in radians or an array of arrays each containing [float angle, Handle1, Handle2],
    # where Handle is [int InterpolationType, float dTime, float dAngle] describing the handle offsets relative
    # to the angle and time of the point. The first Bezier param describes the handle that controls the curve
    # preceding the point, the second describes the curve following the point.
'''

import numpy as np
import pickle
from pid import PIDAgent
import keyframes
from bezier_interpolators import BezierInterpolators
from spark_agent import JOINT_CMD_NAMES

from spline_interpolators import SplineInterpolators


class AngleInterpolationAgent(PIDAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(AngleInterpolationAgent, self).__init__(simspark_ip,
                                                      simspark_port, teamname,
                                                      player_id, sync_mode)
        self.interpolators = None
        self.motion_start = None

        self.joints_log = []
        self.target_joints_log = []
        self.actions_log = []

        self.still = True
        self.in_motion = False

    def think(self, perception):
        if self.still:
            return super(AngleInterpolationAgent, self).think(perception)

        target_joints = self.angle_interpolation(perception)
        self.target_joints.update(target_joints)
        self.target_joints_log.append(self.target_joints.copy())
        self.joints_log.append(perception.joint.copy())

        result = super(AngleInterpolationAgent, self).think(perception)
        self.actions_log.append(
            dict(zip(JOINT_CMD_NAMES.iterkeys(), self.joint_controller.u)))

        return result

    def set_keyframes(self, keyframes, speed_factor=1):
        # convert keyframes to bezier sections
        self.keyframes = keyframes
        self.still = False
        self.in_motion = False
        self.speed_factor = speed_factor

    def angle_interpolation(self, perception):
        if not self.in_motion:
            self.motion_start = perception.time
            self.initial_joints = perception.joint.copy()
            self.in_motion = True
            self.interpolators = SplineInterpolators(self.keyframes,
                                                     self.initial_joints)

        result = self.interpolators.compute(
            perception.time - self.motion_start,
            speed_factor=self.speed_factor)
        if not result:
            with open('logs.pickle', 'wb') as fp:
                pickle.dump(
                    (self.joints_log, self.target_joints_log,
                     self.actions_log), fp)

            self.actions_log = []
            self.joints_log = []
            self.target_joints_log = []

            self.still = True
            self.in_motion = False

        return result


if __name__ == '__main__':
    agent = AngleInterpolationAgent()
    agent.set_keyframes(keyframes.hello())
    agent.run()

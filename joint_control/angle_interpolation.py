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
from keyframes import hello
from bezier_interpolators import BezierInterpolators


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
        self.start_delay = 5

        self.joints_log = []
        self.target_joints_log = []
        self.actions_log = []

    def think(self, perception):
        print(perception.joint['LElbowYaw'], perception.joint['LElbowRoll'])
        target_joints = self.angle_interpolation(perception)
        self.target_joints.update(target_joints)
        self.target_joints_log.append(self.target_joints.copy())
        self.joints_log.append(perception.joint.copy())

        result = super(AngleInterpolationAgent, self).think(perception)
        self.actions_log.append(self.joint_controller.u.copy())

        return result

    def set_keyframes(self, keyframes):
        # convert keyframes to bezier sections
        names, times, keys = keyframes
        self.interpolators = BezierInterpolators(names, times, keys)

    def angle_interpolation(self, perception):
        assert self.interpolators

        if not self.motion_start:
            self.motion_start = perception.time

        result = self.interpolators.compute(perception.time - self.motion_start)
        if not result:
            exit()
        return result


if __name__ == '__main__':
    np.seterr(all='raise')
    agent = AngleInterpolationAgent()
    agent.set_keyframes(hello())
    agent.start()

    agent.thread.join()
    with open('logs.pickle', 'wb') as fp:
        pickle.dump(
            (agent.joints_log, agent.target_joints_log, agent.actions_log), fp)

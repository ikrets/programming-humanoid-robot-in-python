'''In this exercise you need to put all code together to make the robot be able to stand up by its own.

* Task:
    complete the `StandingUpAgent.standing_up` function, e.g. call keyframe motion corresponds to current posture

'''

from recognize_posture import PostureRecognitionAgent
from keyframes import *
import numpy as np


class StandingUpAgent(PostureRecognitionAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(StandingUpAgent, self).__init__(simspark_ip,
                                              simspark_port, teamname,
                                              player_id, sync_mode)
        self.trying_to_stand_up_from = None
        self.previous_posture = 'Start'

    def think(self, perception):
        self.standing_up()
        return super(StandingUpAgent, self).think(perception)

    def standing_up(self):
        if not self.postures:
            return

        posture = self.postures[-1]
        if posture != self.previous_posture:
            print(self.previous_posture + ' -> ' + posture)

        # Do not stand up from those poses
        if posture in ('Crouch', 'Frog', 'Knee', 'Stand', 'StandInit'):
            self.previous_posture = posture
            return

        if self.trying_to_stand_up_from and self.still == True:
            self.trying_to_stand_up_from = None

        speed_factor = 1

        if not self.trying_to_stand_up_from:
            if posture == 'Belly':
                print('Trying to stand up from belly')
                self.set_keyframes(rightBellyToStand(),
                                   speed_factor=speed_factor)

            if posture == 'Back':
                print('Trying to stand up from back')
                self.set_keyframes(rightBackToStand(),
                                   speed_factor=speed_factor)

            if posture in ('Left', 'Right'):
                action = np.random.choice(
                    [leftBackToStand, rightBackToStand, leftBellyToStand,
                     rightBellyToStand])
                self.set_keyframes(action(),
                                   speed_factor=speed_factor)

            self.trying_to_stand_up_from = posture
        # else:
        #     if posture == 'Back' and self.trying_to_stand_up_from == 'Belly':
        #         print('Fell on the back while trying to stand up from belly. Standing up again')
        #         self.set_keyframes(rightBackToStand())
        #         self.trying_to_stand_up_from = posture
        #
        #     if posture == 'Belly' and self.trying_to_stand_up_from == 'Back':
        #         print('Fell on the belly while trying to stand up from back. Standing up again')
        #         self.set_keyframes(rightBellyToStand())
        #         self.trying_to_stand_up_from = posture

        self.previous_posture = posture


class TestStandingUpAgent(StandingUpAgent):
    '''this agent turns off all motor to falls down in fixed cycles
    '''

    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(TestStandingUpAgent, self).__init__(simspark_ip, simspark_port,
                                                  teamname, player_id,
                                                  sync_mode)
        self.stiffness_on_off_time = 0
        self.stiffness_on_cycle = 100000  # in seconds
        self.stiffness_off_cycle = 1  # in seconds

    def think(self, perception):
        action = super(TestStandingUpAgent, self).think(perception)
        time_now = perception.time
        if self.stiffness_on_off_time == 0:
            self.stiffness_on_off_time = time_now

        if time_now - self.stiffness_on_off_time < self.stiffness_off_cycle:
            action.stiffness = {j: 0 for j in
                                self.joint_names}  # turn off joints
        else:
            action.stiffness = {j: 10 for j in
                                self.joint_names}  # turn on joints
        if time_now - self.stiffness_on_off_time > self.stiffness_on_cycle + self.stiffness_off_cycle:
            self.stiffness_on_off_time = time_now

        return action


if __name__ == '__main__':
    agent = TestStandingUpAgent()
    agent.run()

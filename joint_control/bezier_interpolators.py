import numpy as np


class BezierInterpolators:
    def __init__(self, choregraph_export):
        self.names, times, keys = choregraph_export
        maximum_key_count = max(len(k) for k in keys)

        bezier_sections = np.zeros((len(keys), maximum_key_count - 1, 4, 2),
                                   dtype=np.float32)

        # pre section is the section from time 0 to the first keyframe
        # P2 and P3 of it are known here, P0 and P1 depend on the
        # initial joint values. They will be calculated in compute
        self.pre_section = np.empty((len(keys), 4, 2), dtype=np.float32)
        # initial time is always zero
        self.pre_section[:, 0] = 0
        # initial velocity is always zero. 0.1 is arbitrary and could be changed
        self.pre_section[:, 1] = [0.1, 0]

        self.key_count = np.array([len(k) for k in keys])

        for i in range(len(keys)):
            for j in range(len(keys[i]) - 1):
                # P0 and P4: (time, value) coordinates
                start_value = keys[i][j][0]
                end_value = keys[i][j + 1][0]
                start_time, end_time = times[i][j: j + 2]

                # first handle is P1 and second handle is P2
                # in the cubic bezier formula
                # choregraph stores them relative to P0 and P4
                first_handle = keys[i][j][2][1:3]
                second_handle = keys[i][j + 1][1][1:3]

                bezier_sections[i, j] = [
                    [start_time, start_value],
                    [first_handle[0] + start_time,
                     first_handle[1] + start_value],
                    [second_handle[0] + end_time,
                     second_handle[1] + end_value],
                    [end_time, end_value]
                ]

                if j == 0:
                    self.pre_section[i, 2:] = [
                        [keys[i][0][1][1] + start_time,
                         keys[i][0][1][2] + start_value],
                        [times[i][0], keys[i][0][0]]]

        self.times = np.zeros((len(times), maximum_key_count),
                              dtype=np.float32)
        for i in range(len(times)):
            self.times[i, :len(times[i])] = times[i]

        self.bezier_sections = bezier_sections

    def compute(self, t, initial_joints, speed_factor=1):
        t *= speed_factor
        initial_joints = np.array(
            [initial_joints[name] if name in initial_joints else 0 for name in
             self.names])
        in_pre_section = np.where(self.times[:, 0] > t)[0]

        finished = np.where(self.times[np.arange(
            len(self.times)), self.key_count - 1] < t)[0]

        sections = np.argmax(self.times > t, axis=1) - 1
        # correct last points of section
        last_points = self.times[
                          np.arange(len(self.times)), self.key_count - 1] == t
        sections[last_points] = self.key_count[last_points] - 2
        sections[finished] = 0

        finished = set(finished)

        # get P0, P1, P2, P3
        P = self.bezier_sections[np.arange(len(self.bezier_sections)), sections]

        if len(in_pre_section):
            concrete_pre_section = self.pre_section[in_pre_section]
            concrete_pre_section[:, 0:2, 1] += initial_joints[in_pre_section][:,
                                               np.newaxis]
            P[in_pre_section] = concrete_pre_section

        joint_numbers = np.arange(len(self.times))
        # find out to which i does t correspond
        i = (t - self.times[joint_numbers, sections]) / (
                self.times[joint_numbers, sections + 1] - self.times[
            joint_numbers, sections])
        i[in_pre_section] = t / self.times[in_pre_section, 0]
        i = i[:, np.newaxis]

        result = np.power(1 - i, 3) * P[:, 0]
        result += 3 * np.power(1 - i, 2) * i * P[:, 1]
        result += 3 * (1 - i) * np.power(i, 2) * P[:, 2]
        result += np.power(i, 3) * P[:, 3]

        return {self.names[i]: result[i, 1]
                for i
                in range(len(self.names)) if i not in finished}

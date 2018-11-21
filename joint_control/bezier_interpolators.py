import numpy as np


class BezierInterpolators:
    def __init__(self, names, times, keys):
        self.names = names
        maximum_key_count = max(len(k) for k in keys)

        bezier_sections = np.zeros((len(keys), maximum_key_count, 4, 2),
                                   dtype=np.float32)

        self.key_count = np.array([len(k) for k in keys]) + 1

        for i in range(len(keys)):
            for j in range(len(keys[i])):
                if j == 0:
                    # make implicit 0th key explicit
                    bezier_sections[i, j, 0:2] = [[0, 0], [0.1, 0]]
                else:
                    bezier_sections[i, j, 0:2] = [
                        [times[i][j - 1], keys[i][j - 1][0]],
                        keys[i][j - 1][2][1:3]]

                bezier_sections[i, j, 2:] = [keys[i][j][1][1:],
                                             [times[i][j],
                                              keys[i][j][0]]]

        self.times = np.zeros((len(times), maximum_key_count + 1),
                              dtype=np.float32)
        # make implicit 0th key explicit
        self.times[:, 0] = 0
        for i in range(len(times)):
            self.times[i, 1:len(times[i]) + 1] = times[i]

        self.bezier_sections = bezier_sections

    def compute(self, t):
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

        joint_numbers = np.arange(len(self.times))
        # find out to which i does t correspond
        i = (t - self.times[joint_numbers, sections]) / (
                self.times[joint_numbers, sections + 1] - self.times[
            joint_numbers, sections])
        i = i[:, np.newaxis]

        result = np.power(1 - i, 3) * P[:, 0]
        result += 3 * np.power(1 - i, 2) * i * P[:, 1]
        result += 3 * (1 - i) * np.power(i, 2) * P[:, 2]
        result += np.power(i, 3) * P[:, 3]

        return {self.names[i]: result[i, 1]
                for i
                in range(len(self.names)) if i not in finished}

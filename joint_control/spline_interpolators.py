import numpy as np


class SplineInterpolators:
    def __init__(self, choregraph_export, initial_values):
        self.names, times, keys = choregraph_export
        initial_values = np.array(
            [initial_values[name] if name in initial_values else 0 for name in
             self.names])

        max_key_count = max(len(k) for k in keys)
        self.spline_sections = np.empty((len(keys), max_key_count, 4),
                                        dtype=np.float32)
        self.times = np.empty((len(keys), max_key_count + 1), dtype=np.float32)
        self.times[:, 0] = 0

        spline_matrix = lambda t1, t2: np.array(
            [[t1 ** 3, t1 ** 2, t1, 1],
             [t2 ** 3, t2 ** 2, t2, 1],
             [3 * t1 ** 2, 2 * t1, 1, 0],
             [3 * t2 ** 2, 2 * t2, 1, 0]],
            dtype=np.float32)

        self.spline_coeffs = np.empty((len(keys), max_key_count + 1, 4),
                                      dtype=np.float32)
        self.key_count = np.array([len(k) for k in keys])
        for i in range(len(keys)):
            self.times[i, 1:len(times[i]) + 1] = times[i]
            # section from t=0 to first key added
            for j in range(len(keys[i])):
                y_0 = keys[i][j - 1][0] if j != 0 else initial_values[i]
                y_1 = keys[i][j][0]
                if j < len(keys[i]) - 1:
                    y_2 = keys[i][j + 1][0]

                ydot_0 = 0 if j == 0 else ydot_1
                if (y_2 - y_1) * (y_1 - y_0) < 0:
                    ydot_1 = 0
                else:
                    ydot_1 = (y_2 - 2 * y_1 + y_0) / 2 / (
                                self.times[i, j + 1] - self.times[i, j])

                section_matrix = spline_matrix(self.times[i, j],
                                               self.times[i, j + 1])
                self.spline_coeffs[i, j] = np.linalg.solve(
                    section_matrix,
                    [y_0, y_1, ydot_0,
                     ydot_1])

    def compute(self, t, speed_factor=1.):
        t *= speed_factor

        finished = np.where(self.times[np.arange(
            len(self.times)), self.key_count] < t)[0]

        sections = np.argmax(self.times > t, axis=1) - 1
        # correct last points of section
        last_points = self.times[
                          np.arange(len(self.times)), self.key_count] == t
        sections[last_points] = self.key_count[last_points] - 1
        sections[finished] = 0

        finished = set(finished)

        section_coeffs = self.spline_coeffs[
            np.arange(len(self.times)), sections]

        result = section_coeffs.dot([t ** 3, t ** 2, t, 1])

        return {self.names[i]: result[i]
                for i
                in range(len(self.names)) if i not in finished}

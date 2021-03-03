import numpy as np
import itertools
import pyflann

import matplotlib.pyplot as plt


"""
    This class represents a n-dimensional unit cube with a specific number of points embeded.
    Points are distributed uniformly in the initialization. A search can be made using the
    search_point function that returns the k (given) nearest neighbors of the input point.
"""


class Space:

    def __init__(self, low, high, points):

        self._low = np.array(low)
        self._high = np.array(high)
        self._range = self._high - self._low
        self._dimensions = len(low)
        self.__space = init_uniform_space([0] * self._dimensions,
                                          [1] * self._dimensions,
                                          points)
        self._flann = pyflann.FLANN()
        self.rebuild_flann()
        print(self.__space)

    def rebuild_flann(self):
        self._index = self._flann.build_index(self.__space, algorithm='kdtree')

    def search_point(self, point, k):
        p_in = self.import_point(point)
        search_res, _ = self._flann.nn_index(p_in, k)
        knns = self.__space[search_res]
        p_out = []
        for p in knns:
            # print('p ',p)
            p_out.append(self.export_point(p))

        if k == 1:
            p_out = [p_out]
        
        return knns[0], np.array(p_out)[0]

    def import_point(self, point):
        return (point - self._low) / self._range

    def export_point(self, point):
        return self._low + point * self._range

    def get_space(self):
        return self.__space

    def shape(self):
        return self.__space.shape

    def get_number_of_actions(self):
        return self.shape()[0]


class Discrete_space(Space):
    """
        Discrete action space with n actions (the integers in the range [0, n))
        0, 1, 2, ..., n-2, n-1
    """

    def __init__(self, n):  # n: the number of the discrete actions
        super().__init__([0], [n - 1], n)

    def export_point(self, point):
        return super().export_point(point).astype(int)


def init_uniform_space(low, high, points):
    dims = len(low)
    points_in_each_axis = round(points**(1 / dims))

    axis = []
    for i in range(dims):
        axis.append(list(np.linspace(low[i], high[i], points_in_each_axis)))
    print(axis)
    space = []
    for _ in itertools.product(*axis):
        space.append(list(_))

    return np.array(space)
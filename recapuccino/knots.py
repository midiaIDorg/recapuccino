from functools import partial
from math import inf

import numpy.typing as npt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer


def make_knots(xx: npt.NDArray, idx_thr: int, value_thr: float) -> npt.NDArray:
    """
    generate a list of x values at which nots should be placed for spline fitting
    :param xx: a list of x values like mz or retention time
    :param idx_thr: the minimum number of datapoints that should be placed into a segment fit
    :param value_thr: the minimum distance on the x axis between consecutive segments of the fit
    :return: a list of not placements
    """
    xx = np.sort(xx)
    knots = []
    prev_x = -inf
    prev_idx = -inf
    for idx, x in enumerate(xx):
        if (idx - prev_idx >= idx_thr) and (x - prev_x >= value_thr):
            knots.append(x)
            prev_idx = idx
            prev_x = x
    return np.array(knots).reshape(-1, 1)

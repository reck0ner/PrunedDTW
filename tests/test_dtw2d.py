import math
import pytest
import numpy as np
from dtaidistance import dtw_ndim


def test_distance1_a():
    s1 = np.array([[0, 0], [0, 1], [2, 1], [0, 1],  [0, 0]], dtype=np.double)
    s2 = np.array([[0, 0], [2, 1], [0, 1], [0, .5], [0, 0]], dtype=np.double)
    d1 = dtw_ndim.distance(s1, s2)
    d1p, paths = dtw_ndim.warping_paths(s1, s2)
    assert d1 == pytest.approx(d1p)


if __name__ == "__main__":
    test_distance1_a()

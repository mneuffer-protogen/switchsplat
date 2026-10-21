import numpy as np

from threedgrut.datasets.utils import compute_inscribed_radius


def test_inscribed_radius_uses_nearest_image_edge():
    assert compute_inscribed_radius(np.array([720, 720]), np.array([360, 360])) == 360.0
    assert compute_inscribed_radius(np.array([800, 600]), np.array([300, 250])) == 250.0

import numpy as np
from PIL import Image

from threedgrut.datasets.dataset_colmap import ColmapDataset


def test_valid_mask_loads_and_caches_per_resolution(tmp_path):
    path = tmp_path / "mask.png"
    Image.fromarray(np.array([[255, 0], [0, 255]], dtype=np.uint8)).save(path)

    dataset = ColmapDataset.__new__(ColmapDataset)
    dataset.valid_mask_path = str(path)
    dataset._valid_mask_cache = {}

    mask = dataset._get_valid_mask(2, 2)

    np.testing.assert_array_equal(mask, np.array([[True, False], [False, True]]))
    assert dataset._get_valid_mask(2, 2) is mask

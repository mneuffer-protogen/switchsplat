# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("ncore")

from threedgrut.datasets.dataset_colmap import (
    ColmapDataset,
    _novel_eval_metadata,
    _opencv_pinhole_intrinsics_from_colmap,
    _resize_image_folder,
    _test_frame_mask_from_file,
    derive_state_code_mapping,
    remap_novel_state_code,
)
from threedgrut.datasets.utils import collate_same_resolution, parse_frame_time_indices


def test_frame_filename_ids_are_raw_time_and_survive_collation():
    paths = [
        "camera1/frame_0001.png",
        "camera1/frame_0041.png",
        "camera2/frame_0041.png",
    ]

    time_indices = parse_frame_time_indices(paths)

    np.testing.assert_array_equal(time_indices, np.array([1, 41, 41], dtype=np.int64))
    samples = [{"data": np.zeros((1, 2, 3, 3), dtype=np.uint8), "time_idx": int(time_idx)} for time_idx in time_indices]
    assert collate_same_resolution(samples)["time_idx"].tolist() == [1, 41, 41]
    assert parse_frame_time_indices(["camera1/frame_0009_test.png"]).tolist() == [9]


def test_non_frame_filenames_do_not_require_time():
    assert parse_frame_time_indices(["train/r_0.png"]) is None


def test_explicit_test_frame_file_selects_paths_and_training_complement(tmp_path):
    (tmp_path / "test_frames.txt").write_text("camera1/frame_0001.png\ncamera2/frame_0009.png\n")
    image_names = [
        "camera1/frame_0001.png",
        "camera1/frame_0002.png",
        "camera2/frame_0009.png",
    ]

    test_mask = _test_frame_mask_from_file(str(tmp_path), "test_frames.txt", image_names)

    np.testing.assert_array_equal(test_mask, np.array([True, False, True]))
    np.testing.assert_array_equal(~test_mask, np.array([False, True, False]))


def test_extended_test_manifest_keeps_evaluation_kinds_separate(tmp_path):
    names = [
        "camera1/frame_0001.png",
        "camera1/frame_test_novel_view_001_0001.png",
        "camera1/frame_test_novel_combo_110_0001.png",
    ]
    (tmp_path / "test_frames.txt").write_text("\n".join(names))

    np.testing.assert_array_equal(
        _test_frame_mask_from_file(str(tmp_path), "test_frames.txt", names, "test"),
        [True, False, False],
    )
    np.testing.assert_array_equal(
        _test_frame_mask_from_file(str(tmp_path), "test_frames.txt", names, "novel_view"),
        [False, True, False],
    )
    np.testing.assert_array_equal(
        _test_frame_mask_from_file(str(tmp_path), "test_frames.txt", names, "novel_combo"),
        [False, False, True],
    )
    assert _novel_eval_metadata(names[0]) is None
    assert _novel_eval_metadata(names[1]) == ("novel_view", "001")
    assert _novel_eval_metadata(names[2]) == ("novel_combo", "110")


@pytest.mark.parametrize(
    ("slot_windows", "expected_mapping", "expected_code"),
    [
        ([], [], []),
        ([[152, 300]], [0], [1]),
        ([[152, 300], [372, 552]], [0, 1], [1, 0]),
        ([[152, 300], [372, 552], [660, 813], [310, 362]], [0, 1, 2, None], [1, 0, 1, 0]),
        (
            [[152, 300], [372, 552], [660, 813], [310, 362], [565, 623]],
            [0, 1, 2, None, None],
            [1, 0, 1, 0, 0],
        ),
    ],
)
def test_slot_ablation_state_codes_follow_timeline_semantics(slot_windows, expected_mapping, expected_code):
    canonical_windows = [[152, 300], [372, 552], [660, 813]]

    mapping = derive_state_code_mapping(slot_windows, canonical_windows)

    assert mapping == expected_mapping
    assert remap_novel_state_code("101", mapping) == expected_code


def test_state_code_mapping_rejects_missing_canonical_bit():
    with pytest.raises(ValueError, match="outside canonical code"):
        remap_novel_state_code("01", [2])


@pytest.mark.parametrize(
    "contents,match",
    [
        ("camera1/missing.png\n", "absent from COLMAP"),
        ("camera1/frame_0001.png\ncamera1/frame_0001.png\n", "duplicate"),
        ("\n", "empty"),
    ],
)
def test_explicit_test_frame_file_rejects_invalid_lists(tmp_path, contents, match):
    (tmp_path / "test_frames.txt").write_text(contents)

    with pytest.raises(ValueError, match=match):
        _test_frame_mask_from_file(
            str(tmp_path),
            "test_frames.txt",
            ["camera1/frame_0001.png"],
        )


@pytest.mark.parametrize(
    "model,params,expected_focal,expected_principal,expected_radial,expected_tangential",
    [
        (
            "SIMPLE_RADIAL",
            [100.0, 40.0, 45.0, 0.1],
            [50.0, 50.0],
            [20.0, 22.5],
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0],
        ),
        (
            "RADIAL",
            [100.0, 40.0, 45.0, 0.1, 0.2],
            [50.0, 50.0],
            [20.0, 22.5],
            [0.1, 0.2, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0],
        ),
        (
            "OPENCV",
            [100.0, 110.0, 40.0, 45.0, 0.1, 0.2, 0.01, 0.02],
            [50.0, 55.0],
            [20.0, 22.5],
            [0.1, 0.2, 0.0, 0.0, 0.0, 0.0],
            [0.01, 0.02],
        ),
        (
            "FULL_OPENCV",
            [100.0, 110.0, 40.0, 45.0, 0.1, 0.2, 0.01, 0.02, 0.3, 0.4, 0.5, 0.6],
            [50.0, 55.0],
            [20.0, 22.5],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.01, 0.02],
        ),
    ],
)
def test_opencv_pinhole_intrinsics_from_colmap(
    model,
    params,
    expected_focal,
    expected_principal,
    expected_radial,
    expected_tangential,
):
    focal, principal, radial, tangential, thin_prism = _opencv_pinhole_intrinsics_from_colmap(
        model, np.asarray(params), scaling_factor=2
    )

    np.testing.assert_allclose(focal, expected_focal)
    np.testing.assert_allclose(principal, expected_principal)
    np.testing.assert_allclose(radial, expected_radial)
    np.testing.assert_allclose(tangential, expected_tangential)
    np.testing.assert_allclose(thin_prism, np.zeros((4,), dtype=np.float32))


def test_opencv_pinhole_intrinsics_rejects_unsupported_model():
    with pytest.raises(ValueError, match="Unsupported distorted pinhole camera model"):
        _opencv_pinhole_intrinsics_from_colmap("FOV", np.zeros((5,)), scaling_factor=1)


def test_resize_image_folder_matches_gsplat_rounding(tmp_path):
    source = tmp_path / "images"
    target = tmp_path / "images_2_png"
    source.mkdir()
    Image.new("RGB", (7, 5), color=(10, 20, 30)).save(source / "frame.jpg")

    _resize_image_folder(str(source), str(target), factor=2)

    with Image.open(target / "frame.png") as image:
        assert image.size == (4, 2)


def test_preloaded_data_matches_decoder_and_survives_source_removal(tmp_path):
    image_path = tmp_path / "frame.png"
    mask_path = tmp_path / "frame_mask.png"
    image_data = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    mask_data = np.array([[0, 255, 0], [255, 255, 0]], dtype=np.uint8)
    Image.fromarray(image_data).save(image_path)
    Image.fromarray(mask_data).save(mask_path)

    dataset = ColmapDataset.__new__(ColmapDataset)
    dataset.image_paths = np.asarray([str(image_path)])
    dataset.mask_paths = np.asarray([str(mask_path)])
    dataset.fullcircle_distractor_mask_paths = np.asarray([str(tmp_path / "missing.png")])
    dataset.distractor_mask_white_is_valid = True
    dataset.valid_mask_path = ""
    dataset._valid_mask_cache = {}
    dataset.n_frames = 1
    dataset._preloaded_data = None

    expected_data, expected_mask = dataset._load_data_and_mask(0)
    dataset._preload_decoded_data()
    image_path.unlink()
    mask_path.unlink()

    cached_data, cached_mask = dataset._preloaded_data[0]
    assert torch.equal(cached_data, expected_data)
    assert torch.equal(cached_mask, expected_mask)

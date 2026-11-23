from unittest.mock import Mock

import numpy as np
import pytest
from concord.bands import BandArray


def test_band_array_stores_and_returns_bands():
    bands = BandArray()
    data = np.arange(6).reshape(2, 3)

    bands["C13"] = data

    assert bands["C13"] is data


def test_band_array_rejects_data_with_the_wrong_grid_shape():
    grid = Mock(shape=(2, 3))
    bands = BandArray(grid)

    with pytest.raises(
        ValueError,
        match=r"Data shape \(3, 2\) does not match grid shape \(2, 3\)",
    ):
        bands["C13"] = np.zeros((3, 2))


def test_valid_mask_combines_bands_and_erodes_around_missing_pixels():
    bands = BandArray()
    first = np.ones((5, 5))
    second = np.ones((5, 5))
    first[2, 2] = np.nan
    second[0, 0] = np.nan
    bands["first"] = first
    bands["second"] = second

    expected = np.array(
        [
            [False, False, True, True, True],
            [False, True, False, True, True],
            [True, False, False, False, True],
            [True, True, False, True, True],
            [True, True, True, True, True],
        ]
    )

    np.testing.assert_array_equal(bands.valid_mask, expected)


def test_valid_mask_requires_at_least_one_band():
    with pytest.raises(ValueError, match="No bands available"):
        _ = BandArray().valid_mask


def test_valid_mask_is_cached_after_first_access():
    bands = BandArray()
    bands["first"] = np.ones((3, 3))
    first_mask = bands.valid_mask
    bands["second"] = np.full((3, 3), np.nan)

    assert bands.valid_mask is first_mask
    assert first_mask.all()


def test_reproject_delegates_to_the_source_grid():
    source_grid = Mock()
    target_grid = Mock()
    reprojected = BandArray(target_grid)
    source_grid.reproject.return_value = reprojected
    bands = BandArray(source_grid)

    assert bands.reproject(target_grid) is reprojected
    source_grid.reproject.assert_called_once_with(bands, target_grid)


def test_reproject_requires_a_source_grid():
    with pytest.raises(ValueError, match="without a grid"):
        BandArray().reproject(Mock())

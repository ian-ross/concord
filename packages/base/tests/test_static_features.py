from itertools import islice
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest
from concord.grid import LONLAT_CRS, Grid, Swath
from concord.instruments import static_features as static_module
from concord.instruments.static_features import (
    NATURAL_EARTH,
    StaticFeatures,
    _project_line,
    _rasterize_grid_features,
    _rasterize_static_features,
    _rasterize_swath_features,
    _read_shapefile_lines,
)


def make_filter(features: np.ndarray) -> StaticFeatures:
    result = object.__new__(StaticFeatures)
    result.features = features
    return result


def test_static_features_removes_components_at_or_above_overlap_threshold():
    detections = np.array(
        [
            [True, True, False, False],
            [False, False, False, True],
            [False, False, False, True],
        ]
    )
    features = np.array(
        [
            [True, False, False, False],
            [False, False, False, True],
            [False, False, False, True],
        ]
    )

    result = make_filter(features)(detections, threshold=0.75)

    np.testing.assert_array_equal(
        result,
        [
            [True, True, False, False],
            [False, False, False, False],
            [False, False, False, False],
        ],
    )
    np.testing.assert_array_equal(
        detections,
        [
            [True, True, False, False],
            [False, False, False, True],
            [False, False, False, True],
        ],
    )


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_static_features_rejects_invalid_threshold(threshold):
    filter_features = make_filter(np.zeros((2, 2), dtype=bool))

    with pytest.raises(ValueError, match="between zero and one"):
        filter_features(np.zeros((2, 2), dtype=bool), threshold=threshold)


def test_static_features_validates_detection_shape():
    filter_features = make_filter(np.zeros((2, 2), dtype=bool))

    with pytest.raises(ValueError, match="two-dimensional"):
        filter_features(np.zeros(2, dtype=bool))
    with pytest.raises(ValueError, match="identical shapes"):
        filter_features(np.zeros((3, 2), dtype=bool))


def test_project_line_transforms_each_coordinate():
    transformer = Mock()
    transformer.transform.side_effect = lambda lon, lat: (lon + 100, lat - 100)

    result = _project_line(
        {"type": "LineString", "coordinates": [(1, 2), (3, 4)]}, transformer
    )

    assert result == {
        "type": "LineString",
        "coordinates": [(101, -98), (103, -96)],
    }


def test_project_line_rejects_other_geometry_types():
    with pytest.raises(
        ValueError, match="unsupported static feature geometry: Polygon"
    ):
        _project_line({"type": "Polygon", "coordinates": []}, Mock())


def test_rasterize_grid_features_uses_native_grid_coordinates():
    grid = Grid(
        "lonlat",
        crs=LONLAT_CRS,
        nx=5,
        ny=5,
        extent=(0.0, 0.0, 5.0, 5.0),
    )
    features = [
        {
            "type": "LineString",
            "coordinates": [(0.0, 2.5), (5.0, 2.5)],
        }
    ]

    result = _rasterize_grid_features(features, grid)

    assert result.dtype == bool
    assert result.shape == (5, 5)
    assert result[2].all()
    assert result.sum() == 5


def test_rasterize_grid_features_returns_empty_mask_without_coordinates():
    grid = Grid(
        "lonlat",
        crs=LONLAT_CRS,
        nx=3,
        ny=2,
        extent=(0.0, 0.0, 3.0, 2.0),
    )

    result = _rasterize_grid_features([{"type": "LineString", "coordinates": []}], grid)

    assert result.shape == (2, 3)
    assert not result.any()


def test_rasterize_static_features_dilates_grid_mask(monkeypatch):
    grid = Grid(
        "lonlat",
        crs=LONLAT_CRS,
        nx=3,
        ny=3,
        extent=(0.0, 0.0, 3.0, 3.0),
    )
    center = np.zeros((3, 3), dtype=bool)
    center[1, 1] = True
    rasterize = Mock(return_value=center)
    monkeypatch.setattr(static_module, "_rasterize_grid_features", rasterize)

    result = _rasterize_static_features([], grid)

    np.testing.assert_array_equal(
        result,
        [
            [False, True, False],
            [True, True, True],
            [False, True, False],
        ],
    )
    rasterize.assert_called_once_with([], grid)


def test_rasterize_swath_features_resamples_regular_mask(monkeypatch):
    swath = Swath(
        lats=np.array([[1.0, 1.0], [0.0, 0.0]]),
        lons=np.array([[0.0, 1.0], [0.0, 1.0]]),
    )
    area = Mock(
        shape=(2, 2),
        area_extent=(0.0, 0.0, 2.0, 2.0),
        width=2,
        height=2,
        crs=LONLAT_CRS,
    )
    area.geocentric_resolution.return_value = 10.0
    swath_definition = Mock()
    swath_definition.compute_optimal_bb_area.return_value = area
    swath_definition.geocentric_resolution.return_value = 20.0
    monkeypatch.setattr(
        static_module, "SwathDefinition", Mock(return_value=swath_definition)
    )
    monkeypatch.setattr(
        static_module.Transformer,
        "from_crs",
        Mock(return_value=Mock(transform=lambda lon, lat: (lon, lat))),
    )
    regular_mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    monkeypatch.setattr(static_module, "rasterize", Mock(return_value=regular_mask))
    resampled = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    resample = Mock(return_value=resampled)
    monkeypatch.setattr(static_module, "resample_nearest", resample)

    result = _rasterize_swath_features(
        [{"type": "LineString", "coordinates": [(0.0, 0.0), (1.0, 1.0)]}],
        swath,
    )

    np.testing.assert_array_equal(result, resampled.astype(bool))
    assert resample.call_args.kwargs["radius_of_influence"] == pytest.approx(
        np.sqrt(2.0) * 20.0
    )
    assert resample.call_args.kwargs["fill_value"] == 0


def test_read_shapefile_lines_reads_vendored_polyline_records():
    records = list(
        islice(
            _read_shapefile_lines(cast(Path, NATURAL_EARTH / "ne_10m_coastline.zip")),
            2,
        )
    )

    assert len(records) == 2
    assert all(record["type"] == "LineString" for record in records)
    assert all(len(record["coordinates"]) >= 2 for record in records)
    assert all(len(point) == 2 for record in records for point in record["coordinates"])

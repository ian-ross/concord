import numpy as np
from concord.outlines import mask_to_outlines
from shapely.geometry import shape


class RegularCells:
    shape: tuple[int, int] = (2, 3)

    def cell_corners(self):
        longitudes = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 3.0],
            ]
        )
        latitudes = np.array(
            [
                [2.0, 2.0, 2.0, 2.0],
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        return longitudes, latitudes


def test_empty_mask_returns_an_empty_feature_collection():
    result = mask_to_outlines(np.zeros((2, 3), dtype=bool), RegularCells())

    assert result == {"type": "FeatureCollection", "features": []}


def test_adjacent_pixels_are_merged_into_one_feature():
    mask = np.array([[True, True, False], [False, False, False]])

    result = mask_to_outlines(mask, RegularCells())

    assert result["type"] == "FeatureCollection"
    assert len(result["features"]) == 1
    feature = result["features"][0]
    assert feature["properties"] == {"pixel_count": 2}
    geometry = shape(feature["geometry"])
    assert geometry.geom_type == "Polygon"
    assert geometry.area == 2.0
    assert geometry.bounds == (0.0, 1.0, 2.0, 2.0)


def test_disconnected_components_become_separate_features():
    mask = np.array([[True, False, False], [False, False, True]])

    result = mask_to_outlines(mask, RegularCells())

    assert len(result["features"]) == 2
    assert [feature["properties"]["pixel_count"] for feature in result["features"]] == [
        1,
        1,
    ]
    assert [shape(feature["geometry"]).area for feature in result["features"]] == [
        1.0,
        1.0,
    ]


def test_diagonally_touching_pixels_use_eight_connectivity():
    mask = np.array([[True, False, False], [False, True, False]])

    result = mask_to_outlines(mask, RegularCells())

    assert len(result["features"]) == 1
    assert result["features"][0]["properties"] == {"pixel_count": 2}
    geometry = shape(result["features"][0]["geometry"])
    assert geometry.geom_type == "MultiPolygon"
    assert geometry.area == 2.0

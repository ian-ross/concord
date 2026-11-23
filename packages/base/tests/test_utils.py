from datetime import UTC, datetime

import numpy as np
import pytest
from concord.utils import Box, round_time, tstr


def test_tstr_formats_utc_timestamp():
    value = datetime(2026, 4, 5, 6, 7, 8, tzinfo=UTC)

    assert tstr(value) == "20260405T060708Z"


def test_round_time_uses_pandas_frequency_rules():
    value = datetime(2026, 4, 5, 6, 7, 31, tzinfo=UTC)

    assert round_time(value, "min") == datetime(2026, 4, 5, 6, 8, tzinfo=UTC)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"north": 20, "south": -91, "east": 10, "west": -10}, "south latitude"),
        ({"north": 91, "south": 0, "east": 10, "west": -10}, "north latitude"),
        ({"north": 10, "south": 10, "east": 20, "west": 0}, "below north"),
        ({"north": 10, "south": 0, "east": 20, "west": -181}, "west longitude"),
        ({"north": 10, "south": 0, "east": 181, "west": 0}, "east longitude"),
        ({"north": 10, "south": 0, "east": 20, "west": 20}, "nonzero longitude"),
    ],
)
def test_box_rejects_invalid_bounds(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Box(**kwargs)


def test_box_contains_scalar_points_including_edges():
    box = Box(north=20, south=10, east=40, west=30)

    assert box.contains(10, 30)
    assert box.contains(20, 40)
    assert box.contains(15, 35)
    assert not box.contains(9.9, 35)
    assert not box.contains(15, 40.1)
    assert box.longitude_width == 10
    assert not box.all_longitudes


def test_box_crosses_the_date_line():
    box = Box(north=20, south=10, east=-170, west=170)

    assert box.longitude_width == 20
    assert box.contains(15, 175)
    assert box.contains(15, -175)
    assert not box.contains(15, 0)


def test_box_can_cover_all_longitudes():
    box = Box(north=20, south=10, east=180, west=-180)

    assert box.all_longitudes
    assert box.longitude_width == 360
    assert box.contains(15, -50_000)
    assert not box.contains(30, 0)


def test_box_contains_numpy_arrays():
    box = Box(north=20, south=10, east=-170, west=170)
    latitudes = np.array([10, 15, 20, 30])
    longitudes = np.array([170, -175, 0, 175])

    result = box.contains(latitudes, longitudes)

    np.testing.assert_array_equal(result, [True, True, False, False])


def test_box_requires_matching_scalar_or_array_inputs():
    box = Box(north=20, south=10, east=40, west=30)

    with pytest.raises(ValueError, match="both be scalars or both be numpy arrays"):
        box.contains(np.array([15]), 35)

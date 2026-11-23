from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest
from concord.instruments.transits import (
    TransitFinder,
    _angular_difference,
    _brackets_root,
    _PositionCache,
)
from concord.utils import Box


class FunctionalOrbital:
    def __init__(self, position):
        self.position = position
        self.requested_times = []

    def get_lonlatalt(self, time):
        self.requested_times.append(time)
        return self.position(time)


class TLEs:
    def __init__(self, orbital):
        self.orbital = orbital
        self.calls = []

    def lookup(self, satellite, time):
        self.calls.append((satellite, time))
        return self.orbital


@pytest.mark.parametrize(
    ("search_step", "time_tolerance", "message"),
    [
        (timedelta(0), timedelta(milliseconds=10), "search_step must be positive"),
        (timedelta(seconds=1), timedelta(0), "time_tolerance must be positive"),
        (
            timedelta(milliseconds=10),
            timedelta(milliseconds=10),
            "greater than time_tolerance",
        ),
    ],
)
def test_transit_finder_validates_steps(search_step, time_tolerance, message):
    with pytest.raises(ValueError, match=message):
        TransitFinder(Mock(), search_step, time_tolerance)


def test_sample_times_includes_a_short_final_interval():
    finder = TransitFinder(Mock(), search_step=timedelta(seconds=4))
    start = datetime(2026, 1, 1, tzinfo=UTC)

    assert finder._sample_times(start, start + timedelta(seconds=10)) == [
        start,
        start + timedelta(seconds=4),
        start + timedelta(seconds=8),
        start + timedelta(seconds=10),
    ]


def test_deduplicate_times_sorts_and_merges_nearby_roots():
    finder = TransitFinder(Mock(), time_tolerance=timedelta(milliseconds=10))
    start = datetime(2026, 1, 1, tzinfo=UTC)

    result = finder._deduplicate_times(
        [
            start + timedelta(seconds=2),
            start,
            start + timedelta(milliseconds=15),
            start + timedelta(milliseconds=25),
        ]
    )

    assert result == [
        start,
        start + timedelta(milliseconds=25),
        start + timedelta(seconds=2),
    ]


def test_finds_ascending_latitude_transit_and_refines_boundaries():
    start = datetime(2026, 1, 1, tzinfo=UTC)

    def position(time):
        seconds = (time - start.replace(tzinfo=None)).total_seconds()
        return 0.0, seconds - 5.0, 800.0

    orbital = FunctionalOrbital(position)
    finder = TransitFinder(
        TLEs(orbital),  # type: ignore[arg-type]
        search_step=timedelta(seconds=4),
        time_tolerance=timedelta(milliseconds=1),
    )
    box = Box(north=10, south=0, east=1, west=-1)

    transits = list(finder("TEST", start, start + timedelta(seconds=20), box))

    assert len(transits) == 1
    transit = transits[0]
    assert transit.enter.time == pytest.approx(
        start + timedelta(seconds=5), abs=timedelta(milliseconds=1)
    )
    assert transit.leave.time == pytest.approx(
        start + timedelta(seconds=15), abs=timedelta(milliseconds=1)
    )
    assert transit.enter.lat == pytest.approx(0.0, abs=0.002)
    assert transit.leave.lat == pytest.approx(10.0, abs=0.002)
    assert transit.enter.lon == 0.0
    assert transit.enter.direction is TransitFinder.Direction.ASCENDING
    assert transit.leave.direction is TransitFinder.Direction.ASCENDING


def test_finds_transit_through_a_date_line_crossing_box():
    start = datetime(2026, 1, 1, tzinfo=UTC)

    def position(time):
        seconds = (time - start.replace(tzinfo=None)).total_seconds()
        longitude = (160.0 + 2.0 * seconds + 180.0) % 360.0 - 180.0
        return longitude, 0.0, 800.0

    finder = TransitFinder(
        TLEs(FunctionalOrbital(position)),  # type: ignore[arg-type]
        search_step=timedelta(seconds=4),
        time_tolerance=timedelta(milliseconds=1),
    )
    box = Box(north=5, south=-5, east=-170, west=170)

    (transit,) = list(finder("TEST", start, start + timedelta(seconds=20), box))

    assert transit.enter.time == pytest.approx(
        start + timedelta(seconds=5), abs=timedelta(milliseconds=1)
    )
    assert transit.leave.time == pytest.approx(
        start + timedelta(seconds=15), abs=timedelta(milliseconds=1)
    )
    assert transit.enter.lon == pytest.approx(170.0, abs=0.002)
    assert transit.leave.lon == pytest.approx(-170.0, abs=0.002)


def test_transit_is_clipped_to_the_requested_window():
    start = datetime(2026, 1, 1, tzinfo=UTC)

    def position(time):
        seconds = (time - start.replace(tzinfo=None)).total_seconds()
        return 0.0, seconds, 800.0

    finder = TransitFinder(
        TLEs(FunctionalOrbital(position)),  # type: ignore[arg-type]
        search_step=timedelta(seconds=3),
        time_tolerance=timedelta(milliseconds=1),
    )
    box = Box(north=20, south=-20, east=1, west=-1)

    (transit,) = list(finder("TEST", start, start + timedelta(seconds=10), box))

    assert transit.enter.time == start
    assert transit.leave.time == start + timedelta(seconds=10)


def test_transit_finder_rejects_empty_time_window():
    finder = TransitFinder(Mock())
    start = datetime(2026, 1, 1, tzinfo=UTC)
    box = Box(north=10, south=0, east=10, west=0)

    with pytest.raises(ValueError, match="end must be later than start"):
        list(finder("TEST", start, start, box))


def test_direction_detects_descending_motion():
    time = datetime(2026, 1, 1, tzinfo=UTC)
    positions = {
        time - timedelta(seconds=1): (0.0, 2.0, 0.0),
        time + timedelta(seconds=1): (0.0, 1.0, 0.0),
    }

    assert (
        TransitFinder.Direction.at_time(time, positions)  # type: ignore[arg-type]
        is TransitFinder.Direction.DESCENDING
    )


def test_position_cache_looks_up_each_time_once_and_passes_naive_time_to_orbital():
    time = datetime(2026, 1, 1, 12, tzinfo=UTC)
    orbital = FunctionalOrbital(lambda value: (1, 2, 3))
    tles = TLEs(orbital)
    positions = _PositionCache("TEST", tles)  # type: ignore[arg-type]

    assert positions[time] == (1.0, 2.0, 3.0)
    assert positions[time] == (1.0, 2.0, 3.0)
    assert tles.calls == [("TEST", time)]
    assert orbital.requested_times == [time.replace(tzinfo=None)]


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [(-1, 1, True), (1, -1, True), (0, 3, True), (1, 2, False), (-2, -1, False)],
)
def test_brackets_root(left, right, expected):
    assert _brackets_root(left, right) is expected


@pytest.mark.parametrize(
    ("value", "reference", "expected"),
    [(181, 179, 2), (-179, 179, 2), (179, -179, -2), (10, 20, -10)],
)
def test_angular_difference_uses_shortest_signed_distance(value, reference, expected):
    assert _angular_difference(value, reference) == expected

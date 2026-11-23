"""Find intervals when a satellite's nadir point is inside a lat/lon box.

The public :func:`boxtimes` function mirrors the useful part of OrbNav's
``boxtimes`` response, but propagates the orbit locally with pyorbital. It
never contacts the OrbNav service.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from timelines.apis.space_track import TLEDatabase
from ..utils import Box


DEFAULT_SEARCH_STEP = timedelta(seconds=30)
DEFAULT_TIME_TOLERANCE = timedelta(milliseconds=10)


@dataclass
class TransitFinder:
    tles: TLEDatabase
    search_step: timedelta = DEFAULT_SEARCH_STEP
    time_tolerance: timedelta = DEFAULT_TIME_TOLERANCE

    class Direction(str, Enum):
        ASCENDING = "A"
        DESCENDING = "D"

        @classmethod
        def at_time(
                cls, time: datetime, positions: _PositionCache
        ) -> TransitFinder.Direction:
            delta = timedelta(seconds=1)
            before_lat = positions[time - delta][1]
            after_lat = positions[time + delta][1]
            if after_lat >= before_lat:
                return TransitFinder.Direction.ASCENDING
            else:
                return TransitFinder.Direction.DESCENDING

    @dataclass
    class Event:
        time: datetime
        lat: float
        lon: float
        direction: TransitFinder.Direction

        @classmethod
        def at_time(
                cls, time: datetime, positions: _PositionCache
        ) -> TransitFinder.Event:
            lon, lat, _ = positions[time]
            direction = TransitFinder.Direction.at_time(time, positions)
            return cls(time, lat, lon, direction)

    @dataclass
    class Transit:
        enter: TransitFinder.Event
        leave: TransitFinder.Event

    def __post_init__(self) -> None:
        if self.search_step <= timedelta(0):
            raise ValueError("search_step must be positive")
        if self.time_tolerance <= timedelta(0):
            raise ValueError("time_tolerance must be positive")
        if self.search_step <= self.time_tolerance:
            raise ValueError("search_step must be greater than time_tolerance")

    def __call__(
            self,
            sat: str,
            start_utc: datetime,
            end_utc: datetime,
            box: Box,
    ) -> Generator[TransitFinder.Transit, None, None]:
        """Return nadir transits through a latitude/longitude box.

        ``sat`` is a satellite name known to pyorbital.

        All times are in UTC.

        Transits already in progress at ``start`` or still in progress at ``end``
        are clipped to the requested window.
        """
        if end_utc <= start_utc:
            raise ValueError("end must be later than start")

        positions = _PositionCache(sat, self.tles)

        samples = self._sample_times(start_utc, end_utc)
        boundaries = self._deduplicate_times(
            self._find_boundary_times(samples, positions, box)
        )
        cuts = [
            start_utc,
            *(time for time in boundaries if start_utc < time < end_utc),
            end_utc,
        ]

        for enter, leave in self._inside_intervals(cuts, positions, box):
            yield TransitFinder.Transit(
                TransitFinder.Event.at_time(enter, positions),
                TransitFinder.Event.at_time(leave, positions)
            )

    def _sample_times(self, start: datetime, end: datetime) -> list[datetime]:
        result = [start]
        while result[-1] < end:
            result.append(min(result[-1] + self.search_step, end))
        return result

    def _deduplicate_times(self, times: list[datetime]) -> list[datetime]:
        if not times:
            return []

        ordered = sorted(times)
        result = [ordered[0]]
        for time in ordered[1:]:
            if time - result[-1] > self.time_tolerance * 2:
                result.append(time)
        return result

    def _find_boundary_times(
            self,
            samples: list[datetime],
            positions: _PositionCache,
            box: Box
    ) -> list[datetime]:
        roots: list[datetime] = []

        for left_time, right_time in zip(samples, samples[1:]):
            left_lon, left_lat, _ = positions[left_time]
            right_lon, right_lat, _ = positions[right_time]

            for boundary_lat in (box.south, box.north):
                left_value = left_lat - boundary_lat
                right_value = right_lat - boundary_lat
                if _brackets_root(left_value, right_value):
                    roots.append(self._bisect_time(
                        left_time,
                        right_time,
                        left_value,
                        right_value,
                        lambda time, boundary=boundary_lat: (
                            positions[time][1] - boundary
                        )
                    ))

            if box.all_longitudes:
                continue

            unwrapped_right = left_lon + _angular_difference(right_lon, left_lon)
            low_lon, high_lon = sorted((left_lon, unwrapped_right))
            for boundary_lon in (box.west, box.east):
                first_turn = int((low_lon - boundary_lon) // 360.0) - 1
                last_turn = int((high_lon - boundary_lon) // 360.0) + 1
                for turn in range(first_turn, last_turn + 1):
                    target = boundary_lon + 360.0 * turn
                    if not low_lon <= target <= high_lon:
                        continue
                    left_value = left_lon - target
                    right_value = unwrapped_right - target
                    if not _brackets_root(left_value, right_value):
                        continue
                    roots.append(self._bisect_time(
                        left_time,
                        right_time,
                        left_value,
                        right_value,
                        lambda time, origin=left_lon, target_lon=target: (
                            origin
                            + _angular_difference(positions[time][0], origin)
                            - target_lon
                        )
                    ))

        return roots

    def _bisect_time(
            self,
            left_time: datetime,
            right_time: datetime,
            left_value: float,
            right_value: float,
            function,
    ) -> datetime:
        if left_value == 0.0:
            return left_time
        if right_value == 0.0:
            return right_time

        while right_time - left_time > self.time_tolerance:
            middle_time = left_time + (right_time - left_time) / 2
            middle_value = float(function(middle_time))
            if middle_value == 0.0:
                return middle_time
            if (left_value < 0.0) == (middle_value < 0.0):
                left_time, left_value = middle_time, middle_value
            else:
                right_time, right_value = middle_time, middle_value

        return left_time + (right_time - left_time) / 2

    def _inside_intervals(
            self,
            cuts: list[datetime],
            positions: _PositionCache,
            box: Box,
    ) -> list[tuple[datetime, datetime]]:
        intervals: list[tuple[datetime, datetime]] = []
        for left, right in zip(cuts, cuts[1:]):
            if right - left <= self.time_tolerance:
                continue
            middle = left + (right - left) / 2
            lon, lat, _ = positions[middle]
            if not box.contains(lat, lon):
                continue
            if intervals and left - intervals[-1][1] <= self.time_tolerance * 2:
                intervals[-1] = intervals[-1][0], right
            else:
                intervals.append((left, right))
        return intervals


class _PositionCache:
    def __init__(self, satellite: str, tles: TLEDatabase) -> None:
        self.satellite = satellite
        self.tles = tles
        self._positions: dict[datetime, tuple[float, float, float]] = {}

    def __getitem__(self, time: datetime) -> tuple[float, float, float]:
        if time not in self._positions:
            orbital = self.tles.lookup(self.satellite, time)
            lon, lat, alt = orbital.get_lonlatalt(time.replace(tzinfo=None))
            self._positions[time] = float(lon), float(lat), float(alt)
        return self._positions[time]


def _brackets_root(left: float, right: float) -> bool:
    return left == 0.0 or right == 0.0 or (left < 0.0 < right) or (right < 0.0 < left)


def _angular_difference(value: float, reference: float) -> float:
    return (value - reference + 180.0) % 360.0 - 180.0


__all__ = [
    "TransitFinder"
]

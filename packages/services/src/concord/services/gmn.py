# TODO: Rework this as needed for use in Concord.

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cached_property
from pathlib import Path

import httpx
import libarchive
import pyarrow.parquet as pq
from pydantic import BaseModel
from httpx_retries import Retry, RetryTransport

# Retry logic for GMN downloads.

retry = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[401, 404, 502],
    allowed_methods=['GET'],
    respect_retry_after_header=True
)
transport = RetryTransport(retry=retry)


BASE_URL = 'https://nimbus.physics.uwo.ca/filebrowser/api'


# QUESTION: Is this reusable anywhere else?

class _ChunkReader:
    """Expose an iterable of byte chunks through the readinto interface."""

    def __init__(self, chunks: Iterable[bytes]):
        self.chunks = iter(chunks)
        self.pending = memoryview(b'')

    def seekable(self) -> bool:
        return False

    def readinto(self, buffer: bytearray) -> int:
        while not self.pending:
            try:
                self.pending = memoryview(next(self.chunks))
            except StopIteration:
                return 0

        size = min(len(buffer), len(self.pending))
        buffer[:size] = self.pending[:size]
        self.pending = self.pending[size:]
        return size


class Run:
    def __init__(self, run: str):
        # e.g., CAC0B1_20261002_022749_to_20261002_130714
        self.run = run
        parts = re.split(r'[-_]', run)
        self.station = parts[0]
        self.start = datetime.strptime(parts[1] + '_' + parts[2], '%Y%m%d_%H%M%S').replace(tzinfo=UTC)
        self.end = datetime.strptime(parts[4] + '_' + parts[5], '%Y%m%d_%H%M%S').replace(tzinfo=UTC)

    @property
    def path(self) -> str:
        return f'/{self.station}/{self.run}'

    @property
    def time_label(self) -> str:
        return f'{self.start.strftime("%Y%m%dT%H%M%SZ")}_{self.end.strftime("%Y%m%dT%H%M%SZ")}'


@dataclass(frozen=True)
class CameraFrame:
    frame_id: int
    capture_time: datetime


def parse_frame_time(value: str) -> datetime:
    timestamp, separator, illumination = value.rpartition('_')
    if not separator or illumination not in {'d', 'n'}:
        raise ValueError(f'Invalid GMN frame time: {value!r}')
    return datetime.strptime(timestamp, '%Y%m%d_%H%M%S_%f').replace(tzinfo=UTC)


@dataclass(frozen=True)
class FlightTrackOverlayPoint:
    x: float
    y: float
    age_seconds: float
    waypoint: int


@dataclass(frozen=True)
class ContrailDetectionOverlay:
    frame_id: int
    instance_id: int
    confidence: float
    rings: tuple[tuple[tuple[float, float], ...], ...]
    track_id: int
    attributed: bool


@dataclass(frozen=True)
class RunDirectory:
    base_dir: Path
    run: Run

    @cached_property
    def run_dir(self) -> Path:
        return self.base_dir / self.run.station / self.run.run

    @cached_property
    def _name_pattern(self) -> str:
        # e.g., CAC0B1_20261002_022749_to_20261002_130714
        parts = re.split(r'[-_]', self.run.run)
        return rf'{parts[0]}_{parts[1]}[-_]{parts[2]}_to_{parts[4]}[-_]{parts[5]}'

    @cached_property
    def _date_pattern(self) -> str:
        # e.g., CAC0B1_20261002_022749
        parts = re.split(r'[-_]', self.run.run)
        return rf'{parts[0]}_{parts[1]}[-_]{parts[2]}'

    def _find_file(self, pattern: str) -> Path:
        for f in self.run_dir.iterdir():
            if re.search(pattern, f.name):
                return f

        # Search subdirectories for job_*_output/<NAME>_polygons.json, etc.,
        # starting with the most recent job_*_output directory (i.e., the
        # highest numeric job ID).
        for subdir in sorted(self.run_dir.glob('job_*_output'), reverse=True):
            for f in subdir.iterdir():
                if re.search(pattern, f.name):
                    return f
            for f in (subdir / 'output').iterdir():
                if re.search(pattern, f.name):
                    return f
        raise FileNotFoundError(f'No file matching {pattern} in {self.run_dir}')

    # TODO: Convert some of these to read the files and return useful data
    # structures?

    @cached_property
    def calibration(self) -> Path:
        return self._find_file('platepars_flux_recalibrated.json')

    @cached_property
    def video(self) -> Path:
        return self._find_file(f'{self._name_pattern}_frames_timelapse.mp4')

    @cached_property
    def frame_times(self) -> Path:
        return self._find_file(f'{self._name_pattern}_frametimes.json')

    @cached_property
    def polygons(self) -> Path:
        return self._find_file(f'{self._name_pattern}_polygons.json')

    @cached_property
    def associations(self) -> Path:
        return self._find_file(f'{self._date_pattern}_associations_final.csv')

    @cached_property
    def mask_data(self) -> Path:
        return self._find_file(f'{self._date_pattern}_mask_data.json')

    @cached_property
    def advection_raw_core(self) -> Path:
        return self._find_file(f'{self._date_pattern}_advection_raw_core.parquet')

    @cached_property
    def advection_raw_metadata(self) -> Path:
        return self._find_file(f'{self._date_pattern}_advection_raw_metadata.parquet')

    @cached_property
    def object_id_index(self) -> Path:
        return self._find_file('object_id_index.json')

    @cached_property
    def frames(self) -> tuple[CameraFrame, ...]:
        with self.frame_times.open(encoding='utf-8') as file:
            values = json.load(file)
        return tuple(sorted(
            (
                CameraFrame(int(frame_id), parse_frame_time(value))
                for frame_id, value in values.items()
            ),
            key=lambda frame: frame.frame_id,
        ))

    def nearest_frame(self, target: datetime) -> CameraFrame:
        if target.tzinfo is None:
            target = target.replace(tzinfo=UTC)
        else:
            target = target.astimezone(UTC)
        if not self.frames:
            raise ValueError(f'No frame times found for {self.run.run}.')
        return min(
            self.frames,
            key=lambda frame: (abs(frame.capture_time - target), frame.capture_time),
        )

    def camera_mask_times(self, flight_id: str) -> tuple[datetime, ...]:
        with self.mask_data.open(encoding='utf-8') as file:
            mask_data = json.load(file)
        frame_values = mask_data.get(flight_id, {}).get(self.run.station, {})
        return tuple(sorted(
            datetime.strptime(value, '%Y-%m-%d %H:%M:%S').replace(tzinfo=UTC)
            for value in frame_values
        ))

    def attribution_frame(
        self,
        flight_id: str,
        attribution_time: datetime,
    ) -> CameraFrame:
        mask_times = self.camera_mask_times(flight_id)
        if not mask_times:
            raise ValueError(
                f'No camera mask data for flight {flight_id} in {self.run.run}.'
            )
        if attribution_time.tzinfo is None:
            attribution_time = attribution_time.replace(tzinfo=UTC)
        else:
            attribution_time = attribution_time.astimezone(UTC)
        mask_time = min(
            mask_times,
            key=lambda value: (abs(value - attribution_time), value),
        )
        return self.nearest_frame(mask_time)

    def flight_track_overlay_data(
        self,
        flight_id: str,
        frame: CameraFrame,
    ) -> list[FlightTrackOverlayPoint]:

        metadata = pq.read_table(
            self.advection_raw_metadata,
            columns=['flight_idx'],
            filters=[('flight_id', '=', flight_id)],
        )
        if metadata.num_rows != 1:
            raise ValueError(
                f'Expected one advection metadata row for flight {flight_id} in '
                f'{self.run.run}, found {metadata.num_rows}.'
            )
        flight_index = metadata.column('flight_idx')[0].as_py()
        table = pq.read_table(
            self.advection_raw_core,
            columns=['x', 'y', 'age', 'waypoint'],
            filters=[
                ('flight_idx', '=', flight_index),
                ('timestamp', '=', frame.capture_time),
            ],
        )
        result = []
        for row in table.to_pylist():
            x = float(row['x'])
            y = float(row['y'])
            if not math.isfinite(x) or not math.isfinite(y):
                continue
            age = row['age']
            age_seconds = float(age.total_seconds())
            result.append(FlightTrackOverlayPoint(
                x=x,
                y=y,
                age_seconds=age_seconds,
                waypoint=int(row['waypoint']),
            ))
        return sorted(result, key=lambda point: point.age_seconds, reverse=True)

    def contrail_detection_overlay_data(
        self,
        frame: CameraFrame,
        flight_id: str | None = None,
    ) -> list[ContrailDetectionOverlay]:
        with self.polygons.open(encoding='utf-8') as file:
            polygons = json.load(file)
        ownership: dict[str, list[str]] = {}
        if flight_id is not None:
            with self.object_id_index.open(encoding='utf-8') as file:
                ownership = json.load(file)

        result = []
        for polygon in polygons:
            if int(polygon['f']) != frame.frame_id:
                continue
            instance_id = int(polygon['i'])
            rings = tuple(
                tuple((float(x), float(y)) for x, y in ring)
                for ring in polygon['p']
            )
            attributed = False
            if flight_id is not None:
                attributed = any(
                    flight_id in ownership.get(
                        str(frame.frame_id * 10000 + instance_id * 100 + ring_index),
                        [],
                    )
                    for ring_index in range(len(rings))
                )
            result.append(ContrailDetectionOverlay(
                frame_id=frame.frame_id,
                instance_id=instance_id,
                confidence=float(polygon['c']),
                rings=rings,
                track_id=int(polygon['track_id']),
                attributed=attributed,
            ))
        return result


class Config(BaseModel):
    """Configuration for GMN access."""
    gmn_token: str


class CameraArchive:
    @classmethod
    def from_config(cls, config: Config) -> CameraArchive:
        """Create a CameraArchive from a Config object."""
        return cls(config.gmn_token)

    def __init__(self, token: str):
        self.token = token

    def ls(self, path: str = '/'):
        with httpx.Client() as client:
            client.headers['Authorization'] = 'Bearer ' + self.token
            d = client.get(
                f'{BASE_URL}/resources',
                params={'source': 'mit', 'path': path}
            ).json()
            return (
                [f['name'] for f in d.get('folders', [])],
                [(f['name'], f['size']) for f in d.get('files', [])]
            )

    def stations(self):
        return self.ls('/')[0]

    def runs(self, station: str) -> list[Run]:
        return [Run(r) for r in self.ls(f'/{station}')[0]]

    def select(self, path: str, pattern: str):
        """Select files in a folder matching a regex pattern."""
        _, files = self.ls(path)
        retval = [f for f in files if re.search(pattern, f[0])]
        return [path + '/' + f[0] for f in retval]

    # subdirs, files = ls(f'/CAC0B1/{runs[0]}')      # cache/, job_*_output/, mp4, json, ...

    # search by file name across all stations
    # -> [{'path': 'CAC0B1/.../..._frametimes.json', 'size': 438460, ...}, ...]
    def search(self, query: str):
        with httpx.Client() as client:
            client.headers['Authorization'] = 'Bearer ' + self.token
            return client.get(
                f'{BASE_URL}/tools/search',
                params={'sources': 'mit', 'query': query}
            ).json()

    def get(self, path: str):
        """Get a file's contents as bytes."""
        with httpx.Client() as client:
            client.headers['Authorization'] = 'Bearer ' + self.token
            r = client.get(
                f'{BASE_URL}/resources/download',
                params={'source': 'mit', 'file': path}
            )
            r.raise_for_status()
            return r.content

    def download(self, path, dest: Path):
        """Download one file, resuming if a partial copy already exists."""
        with httpx.Client(transport=transport) as client:
            client.headers['Authorization'] = 'Bearer ' + self.token
            have = os.path.getsize(dest) if os.path.exists(dest) else 0
            hdr = {'Range': f'bytes={have}-'} if have else {}
            with client.stream(
                    'GET', f'{BASE_URL}/resources/download',
                    params={'source': 'mit', 'file': path},
                    headers=hdr
            ) as r:
                # already complete
                if r.status_code == 416:
                    return
                r.raise_for_status()
                with open(dest, 'ab' if r.status_code == 206 else 'wb') as fh:
                    for chunk in r.iter_bytes(1 << 20):
                        fh.write(chunk)

    def download_dir(self, path, dest: Path):
        """Download a ZIP archive and extract it into dest as it arrives."""
        with httpx.Client(transport=transport) as client:
            client.headers["Authorization"] = "Bearer " + self.token
            with client.stream(
                    'GET', f"{BASE_URL}/resources/download",
                    params={"source": "mit", "file": path, "algo": "zip"}
            ) as r:
                r.raise_for_status()

                dest = Path(dest)
                dest.mkdir(parents=True, exist_ok=True)
                dest_root = dest.resolve()

                reader = _ChunkReader(r.iter_bytes(1 << 20))
                with libarchive.stream_reader(
                        reader, format_name="zip", block_size=1 << 20
                ) as archive:
                    for entry in archive:
                        member = entry.pathname
                        target = (dest_root / member).resolve()
                        try:
                            target.relative_to(dest_root)
                        except ValueError as exc:
                            raise ValueError(
                                f"ZIP member escapes destination: {member!r}"
                            ) from exc

                        if entry.isdir:
                            target.mkdir(parents=True, exist_ok=True)
                            continue
                        if not entry.isfile:
                            raise ValueError(
                                f"Unsupported ZIP member type: {member!r}"
                            )

                        target.parent.mkdir(parents=True, exist_ok=True)
                        with open(target, "wb") as fh:
                            for chunk in entry.get_blocks(1 << 20):
                                fh.write(chunk)

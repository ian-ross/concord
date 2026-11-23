import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from concord.services.gmn import Run, RunDirectory


def make_run_directory(tmp_path: Path) -> RunDirectory:
    run = Run("CAM001_20261008_000000_to_20261008_010000")
    run_path = tmp_path / run.station / run.run
    output_path = run_path / "job_1_output" / "output"
    output_path.mkdir(parents=True)
    (run_path / "CAM001_20261008-000000_to_20261008-010000_frames_timelapse.mp4").touch()
    (run_path / "CAM001_20261008-000000_to_20261008-010000_frametimes.json").write_text(
        json.dumps({
            "0": "20261008_000000_000_d",
            "1": "20261008_000005_250_n",
        })
    )
    (run_path / "job_1_output" / "CAM001_20261008-000000_to_20261008-010000_polygons.json").write_text(
        json.dumps([
            {
                "f": 1,
                "t": 1791417605,
                "c": 0.8,
                "i": 2,
                "p": [[[10, 20], [30, 20], [20, 40]]],
                "b": [10, 20, 30, 40],
                "track_id": 7,
            },
        ])
    )
    (output_path / "CAM001_20261008_000000_mask_data.json").write_text(
        json.dumps({
            "flight-1": {
                "CAM001": {
                    "2026-10-08 00:00:05": {"lat": [], "lon": [], "ht": []},
                },
            },
        })
    )
    (output_path / "object_id_index.json").write_text(
        json.dumps({"10200": ["flight-1"]})
    )
    pq.write_table(
        pa.table({"flight_idx": [3], "flight_id": ["flight-1"]}),
        output_path / "CAM001_20261008_000000_advection_raw_metadata.parquet",
    )
    pq.write_table(
        pa.table({
            "timestamp": pa.array([
                datetime(2026, 10, 8, 0, 0, 5, 250000, tzinfo=UTC),
                datetime(2026, 10, 8, 0, 0, 5, 250000, tzinfo=UTC),
            ], type=pa.timestamp("ns", tz="UTC")),
            "x": [20.0, 10.0],
            "y": [40.0, 30.0],
            "age": pa.array(
                [timedelta(seconds=5), timedelta(seconds=10)],
                type=pa.duration("ns"),
            ),
            "waypoint": pa.array([2, 1], type=pa.uint32()),
            "flight_idx": pa.array([3, 3], type=pa.uint32()),
        }),
        output_path / "CAM001_20261008_000000_advection_raw_core.parquet",
    )
    return RunDirectory(tmp_path, run)


def test_run_directory_reads_camera_overlay_data(tmp_path: Path):
    run_directory = make_run_directory(tmp_path)

    frame = run_directory.attribution_frame(
        "flight-1",
        datetime(2026, 10, 8, 0, 0, 4, tzinfo=UTC),
    )
    track = run_directory.flight_track_overlay_data("flight-1", frame)
    detections = run_directory.contrail_detection_overlay_data(frame, "flight-1")

    assert frame.frame_id == 1
    assert frame.capture_time == datetime(
        2026, 10, 8, 0, 0, 5, 250000, tzinfo=UTC
    )
    assert [point.age_seconds for point in track] == [10.0, 5.0]
    assert [(point.x, point.y) for point in track] == [(10.0, 30.0), (20.0, 40.0)]
    assert len(detections) == 1
    assert detections[0].instance_id == 2
    assert detections[0].attributed is True
    assert detections[0].rings[0][0] == (10.0, 20.0)

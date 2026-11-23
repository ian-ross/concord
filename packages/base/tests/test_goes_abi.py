from datetime import date, datetime, timedelta
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import numpy as np
import xarray as xr
from concord.instruments.goes_abi import data as goes_data
from concord.instruments.goes_abi.data import GOESFiles
from concord.instruments.goes_abi.grid import ABIGrid


def goes_name(product: str, time: datetime, suffix: str = "file") -> str:
    start_stamp = time.strftime("%Y%j%H%M%S")
    end_stamp = (time + timedelta(minutes=1)).strftime("%Y%j%H%M%S")
    return f"OR_{product}_G19_s{start_stamp}0_e{end_stamp}9_c{end_stamp}9_{suffix}.nc"


def test_goes_files_convert_paths_and_round_trip_through_dict():
    files = GOESFiles(
        cast(Path, "m.nc"),
        cast(Path, "11.nc"),
        cast(Path, "13.nc"),
        cast(Path, "14.nc"),
        cast(Path, "15.nc"),
    )

    assert files == GOESFiles.from_dict(files.to_dict())
    assert all(
        isinstance(getattr(files, name), Path)
        for name in ("mcmipc", "c11", "c13", "c14", "c15")
    )


def test_load_brightness_temperatures_reads_mcmipc_bands(monkeypatch, tmp_path: Path):
    dataset = Mock()
    dataset.CMI_C11.to_numpy.return_value = np.full((2, 2), 11.0)
    dataset.CMI_C13.to_numpy.return_value = np.full((2, 2), 13.0)
    dataset.CMI_C14.to_numpy.return_value = np.full((2, 2), 14.0)
    dataset.CMI_C15.to_numpy.return_value = np.full((2, 2), 15.0)
    open_dataset = Mock(return_value=dataset)
    grid = Mock(shape=(2, 2))
    grid_factory = Mock(return_value=grid)
    monkeypatch.setattr(goes_data.xr, "open_dataset", open_dataset)
    monkeypatch.setattr(goes_data, "ABIGrid", grid_factory)
    files = GOESFiles(
        Path("remote/m.nc"),
        Path("11.nc"),
        Path("13.nc"),
        Path("14.nc"),
        Path("15.nc"),
    )

    result = files.load_brightness_temperatures(tmp_path)

    open_dataset.assert_called_once_with(tmp_path / "m.nc")
    grid_factory.assert_called_once_with(dataset)
    assert result.grid is grid
    for band, value in (("C11", 11), ("C13", 13), ("C14", 14), ("C15", 15)):
        np.testing.assert_array_equal(result[band], np.full((2, 2), value))


def test_list_files_matches_products_with_the_same_scan_start(monkeypatch):
    requested = date(2026, 1, 2)
    matching_time = datetime(2026, 1, 2, 3, 4, 5)
    missing_time = datetime(2026, 1, 2, 4, 5, 6)

    class FakeS3:
        def glob(self, pattern):
            channel = next(ch for ch in ("C11", "C13", "C14", "C15") if ch in pattern)
            hour = int(pattern.split("/")[-2])
            if hour != matching_time.hour:
                return []
            return [
                f"bucket/{goes_name(f'ABI-L1b-RadC-M6{channel}', matching_time)}",
                f"bucket/{goes_name(f'ABI-L1b-RadC-M6{channel}', missing_time, 'ignored')}.tmp",
            ]

        def ls(self, prefix):
            hour = int(prefix.split("/")[-1])
            if hour == matching_time.hour:
                return [f"bucket/{goes_name('ABI-L2-MCMIPC-M6', matching_time)}"]
            if hour == missing_time.hour:
                return [f"bucket/{goes_name('ABI-L2-MCMIPC-M6', missing_time)}"]
            return []

    monkeypatch.setattr(goes_data, "s3", FakeS3())

    result = goes_data._list_files(requested)

    assert list(result) == [matching_time]
    files = result[matching_time]
    assert files.mcmipc.name == goes_name("ABI-L2-MCMIPC-M6", matching_time)
    assert files.c11.name == goes_name("ABI-L1b-RadC-M6C11", matching_time)
    assert files.c15.name == goes_name("ABI-L1b-RadC-M6C15", matching_time)


def test_list_files_writes_then_reads_json_cache(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    timestamp = datetime(2026, 1, 2, 3, 4, 5)
    expected = {
        timestamp: GOESFiles(
            Path("m.nc"),
            Path("11.nc"),
            Path("13.nc"),
            Path("14.nc"),
            Path("15.nc"),
        )
    }
    list_remote = Mock(return_value=expected)
    monkeypatch.setattr(goes_data, "_list_files", list_remote)

    first = goes_data.list_files(date(2026, 1, 2))
    list_remote.return_value = {}
    second = goes_data.list_files(date(2026, 1, 2))

    assert first == expected
    assert second == expected
    list_remote.assert_called_once_with(date(2026, 1, 2))
    assert (tmp_path / "all_files_cache.json").exists()


def test_abi_grid_uses_projection_metadata_and_scan_angles():
    height = 35_786_023.0
    projection = xr.DataArray(
        0,
        attrs={
            "grid_mapping_name": "geostationary",
            "perspective_point_height": height,
            "longitude_of_projection_origin": -75.0,
            "sweep_angle_axis": "x",
            "semi_major_axis": 6_378_137.0,
            "semi_minor_axis": 6_356_752.31414,
        },
    )
    dataset = xr.Dataset(
        data_vars={"goes_imager_projection": projection},
        coords={"x": [-0.1, 0.1], "y": [0.05, -0.05]},
    )

    grid = ABIGrid(dataset)

    assert grid.shape == (2, 2)
    assert grid.extent == (-0.1 * height, -0.05 * height, 0.1 * height, 0.05 * height)
    assert grid.radius_of_influence == 100_000
    assert grid.crs.to_cf()["grid_mapping_name"] == "geostationary"

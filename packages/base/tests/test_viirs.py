from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest
from concord.grid import Swath
from concord.instruments.viirs import data as viirs_data
from concord.instruments.viirs.data import VIIRSFiles, _fill_along_track


class Variable:
    def __init__(self, values, *, valid_max=None):
        self.values = np.asarray(values)
        self.valid_max = valid_max

    def __getitem__(self, key):
        return self.values[key]


class NetCDFDataset:
    def __init__(self, variables):
        self.groups = {"observation_data": SimpleNamespace(variables=variables)}
        self.auto_maskandscale = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def set_auto_maskandscale(self, value):
        self.auto_maskandscale = value


def test_viirs_files_convert_paths_serialize_and_parse_time(tmp_path: Path):
    files = VIIRSFiles(
        "Suomi-NPP",
        cast(Path, "VNP02MOD.A2024001.1718.002.example.nc"),
        cast(Path, "VNP03MOD.A2024001.1718.002.example.nc"),
    )

    restored = VIIRSFiles.from_dict(files.to_dict(), tmp_path)

    assert files.time == datetime(2024, 1, 1, 17, 18, tzinfo=UTC)
    assert restored.platform == "Suomi-NPP"
    assert restored.data == tmp_path / files.data.name
    assert restored.navigation == tmp_path / files.navigation.name


def test_fill_along_track_interpolates_and_fills_empty_columns():
    values = np.array(
        [
            [np.nan, 1.0, 5.0],
            [np.nan, np.nan, 6.0],
            [np.nan, 3.0, 7.0],
        ]
    )

    result = _fill_along_track(values)

    np.testing.assert_allclose(
        result,
        [
            [4.4, 1.0, 5.0],
            [4.4, 2.0, 6.0],
            [4.4, 3.0, 7.0],
        ],
    )
    assert np.isnan(values[0, 0])


def test_valid_mask_uses_m14_valid_max(monkeypatch, tmp_path: Path):
    raw = np.array([[0, 2, 3], [1, 4, 2]])
    dataset = NetCDFDataset({"M14": Variable(raw, valid_max=2)})
    dataset_factory = Mock(return_value=dataset)
    monkeypatch.setattr(viirs_data.nc4, "Dataset", dataset_factory)
    files = VIIRSFiles("Suomi-NPP", tmp_path / "data.nc", tmp_path / "nav.nc")

    result = files.valid_mask

    np.testing.assert_array_equal(result, [[True, True, False], [True, False, True]])
    assert dataset.auto_maskandscale is False
    dataset_factory.assert_called_once_with(files.data)


def test_load_brightness_temperatures_applies_luts_masks_and_fills(
    monkeypatch, tmp_path: Path
):
    raw = np.array([[0, 1], [2, 99], [1, 2]])
    variables = {}
    for index, band in enumerate(("M14", "M15", "M16")):
        variables[band] = Variable(raw, valid_max=2)
        variables[f"{band}_brightness_temperature_lut"] = Variable(
            np.arange(100, dtype=np.float32) + 100 * index
        )
    dataset = NetCDFDataset(variables)
    monkeypatch.setattr(viirs_data.nc4, "Dataset", Mock(return_value=dataset))
    files = VIIRSFiles("Suomi-NPP", tmp_path / "data.nc", tmp_path / "nav.nc")

    result = files.load_brightness_temperatures()

    assert result.grid is None
    for index, band in enumerate(("M14", "M15", "M16")):
        offset = 100 * index
        np.testing.assert_allclose(
            result[band],
            [
                [offset + 0, offset + 1],
                [offset + 2, offset + 1.5],
                [offset + 1, offset + 2],
            ],
        )
        assert result[band].dtype == np.float32
    assert dataset.auto_maskandscale is False


def test_load_navigation_and_shape_are_cached(monkeypatch, tmp_path: Path):
    longitude = Mock()
    longitude.to_numpy.return_value = np.array([[10.0, 11.0], [10.0, 11.0]])
    latitude = Mock()
    latitude.to_numpy.return_value = np.array([[1.0, 1.0], [0.0, 0.0]])
    tree = SimpleNamespace(
        geolocation_data=SimpleNamespace(longitude=longitude, latitude=latitude)
    )
    open_datatree = Mock(return_value=tree)
    monkeypatch.setattr(viirs_data.xr, "open_datatree", open_datatree)
    files = VIIRSFiles("Suomi-NPP", tmp_path / "data.nc", tmp_path / "nav.nc")

    first = files.swath
    second = files.swath

    assert isinstance(first, Swath)
    assert second is first
    assert files.shape == (2, 2)
    open_datatree.assert_called_once_with(files.navigation)


def transit(start: datetime, end: datetime):
    return SimpleNamespace(
        enter=SimpleNamespace(time=start),
        leave=SimpleNamespace(time=end),
    )


def test_list_granules_queries_selected_products_and_pairs_sorted_results(
    tmp_path: Path,
):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = start + timedelta(hours=1)
    transit_finder = Mock(return_value=[transit(start, end)])
    responses = {
        "VNP02IMG": ["VNP02IMG.A2026001.0010.002.x.nc"],
        "VNP03IMG": ["VNP03IMG.A2026001.0010.002.x.nc"],
        "VNP02MOD": ["VNP02MOD.A2026001.0010.002.x.nc"],
        "VNP03MOD": ["VNP03MOD.A2026001.0010.002.x.nc"],
    }
    client = Mock()
    client.list_granules.side_effect = lambda product, start, end: responses[product]

    result = viirs_data.list_granules(
        "Suomi-NPP", start, end, tmp_path, transit_finder, client, bands="MI"
    )

    assert [(item.data.name, item.navigation.name) for item in result] == [
        (responses["VNP02IMG"][0], responses["VNP03IMG"][0]),
        (responses["VNP02MOD"][0], responses["VNP03MOD"][0]),
    ]
    transit_finder.assert_called_once_with("Suomi-NPP", start, end, viirs_data.EXTENT)
    assert client.list_granules.call_count == 4
    assert {call.args[0] for call in client.list_granules.call_args_list} == set(
        responses
    )
    assert all(
        call.kwargs == {"start": start, "end": end}
        for call in client.list_granules.call_args_list
    )


def test_list_granules_deduplicates_names_and_warns_on_mismatch(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = start + timedelta(hours=1)
    transit_finder = Mock(return_value=[transit(start, end), transit(start, end)])
    client = Mock()

    def list_names(product, start, end):
        if product == "VNP02MOD":
            return ["VNP02MOD.A2026001.0010.002.x.nc"]
        return []

    client.list_granules.side_effect = list_names

    result = viirs_data.list_granules(
        "Suomi-NPP", start, end, tmp_path, transit_finder, client, bands="M"
    )

    assert result == []
    assert "Mismatched data and navigation granules" in caplog.text


def test_list_files_for_date_collects_all_platforms(monkeypatch, tmp_path: Path):
    requested = date(2026, 1, 2)
    list_granules = Mock()

    def files_for(platform, *args, **kwargs):
        index = viirs_data.PLATFORMS.index(platform)
        minute = 10 + index
        code = {"NOAA-20": "VJ1", "NOAA-21": "VJ2", "Suomi-NPP": "VNP"}[platform]
        return [
            VIIRSFiles(
                platform,
                tmp_path / f"{code}02MOD.A2026002.00{minute}.002.x.nc",
                tmp_path / f"{code}03MOD.A2026002.00{minute}.002.x.nc",
            )
        ]

    list_granules.side_effect = files_for
    monkeypatch.setattr(viirs_data, "list_granules", list_granules)
    tles = Mock()
    client = Mock()

    result = viirs_data._list_files(requested, tmp_path, tles, client)

    assert len(result) == 3
    assert {item.platform for item in result.values()} == set(viirs_data.PLATFORMS)
    assert list_granules.call_count == 3
    for call in list_granules.call_args_list:
        assert call.kwargs == {"bands": "MI"}
        assert call.args[1] == datetime(2026, 1, 2, tzinfo=UTC)
        assert call.args[2] == datetime(2026, 1, 3, tzinfo=UTC)
        assert call.args[3] == tmp_path
        assert call.args[5] is client


def test_list_files_writes_then_reads_json_cache(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    timestamp = datetime(2026, 1, 2, 3, 4, tzinfo=UTC)
    expected = {
        timestamp: VIIRSFiles(
            "Suomi-NPP",
            data_dir / "VNP02MOD.A2026002.0304.002.x.nc",
            data_dir / "VNP03MOD.A2026002.0304.002.x.nc",
        )
    }
    list_remote = Mock(return_value=expected)
    monkeypatch.setattr(viirs_data, "_list_files", list_remote)
    tles = Mock()
    client = Mock()

    first = viirs_data.list_files(date(2026, 1, 2), data_dir, tles, client)
    list_remote.return_value = {}
    second = viirs_data.list_files(date(2026, 1, 2), data_dir, tles, client)

    assert first == expected
    assert second == expected
    list_remote.assert_called_once_with(date(2026, 1, 2), data_dir, tles, client)

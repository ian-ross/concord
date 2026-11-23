from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr
from concord import grid as grid_module
from concord.bands import BandArray
from concord.grid import (
    LONLAT_CRS,
    Grid,
    Swath,
    _coordinate_corners,
    _do_pixel_area_block,
    _unwrap_longitudes,
)
from pyproj import CRS


def make_grid(
    name: str = "grid",
    *,
    nx: int = 2,
    ny: int = 2,
    extent: tuple[float, float, float, float] = (0.0, 0.0, 2.0, 2.0),
    default_variables: list[str] | None = None,
) -> Grid:
    return Grid(
        name,
        crs=LONLAT_CRS,
        nx=nx,
        ny=ny,
        extent=extent,
        default_variables=default_variables,
    )


def matching_grids() -> tuple[Grid, Grid]:
    source = make_grid("source")
    target = make_grid("target")
    source._row_indexes[target] = np.array([[0, 0], [1, 1]])
    source._col_indexes[target] = np.array([[0, 1], [0, 1]])
    return source, target


@pytest.mark.parametrize(
    ("lats", "lons", "message"),
    [
        (np.zeros((2, 2)), np.zeros((2, 3)), "identical shapes"),
        (np.zeros(2), np.zeros(2), "two-dimensional"),
        (
            np.full((2, 2), np.nan),
            np.full((2, 2), np.nan),
            "at least one finite value",
        ),
    ],
)
def test_swath_rejects_invalid_coordinates(lats, lons, message):
    with pytest.raises(ValueError, match=message):
        Swath(lats=lats, lons=lons)


def test_swath_sets_shape_and_interpolates_cell_corners():
    swath = Swath(
        lats=np.array([[1.0, 1.0], [0.0, 0.0]]),
        lons=np.array([[10.0, 11.0], [10.0, 11.0]]),
    )

    longitudes, latitudes = swath.cell_corners()

    assert swath.shape == (2, 2)
    np.testing.assert_allclose(
        longitudes,
        [[9.5, 10.5, 11.5], [9.5, 10.5, 11.5], [9.5, 10.5, 11.5]],
    )
    np.testing.assert_allclose(
        latitudes,
        [[1.5, 1.5, 1.5], [0.5, 0.5, 0.5], [-0.5, -0.5, -0.5]],
    )


def test_swath_unwraps_cells_across_the_date_line():
    swath = Swath(
        lats=np.array([[1.0, 1.0], [0.0, 0.0]]),
        lons=np.array([[179.0, -179.0], [179.0, -179.0]]),
    )

    longitudes, _ = swath.cell_corners()

    np.testing.assert_allclose(longitudes[1], [178.0, 180.0, 182.0])


def test_coordinate_corners_repeats_singleton_dimensions():
    corners = _coordinate_corners(np.array([[4.0]]))

    np.testing.assert_array_equal(corners, np.full((2, 2), 4.0))


def test_unwrap_longitudes_keeps_nan_separated_runs_independent():
    values = np.array([[179.0, -179.0, np.nan, -179.0, 179.0]])

    result = _unwrap_longitudes(values)

    np.testing.assert_allclose(result[0, :2], [179.0, 181.0])
    assert np.isnan(result[0, 2])
    np.testing.assert_allclose(result[0, 3:], [-179.0, -181.0])


def test_grid_requires_an_area_or_complete_definition():
    with pytest.raises(ValueError, match="either area or all"):
        Grid("incomplete", crs=LONLAT_CRS, nx=2)


def test_grid_properties_and_cell_corners():
    grid = make_grid()

    assert grid.shape == (2, 2)
    assert grid.extent == (0.0, 0.0, 2.0, 2.0)
    assert grid.transform == (1.0, 0.0, 0.0, 0.0, -1.0, 2.0)
    longitudes, latitudes = grid.cell_corners()
    np.testing.assert_allclose(longitudes, [[0, 1, 2], [0, 1, 2], [0, 1, 2]])
    np.testing.assert_allclose(latitudes, [[2, 2, 2], [1, 1, 1], [0, 0, 0]])


def test_rasterio_profile_describes_the_grid():
    grid = make_grid(nx=3, ny=2, extent=(-3.0, -2.0, 3.0, 2.0))

    profile = grid.rasterio_profile(np.float32, channels=4)

    assert profile["driver"] == "GTiff"
    assert profile["dtype"] is np.float32
    assert profile["count"] == 4
    assert profile["width"] == 3
    assert profile["height"] == 2
    assert profile["crs"] == grid.crs
    assert tuple(profile["transform"])[:6] == (2.0, 0.0, -3.0, 0.0, -2.0, 2.0)


def test_coordinate_conversions_round_trip():
    grid = make_grid()
    lon, lat = -100.0, 35.0

    x, y = grid.to_lcc(lon, lat)
    roundtrip_lon, roundtrip_lat = grid.from_lcc(x, y)

    assert roundtrip_lon == pytest.approx(lon)
    assert roundtrip_lat == pytest.approx(lat)
    assert grid.from_lonlat(*grid.to_lonlat(lon, lat)) == pytest.approx((lon, lat))


def test_lookup_indexes_are_calculated_once_and_cached(monkeypatch):
    source = make_grid("source")
    target = make_grid("target")
    rows = np.array([[0, 0], [1, 1]])
    cols = np.array([[0, 1], [0, 1]])
    generate = Mock(return_value=(rows, cols))
    monkeypatch.setattr(
        grid_module, "generate_nearest_neighbour_linesample_arrays", generate
    )

    def use_calculation(keys, calculate):
        assert keys == (source, target)
        return calculate()

    source._index_cache.get = Mock(side_effect=use_calculation)

    assert source.row_indexes(target) is rows
    assert source.col_indexes(target) is cols
    assert source.row_indexes(target) is rows
    generate.assert_called_once_with(
        source.area,
        target.area,
        radius_of_influence=source.radius_of_influence,
        nprocs=grid_module.os.cpu_count() or 1,
    )
    source._index_cache.get.assert_called_once()


def test_reproject_rank_two_and_rank_three_arrays():
    source, target = matching_grids()
    rank_two = np.array([[1, 2], [3, 4]])
    rank_three = np.dstack([rank_two, rank_two + 10])

    np.testing.assert_array_equal(source.reproject(rank_two, target), rank_two)
    np.testing.assert_array_equal(source.reproject(rank_three, target), rank_three)


def test_reproject_data_array_sets_target_coordinates():
    source, target = matching_grids()
    data = xr.DataArray([[1, 2], [3, 4]], dims=["row", "column"])

    result = source.reproject(data, target)

    assert result.dims == ("y", "x")
    np.testing.assert_array_equal(result.values, data.values)
    np.testing.assert_allclose(result.longitude.values, target.lons)
    np.testing.assert_allclose(result.latitude.values, target.lats)


def test_reproject_dataset_uses_selected_two_dimensional_variables_and_keeps_time():
    source, target = matching_grids()
    data = xr.Dataset(
        {
            "selected": (("y", "x"), [[1, 2], [3, 4]]),
            "other": (("y", "x"), [[5, 6], [7, 8]]),
            "rank_one": (("z",), [9, 10]),
            "t": xr.DataArray(np.datetime64("2026-01-01")),
        }
    )

    result = source.reproject(  # type: ignore[call-overload]
        data, target, variables=["selected", "missing", "rank_one"]
    )

    assert set(result.data_vars) == {"selected", "t"}
    np.testing.assert_array_equal(result["selected"], data["selected"])
    assert result["t"].values == data["t"].values


def test_reproject_dataset_uses_default_variables():
    source = make_grid("source", default_variables=["second"])
    target = make_grid("target")
    source._row_indexes[target] = np.array([[0, 0], [1, 1]])
    source._col_indexes[target] = np.array([[0, 1], [0, 1]])
    data = xr.Dataset(
        {
            "first": (("y", "x"), [[1, 2], [3, 4]]),
            "second": (("y", "x"), [[5, 6], [7, 8]]),
        }
    )

    result = source.reproject(data, target)

    assert set(result.data_vars) == {"second"}


def test_reproject_band_array_preserves_band_names_and_sets_target_grid():
    source, target = matching_grids()
    bands = BandArray(source)
    bands["C13"] = np.array([[1, 2], [3, 4]])
    bands["C15"] = np.array([[5, 6], [7, 8]])

    result = source.reproject(bands, target)

    assert result.grid is target
    np.testing.assert_array_equal(result["C13"], bands["C13"])
    np.testing.assert_array_equal(result["C15"], bands["C15"])


def test_reproject_rejects_unsupported_array_ranks_and_types():
    source, target = matching_grids()

    with pytest.raises(ValueError, match="rank 2 or 3"):
        source.reproject(np.zeros(2), target)
    with pytest.raises(ValueError, match="invalid data"):
        source.reproject("not data", target)  # type: ignore[arg-type]


def test_mask_reprojects_an_all_true_source_mask(monkeypatch):
    source = make_grid("source")
    target = make_grid("target")
    reproject = Mock(return_value=np.array([[True, False], [False, True]]))
    monkeypatch.setattr(source, "reproject", reproject)

    result = source.mask(target)

    np.testing.assert_array_equal(result, [[True, False], [False, True]])
    supplied_mask, supplied_target = reproject.call_args.args
    assert supplied_target is target
    assert supplied_mask.dtype == bool
    assert supplied_mask.shape == source.shape
    assert supplied_mask.all()


def test_pixel_areas_returns_the_cached_array():
    grid = make_grid()
    expected = np.array([[1.5, 2.5], [3.5, 4.5]])
    grid._pixel_area_cache.get = Mock(return_value={"pixel_areas": expected})

    assert grid.pixel_areas is expected
    grid._pixel_area_cache.get.assert_called_once()


def test_pixel_areas_handles_grids_with_fewer_rows_than_cpus(monkeypatch):
    grid = make_grid()
    longitudes, latitudes = grid.cell_corners()
    monkeypatch.setattr(grid, "cell_corners", lambda: (longitudes, latitudes))
    monkeypatch.setattr(grid_module.os, "cpu_count", lambda: 64)

    class InlineExecutor:
        def __init__(self, max_workers):
            assert max_workers == 8

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def submit(self, function, *args):
            future = grid_module.concurrent.futures.Future()
            future.set_result(function(*args))
            return future

    monkeypatch.setattr(
        grid_module.concurrent.futures, "ProcessPoolExecutor", InlineExecutor
    )
    grid._pixel_area_cache.get = Mock(side_effect=lambda keys, calculate: calculate())

    result = grid.pixel_areas

    assert result.shape == grid.shape
    assert np.all(result > 0)


def test_pixel_area_block_calculates_square_kilometres():
    longitudes = np.array([[0.0, 1.0], [0.0, 1.0]])
    latitudes = np.array([[1.0, 1.0], [0.0, 0.0]])

    result = _do_pixel_area_block(longitudes, latitudes, CRS.from_epsg(4326))

    assert result.shape == (1, 1)
    assert result[0, 0] == pytest.approx(12_308.78, rel=1e-4)

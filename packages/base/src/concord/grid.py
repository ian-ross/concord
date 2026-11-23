from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
import os
from typing import overload, Self, Protocol
import warnings

import numpy as np
from pyresample.geometry import AreaDefinition
import xarray as xr
from pyproj import CRS, Transformer
import pyresample
import pyresample.area_config
from pyresample.image import ImageContainer
from pyresample.utils import generate_nearest_neighbour_linesample_arrays
import rasterio

from .cache import Cache
from .bands import BandArray


class HasCells(Protocol):
    shape: tuple[int, int]

    def cell_corners(self) -> tuple[np.ndarray, np.ndarray]:
        ...


@dataclass(frozen=True)
class Swath(HasCells):
    """Simple latitude/longitude container for when we don't have a
    simply-defined grid, just coordinates from a satellite swath."""
    lats: np.ndarray
    lons: np.ndarray
    shape: tuple[int, int] = None # type: ignore

    def __post_init__(self):
        if self.lats.shape != self.lons.shape:
            raise ValueError(
                'latitude and longitude arrays must have identical shapes'
            )
        if self.lats.ndim != 2:
            raise ValueError(
                'latitude and longitude arrays must be two-dimensional'
            )
        if not np.any(np.isfinite(self.lats) & np.isfinite(self.lons)):
            raise ValueError(
                'latitude and longitude arrays must contain at least one finite value'
            )
        object.__setattr__(self, 'shape', self.lats.shape)

    def cell_corners(self) -> tuple[np.ndarray, np.ndarray]:
        """Return longitude and latitude arrays for the cell corners.

        Interior corners are the mean of the four surrounding cell centers.
        At the swath boundary, the same midpoint calculation is linearly
        extrapolated by half a cell. Longitudes are unwrapped first so cells
        crossing the date line remain small adjoining quadrilaterals.
        """
        lons = _unwrap_longitudes(self.lons)
        return _coordinate_corners(lons), _coordinate_corners(self.lats)


def _coordinate_corners(centers: np.ndarray) -> np.ndarray:
    """Interpolate cell centers to corners, extrapolating at the boundary."""
    values = np.asarray(centers, dtype=float)
    return _coordinate_edges(_coordinate_edges(values, axis=1), axis=0)


def _coordinate_edges(values: np.ndarray, axis: int) -> np.ndarray:
    """Find the midpoints between values along one axis."""
    moved = np.moveaxis(values, axis, -1)
    if moved.shape[-1] == 1:
        # There is no spacing to infer in this direction.
        return np.moveaxis(np.repeat(moved, 2, axis=-1), -1, axis)

    edges = np.empty((*moved.shape[:-1], moved.shape[-1] + 1), dtype=float)
    edges[..., 1:-1] = (moved[..., :-1] + moved[..., 1:]) / 2
    edges[..., 0] = 1.5 * moved[..., 0] - 0.5 * moved[..., 1]
    edges[..., -1] = 1.5 * moved[..., -1] - 0.5 * moved[..., -2]
    return np.moveaxis(edges, -1, axis)


def _unwrap_longitudes(longitudes: np.ndarray) -> np.ndarray:
    """Unwrap finite longitude runs without spreading NaNs across the swath."""
    result = np.asarray(longitudes, dtype=float).copy()
    for axis in (1, 0):
        lines = np.moveaxis(result, axis, -1)
        for index in np.ndindex(lines.shape[:-1]):
            line = lines[index]
            finite = np.flatnonzero(np.isfinite(line))
            breaks = np.flatnonzero(np.diff(finite) > 1) + 1
            for run in np.split(finite, breaks):
                line[run] = np.unwrap(line[run], period=360.0)
    return result


# ------------------------------------------------------------------------------

# GOES ABI resolution LCC grid.

LCC_CRS = CRS.from_proj4(
    '+proj=lcc +ellps=GRS80 +lat_1=33 +lat_2=45 '
    '+lat_0=35 +lon_0=-95 +units=m +no_defs'
)

LCC_ROWS = 2000
LCC_COLS = 3000

LCC_EXTENT = (-3000000.0, -2000000.0, 3000000.0, 2000000.0)

LCC_AREA = pyresample.area_config.create_area_def(
    'lcc_area', LCC_CRS,
    width=LCC_COLS, height=LCC_ROWS,
    area_extent=LCC_EXTENT,
    nprocs=8
)
assert isinstance(LCC_AREA, AreaDefinition)

LCC_LONS, LCC_LATS = LCC_AREA.get_lonlats()

# ------------------------------------------------------------------------------

# Fine (VIIRS) resolution LCC grid.

# FINE_LCC_ROWS = LCC_ROWS * 4
# FINE_LCC_COLS = LCC_COLS * 4

# FINE_LCC_AREA = pyresample.area_config.create_area_def(
#     'lcc_area', LCC_CRS,
#     width=FINE_LCC_COLS, height=FINE_LCC_ROWS,
#     area_extent=LCC_EXTENT,
#     nprocs=8
# )
# assert isinstance(FINE_LCC_AREA, AreaDefinition)

# FINE_LCC_LONS, FINE_LCC_LATS = FINE_LCC_AREA.get_lonlats()

# ------------------------------------------------------------------------------

# A longitude/latitude CRS based on the same ellipsoid as the LCC grid.

LONLAT_CRS = CRS.from_proj4(
    '+proj=lonlat +ellps=GRS80 +sweep=x +nodefs'
)


# A coordinate transformer to go from the LCC grid to longitude/latitude
# values.

lcc_to_lonlat = Transformer.from_crs(LCC_CRS, LONLAT_CRS)


# LCC grid bounds in longitude/latitude.

LCC_LONLAT_LL = lcc_to_lonlat.transform(LCC_EXTENT[0], LCC_EXTENT[1])
LCC_LONLAT_UR = lcc_to_lonlat.transform(LCC_EXTENT[2], LCC_EXTENT[3])

# ------------------------------------------------------------------------------

# Grid class definition.

# Types that are supported by the Grid.reproject method.

Reprojectable = xr.Dataset | xr.DataArray | np.ndarray | BandArray


class Grid(HasCells):
    """Geospatial grid supporting reprojection.

    We need to be able to reproject data from GOES ABI and other grids to the
    LCC grid that we use for contrail detection. This class wraps the
    functionality that we need.
    """

    def __init__(
            self, name: str,
            area: pyresample.geometry.AreaDefinition | None = None,
            crs: CRS | None = None,
            nx: int | None = None,
            ny: int | None = None,
            extent: tuple[float, float, float, float] | None = None,
            default_variables: list[str] | None = None,
            radius_of_influence: int = 50000
    ):
        # Name just for debugging help.
        self.name = name

        # Make resampling area definition.
        if area is not None:
            self.area = area
        else:
            if not all(v is not None for v in [crs, nx, ny, extent]):
                raise ValueError(
                    'either area or all of crs, nx, ny, extent '
                    'required in concord.Grid'
                )
            self.area = pyresample.geometry.AreaDefinition(
                name, name, name, crs, nx, ny, extent
            )

        self.shape = self.area.shape
        self.crs = self.area.crs
        self.extent = self.area.area_extent

        # Get grid longitudes and latitudes from pyresample area definition.
        self.lons, self.lats = self.area.get_lonlats()

        # Parameter for nearest neighbor resampling.
        self.radius_of_influence = radius_of_influence

        # Default variables for reprojecting Datasets.
        self.default_variables = default_variables

        # Set up for on-demand creation of regridding lookup indexes.
        self._row_indexes = {}
        self._col_indexes = {}

        # Coordinate transformers between ABI grid and LCC and
        # longitude/latitude coordinates.
        self._to_lcc = Transformer.from_crs(self.area.crs, LCC_CRS)
        self._to_lonlat = Transformer.from_crs(self.area.crs, LONLAT_CRS)
        self._from_lcc = Transformer.from_crs(LCC_CRS, self.area.crs)
        self._from_lonlat = Transformer.from_crs(LONLAT_CRS, self.area.crs)

        # Create caches.
        self._index_cache = Cache('reprojection-indexes')
        self._pixel_area_cache = Cache('pixel-areas')
        self._land_sea_mask_cache = Cache('land-sea-masks')

    # Must be hashable for the _row_index and _col_indexes dictionaries. Since
    # pyresample's AreaDefinition is hashable, we just use that, since all the
    # things we want to be able to reference this way depend only on the area
    # definition.
    def __hash__(self):
        return hash(self.area)

    def mask(self, target: Self) -> np.ndarray:
        """Return area mask reprojected to another grid."""
        mask = np.ones(self.area.shape, dtype=bool)
        return self.reproject(mask, target)

    @property
    def transform(self) -> tuple[float, float, float, float, float, float]:
        """GDAL-style affine coefficients for pixel-corner coordinates."""
        xmin, ymin, xmax, ymax = self.extent
        dx = (xmax - xmin) / self.area.width
        dy = (ymax - ymin) / self.area.height
        return (dx, 0.0, xmin, 0.0, -dy, ymax)

    def rasterio_profile(self, dtype, channels: int) -> dict:
        transform = rasterio.transform.from_bounds(
            *self.area.area_extent,
            width=self.area.width,
            height=self.area.height
        )
        return dict(
            driver='GTiff', dtype=dtype, count=channels,
            width=self.area.width, height=self.area.height,
            crs=self.area.crs, transform=transform
        )

    def to_lcc(self, x: float, y: float) -> tuple[float, float]:
        """Convert coordinates from this grid to LCC grid."""
        return self._to_lcc.transform(x, y)

    def to_lonlat(self, x: float, y: float) -> tuple[float, float]:
        """Convert coordinates from this grid to longitude/latitude."""
        return self._to_lonlat.transform(x, y)

    def from_lcc(self, x: float, y: float) -> tuple[float, float]:
        """Convert coordinates from LCC grid to this grid."""
        return self._from_lcc.transform(x, y)

    def from_lonlat(self, x: float, y: float) -> tuple[float, float]:
        """Convert coordinates from longitude/latitude to this grid."""
        return self._from_lonlat.transform(x, y)

    def row_indexes(self, target: Self):
        if target not in self._row_indexes:
            self._calculate_lookup_indexes(target)
        return self._row_indexes[target]

    def col_indexes(self, target: Self):
        if target not in self._col_indexes:
            self._calculate_lookup_indexes(target)
        return self._col_indexes[target]

    def _calculate_lookup_indexes(self, target: Self):
        # Generate index lookup tables for regridding to LCC and Web
        # mercator grids. These are cached since they take a little while to
        # generate. The cache keys are based on the source (self) and target
        # grids via the pyresample AreaDefinition values that we use.

        def calculate():
            row_indexes, col_indexes = (
                generate_nearest_neighbour_linesample_arrays(
                    self.area, target.area,
                    radius_of_influence=self.radius_of_influence,
                    nprocs=os.cpu_count() or 1
                )
            )
            return dict(row_indexes=row_indexes, col_indexes=col_indexes)

        indexes = self._index_cache.get((self, target), calculate)
        self._row_indexes[target] = indexes['row_indexes']
        self._col_indexes[target] = indexes['col_indexes']

    def cell_corners(self):
        # Projection coordinates of edges of grid cells.
        xs, ys = np.zeros(self.area.width + 1), np.zeros(self.area.height + 1)
        xs[0], ys[0] = self.area.area_extent[0], self.area.area_extent[3]
        pxs, pys = self.area.projection_x_coords, self.area.projection_y_coords
        xs[1:-1] = (pxs[1:] + pxs[:-1]) / 2
        ys[1:-1] = (pys[1:] + pys[:-1]) / 2
        xs[-1], ys[-1] = self.area.area_extent[2], self.area.area_extent[1]

        # Grid cell corner longitudes and latitudes.
        cxs, cys = np.meshgrid(xs, ys)
        clons, clats = self.area.get_lonlat_from_projection_coordinates(cxs, cys)
        return clons, clats

    @property
    def pixel_areas(self):
        def calculate():
            clons, clats = self.cell_corners()

            # For each grid cell (pixel), use the pyproj Geod associated with
            # the input area's CRS to calculate the area defined by the grid
            # cell's corner coordinates. We parallelize this by making
            # row-wise blocks.
            pixels = np.zeros((self.area.height, self.area.width))

            with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
                total_rows = self.area.height
                rows_per_task = max(1, total_rows // (os.cpu_count() or 1))
                row_limits = (
                    list(range(0, total_rows, rows_per_task)) + [total_rows]
                )

                # Submit block processing tasks: note the "+1" on the "end"
                # indexes for each block - we need all the corners of the grid
                # cells in the block, so there is one more than the number of
                # cells in each direction!
                futures = {
                    executor.submit(
                        _do_pixel_area_block,
                        clons[irows:irowe+1], clats[irows:irowe+1], self.area.crs
                    ): (irows, irowe)
                    for irows, irowe in zip(row_limits[:-1], row_limits[1:])
                }
                for future in concurrent.futures.as_completed(futures):
                    # Usual destination indexing trick with futures
                    # dictionary.
                    irows, irowe = futures[future]
                    pixels[irows:irowe, :] = future.result()

            # Return in a dictionary for compatibility with caching system.
            return dict(pixel_areas=pixels)

        areas = self._pixel_area_cache.get((self,), calculate)
        return areas['pixel_areas']

    @overload
    def reproject(self, data: xr.Dataset, target: Self) -> xr.Dataset:
        ...

    @overload
    def reproject(self, data: xr.DataArray, target: Self) -> xr.DataArray:
        ...

    @overload
    def reproject(self, data: np.ndarray, target: Self) -> np.ndarray:
        ...

    @overload
    def reproject(self, data: BandArray, target: Self) -> BandArray:
        ...

    def reproject(
            self, data: Reprojectable, target: Self, variables=None
    ) -> Reprojectable:
        with warnings.catch_warnings():
            # The ImageContainerQuick class is deprecated, but it's much
            # easier to use than the alternatives.
            warnings.simplefilter('ignore', category=FutureWarning)

            # There are some warnings about projection setup that aren't
            # relevant.
            warnings.simplefilter('ignore', category=UserWarning)

            row_indexes = self.row_indexes(target)
            col_indexes = self.col_indexes(target)
            coords = dict(
                longitude=(['y', 'x'], target.lons),
                latitude=(['y', 'x'], target.lats)
            )

            match data:
                case xr.Dataset():
                    # Iterate either over a given set of variables, a set of
                    # default variables, or all 2-D variables in the dataset.
                    all_vars = variables
                    if all_vars is None:
                        all_vars = self.default_variables
                    if all_vars is None:
                        all_vars = data.data_vars

                    out_vars = {}
                    for v in all_vars:
                        # Skip missing and non-2D variables.
                        if v not in data.data_vars:
                            continue
                        if len(data.data_vars[v].shape) != 2:
                            continue

                        con = ImageContainer(data[v].values, self.area)
                        out_vars[v] = (
                            ['y', 'x'],
                            con.get_array_from_linesample(
                                row_indexes, col_indexes
                            )
                        )
                    result = xr.Dataset(out_vars, coords=coords)
                    if 't' in data:
                        result['t'] = data['t']
                    return result

                case xr.DataArray():
                    con = ImageContainer(data.values, self.area)
                    out_data = con.get_array_from_linesample(
                        row_indexes, col_indexes
                    )
                    return xr.DataArray(
                        data=out_data,
                        dims=['y', 'x'],
                        coords=coords
                    )

                case np.ndarray():
                    match len(data.shape):
                        case 3:
                            layers = []
                            for i in range(data.shape[2]):
                                con = ImageContainer(data[:, :, i], self.area)
                                layers.append(con.get_array_from_linesample(
                                    row_indexes, col_indexes
                                ))
                            return np.dstack(layers)
                        case 2:
                            con = ImageContainer(data, self.area)
                            return con.get_array_from_linesample(
                                row_indexes, col_indexes
                            )
                        case _:
                            raise ValueError(
                                'can only reproject rank 2 or 3 arrays'
                            )

                case BandArray():
                    out = BandArray(target)
                    for band in data._data:
                        con = ImageContainer(data[band], self.area)
                        out[band] = con.get_array_from_linesample(
                            row_indexes, col_indexes
                        )
                    return out

                case _:
                    raise ValueError('invalid data to Grid.reproject')


def _do_pixel_area_block(clons, clats, crs):
    area_fn = crs.get_geod().polygon_area_perimeter

    pixels = np.zeros((clons.shape[0] - 1, clons.shape[1] - 1))

    for irow in range(clons.shape[0] - 1):
        lon_top = clons[irow]
        lon_bot = clons[irow + 1]
        lat_top = clats[irow]
        lat_bot = clats[irow + 1]

        for icol in range(clons.shape[1] - 1):
            pixels[irow, icol], _ = area_fn(
                [lon_bot[icol], lon_bot[icol+1], lon_top[icol+1], lon_top[icol]],
                [lat_bot[icol], lat_bot[icol+1], lat_top[icol+1], lat_top[icol]]
            )

    return pixels / 1.0E6


# ------------------------------------------------------------------------------

# Grid definitions for fixed grids.

LCC_GRID = Grid(name='LCC', area=LCC_AREA)
# FINE_LCC_GRID = Grid(name='FINE_LCC', area=FINE_LCC_AREA)

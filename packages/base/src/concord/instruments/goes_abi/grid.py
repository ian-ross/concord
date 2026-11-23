import xarray as xr
from pyproj import CRS

import concord.grid as base_grid


class ABIGrid(base_grid.Grid):
    """Set up coordinate reference systems for an ABI data file."""

    def __init__(self, ds: xr.Dataset):
        sat_height = ds.goes_imager_projection.perspective_point_height

        super().__init__(
            name='ABI_CONUS',

            # ABI coordinate reference system taken direct from NetCDF file
            # attributes to ensure correctness of orbital parameters.
            crs=CRS.from_cf(ds.goes_imager_projection.attrs),

            # Grid size.
            nx=len(ds.x), ny=len(ds.y),

            # Data extent in ABI CRS coordinates. (The coordinate values in
            # the NetCDF file are in terms of mirror angles and need to be
            # converted to pyproj satellite projection coordinate values by
            # multiplying by the satellite height.)
            extent=(
                float(ds.x[0]) * sat_height, float(ds.y[-1]) * sat_height,
                float(ds.x[-1]) * sat_height, float(ds.y[0]) * sat_height
            ),

            # Radius of influence (m) for nearest-neighbor resampling.
            # (Conservative value to deal with pixel spread near bounds of ABI
            # images.)
            radius_of_influence=100000,
        )

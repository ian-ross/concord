from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from functools import cached_property
from pathlib import Path

import concord.services.laads as laads
import netCDF4 as nc4
import numpy as np
import xarray as xr

from concord.bands import BandArray
from concord.grid import Swath
from concord.instruments.transits import TransitFinder
from concord.services.space_track import TLEDatabase
from concord.utils import EXTENT

# TODO: Make this configurable.
MAX_WORKERS = 8

logger = logging.getLogger(__name__)


# List of LAADS granule filenames.
Granules = list[str]

@dataclass
class VIIRSProducts:
    """VIIRS product names for a given platform."""
    image_data: str
    image_navigation: str
    moderate_data: str
    moderate_navigation: str

PRODUCTS = {
    'NOAA-20': VIIRSProducts(
        image_data='VJ102IMG',
        image_navigation='VJ103IMG',
        moderate_data='VJ102MOD',
        moderate_navigation='VJ103MOD'
    ),
    'NOAA-21': VIIRSProducts(
        image_data='VJ202IMG',
        image_navigation='VJ203IMG',
        moderate_data='VJ202MOD',
        moderate_navigation='VJ203MOD'
    ),
    'Suomi-NPP': VIIRSProducts(
        image_data='VNP02IMG',
        image_navigation='VNP03IMG',
        moderate_data='VNP02MOD',
        moderate_navigation='VNP03MOD'
    )
}

# Platform IDs.
PLATFORMS = list(PRODUCTS.keys())


@dataclass
class VIIRSFiles:
    platform: str
    data: Path
    navigation: Path

    def to_dict(self) -> dict:
        return {
            'platform': self.platform,
            'data': self.data.name,
            'navigation': self.navigation.name,
        }

    @classmethod
    def from_dict(cls, d: dict, data_dir: Path) -> VIIRSFiles:
        return cls(
            platform=d['platform'],
            data=data_dir / d['data'],
            navigation=data_dir / d['navigation'],
        )

    def __post_init__(self):
        for attr in ['data', 'navigation']:
            if isinstance(getattr(self, attr), str):
                setattr(self, attr, Path(getattr(self, attr)))

    @property
    def time(self) -> datetime:
        # File names are like: VNP02MOD.A2024001.1718.002.2024006101151.nc
        time_parts = '.'.join(self.data.name.split('.')[1:3])
        return datetime.strptime(time_parts, 'A%Y%j.%H%M').replace(tzinfo=UTC)

    @cached_property
    def swath(self) -> Swath:
        """Load the navigation data and return a Swath object."""
        return self.load_navigation()

    @property
    def shape(self) -> tuple[int, int]:
        """Load the navigation data and return a Swath object."""
        return self.swath.shape

    @cached_property
    def valid_mask(self) -> np.ndarray:
        """Return a boolean mask of valid pixels in the swath."""
        with nc4.Dataset(self.data) as ds:
            ds.set_auto_maskandscale(False)
            obs = ds.groups['observation_data']
            raw = obs.variables['M14'][:]
            return raw <= obs.variables['M14'].valid_max

    def load_brightness_temperatures(self) -> BandArray:
        """Load radiance data from the VIIRS data file, using the navigation
        file to set up the grid."""
        with nc4.Dataset(self.data) as ds:
            ds.set_auto_maskandscale(False)
            obs = ds.groups['observation_data']
            data = BandArray(grid=None)
            for band in ['M14', 'M15', 'M16']:
                raw = obs.variables[band][:]
                lut = obs.variables[f'{band}_brightness_temperature_lut'][:]
                valid_max = obs.variables[band].valid_max
                bt = lut[raw].astype(np.float32)
                bt[raw > valid_max] = np.nan
                data[band] = _fill_along_track(bt)
        return data

    def load_navigation(self) -> Swath:
        """Load navigation data from the VIIRS navigation file."""
        ds = xr.open_datatree(self.navigation).geolocation_data
        return Swath(
            lons=ds.longitude.to_numpy(),
            lats=ds.latitude.to_numpy()
        )


def _fill_along_track(bt: np.ndarray) -> np.ndarray:
    """Fill NaN values by column-wise linear interpolation along the along-track direction.

    Each column is interpolated independently using its own valid pixels as
    control points. NaN at the top or bottom of a column (no valid pixels to
    interpolate from on one side) are filled with the nearest valid value in
    that column. Columns that are entirely NaN fall back to the band mean.
    """
    result = bt.copy()
    rows = np.arange(bt.shape[0])
    band_mean = float(np.nanmean(bt))

    for col in range(bt.shape[1]):
        column = result[:, col]
        valid = ~np.isnan(column)
        if valid.all():
            continue
        if not valid.any():
            result[:, col] = band_mean
        else:
            result[:, col] = np.interp(rows, rows[valid], column[valid])

    return result


def list_granules(
        platform: str,
        start_time: datetime,
        end_time: datetime,
        data_dir: Path,
        transit_finder: TransitFinder,
        laads_client: laads.Client,
        bands='MI'
) -> list[VIIRSFiles]:
    """Identify VIIRS granules around a given time.

    VIIRS granules for a given platform are found between a start time (time -
    dt_before) and an end time (time + dt_after). Integer numbers of minutes
    are used for the deltas to fit with the data storage conventions of the
    LAADS DAAC archive, which creates data files with minute-level labeling.

    Parameters
    ----------
    platform: str
        VIIRS platform (OrbNav names: 'NOAA-20' or 'Suomi-NPP').
    start_time: datetime
        Beginning of time interval to look for granules.
    end_time: datetime
        End of time interval to look for granules.
    transit_finder: TransitFinder
        TransitFinder instance to find satellite transits through the CONUS
        extent.
    laads_client: laads.Client
        LAADS client to use for listing granules.
    bands: str
        VIIRS band types to list: string containing one or more of 'M'
        (Moderate) or 'I' (Imaging).

    Returns
    -------
    VIIRSFiles
        List of VIIRSFiles objects, each containing the platform name, data
        file path, and navigation file path for a granule.
    """
    data_granules = []
    nav_granules = []

    def get(product, transit):
        return laads_client.list_granules(
            product,
            start=transit.enter.time, end=transit.leave.time
        )

    futs = []
    prods = PRODUCTS[platform]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for t in transit_finder(platform, start_time, end_time, EXTENT):
            if 'M' in bands:
                futs.append(ex.submit(get, prods.moderate_data, t))
                futs.append(ex.submit(get, prods.moderate_navigation, t))
            if 'I' in bands:
                futs.append(ex.submit(get, prods.image_data, t))
                futs.append(ex.submit(get, prods.image_navigation, t))
        for f in as_completed(futs):
            granules = f.result()
            if not granules:
                continue
            if (
                    granules[0].startswith(prods.image_data) or
                    granules[0].startswith(prods.moderate_data)
            ):
                data_granules += granules
            else:
                nav_granules += granules

        data_granules = sorted(set(data_granules))
        nav_granules = sorted(set(nav_granules))
        if len(data_granules) != len(nav_granules):
            logger.warning(
                'Mismatched data and navigation granules for %s: %s vs %s',
                platform, data_granules, nav_granules
            )

    return [
        VIIRSFiles(platform=platform, data=data_dir / d, navigation=data_dir / n)
        for d, n in zip(data_granules, nav_granules)
    ]


def _list_files(
        dt: date,
        data_dir: Path,
        tles: TLEDatabase,
        laads_client: laads.Client
) -> dict[datetime, VIIRSFiles]:
    """List VIIRS data files for the given date.

    """
    transit_finder = TransitFinder(tles)
    start_time = datetime(dt.year, dt.month, dt.day, tzinfo=UTC)
    end_time = start_time + timedelta(days=1)

    files = []
    for sat in PLATFORMS:
        files += list_granules(
            sat, start_time, end_time, data_dir,
            transit_finder, laads_client, bands='MI'
        )

    return {f.time: f for f in files}


def list_files(
        dt: date, data_dir: Path, tles: TLEDatabase, laads_client: laads.Client
) -> dict[datetime, VIIRSFiles]:
    all_files_cache = Path('all_files_cache.json')
    if all_files_cache.exists():
        with open(all_files_cache) as fp:
            all_files = {
                datetime.fromisoformat(k): VIIRSFiles.from_dict(v, data_dir)
                for k, v in json.load(fp).items()
            }
    else:
        all_files = _list_files(dt, data_dir, tles, laads_client)
        with open(all_files_cache, 'w') as fp:
            json.dump(
                {k.isoformat(): v.to_dict() for k, v in all_files.items()},
                fp, indent=2
            )
    return all_files

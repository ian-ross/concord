from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from itertools import chain
import json
from pathlib import Path

import s3fs
import xarray as xr

from timelines.bands import BandArray

from .grid import ABIGrid


@dataclass
class GOESFiles:
    mcmipc: Path
    c11: Path
    c13: Path
    c14: Path
    c15: Path

    def to_dict(self) -> dict:
        return {
            'mcmipc': str(self.mcmipc),
            'c11': str(self.c11),
            'c13': str(self.c13),
            'c14': str(self.c14),
            'c15': str(self.c15)
        }

    @classmethod
    def from_dict(cls, d: dict) -> GOESFiles:
        return cls(
            mcmipc=Path(d['mcmipc']),
            c11=Path(d['c11']),
            c13=Path(d['c13']),
            c14=Path(d['c14']),
            c15=Path(d['c15'])
        )

    def __post_init__(self):
        for attr in ['mcmipc', 'c11', 'c13', 'c14', 'c15']:
            if isinstance(getattr(self, attr), str):
                setattr(self, attr, Path(getattr(self, attr)))

    def load_brightness_temperatures(self, data_dir: Path) -> BandArray:
        """Load radiance data from the GOES ABI MCMIPC file."""
        ds = xr.open_dataset(data_dir / self.mcmipc.name)
        data = BandArray(grid=ABIGrid(ds))
        data['C11'] = ds.CMI_C11.to_numpy()
        data['C13'] = ds.CMI_C13.to_numpy()
        data['C14'] = ds.CMI_C14.to_numpy()
        data['C15'] = ds.CMI_C15.to_numpy()
        return data


s3 = s3fs.S3FileSystem(anon=True)


def _list_files(dt: date) -> dict[datetime, GOESFiles]:
    """List GOES data files for the given date.

    List files in s3://noaa-goes19/ABI-L1b-RadC/YYYY/DDD/HH for HH in 00, 01,
    ..., 23 and YYYY/DDD for the requested date.
    """
    def start_time(fname: str) -> datetime:
        """Extract the start time from a GOES filename."""
        parts = fname.split('/')[-1].split('_')
        return datetime.strptime(parts[3][1:14], '%Y%j%H%M%S')

    yday = dt.toordinal() - date(dt.year, 1, 1).toordinal() + 1

    radc = {}
    radc_prefix = f'noaa-goes19/ABI-L1b-RadC/{dt.year}/{yday:03d}'
    for ch in ['C11', 'C13', 'C14', 'C15']:
        file_prefix = f'OR_ABI-L1b-RadC-M6{ch}'
        files = chain.from_iterable(
            s3.glob(f'{radc_prefix}/{hr:02d}/{file_prefix}*') for hr in range(24)
        )
        radc[ch] = {start_time(f): f for f in files if f.endswith('.nc')}

    mcmipc_prefix = f'noaa-goes19/ABI-L2-MCMIPC/{dt.year}/{yday:03d}'
    mcmipc_files = chain.from_iterable(
        s3.ls(f'{mcmipc_prefix}/{hr:02d}') for hr in range(24)
    )
    mcmipc = {start_time(f): f for f in mcmipc_files if f.endswith('.nc')}

    result = {}
    for dt in mcmipc.keys():
        if all(dt in radc[ch] for ch in ['C11', 'C13', 'C14', 'C15']):
            result[dt] = GOESFiles(
                mcmipc=Path(mcmipc[dt]),
                c11=Path(radc['C11'][dt]),
                c13=Path(radc['C13'][dt]),
                c14=Path(radc['C14'][dt]),
                c15=Path(radc['C15'][dt])
            )
    return result


def list_files(dt: date) -> dict[datetime, GOESFiles]:
    all_files_cache = Path('all_files_cache.json')
    if all_files_cache.exists():
        with open(all_files_cache) as fp:
            all_files = {
                datetime.fromisoformat(k): GOESFiles.from_dict(v)
                for k, v in json.load(fp).items()
            }
    else:
        all_files = _list_files(dt)
        with open(all_files_cache, 'w') as fp:
            json.dump(
                {k.isoformat(): v.to_dict() for k, v in all_files.items()},
                fp, indent=2
            )
    return all_files

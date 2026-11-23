from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import s3fs


def tstr(dt: datetime) -> str:
    return dt.strftime('%Y%m%dT%H%M%SZ')


def round_time(dt: datetime, rounding: str) -> datetime:
    return pd.Timestamp(dt).round(rounding).to_pydatetime() # type: ignore


s3 = s3fs.S3FileSystem(anon=True)


@dataclass
class Box:
    """A latitude/longitude box.

    The box is defined by a lower-left corner (southwest) and an upper-right
    corner (northeast). The box is inclusive of its edges.

    A box may cross the date line by giving a lower-left longitude greater than
    its upper-right longitude.

    A box may also cover the entire globe by giving a lower-left longitude equal
    to its upper-right longitude. In this case, the box is considered to contain
    all longitudes, and the east/west edges are ignored.

    """
    north: float
    south: float
    east: float
    west: float
    all_longitudes: bool = False
    longitude_width: float = 0.0

    def __post_init__(self) -> None:
        if not -90.0 <= self.south <= 90.0:
            raise ValueError("south latitude must be between -90 and 90")
        if not -90.0 <= self.north <= 90.0:
            raise ValueError("north latitude must be between -90 and 90")
        if self.south >= self.north:
            raise ValueError("south latitude must be below north latitude")
        if not -180.0 <= self.west <= 180.0:
            raise ValueError("west longitude must be between -180 and 180")
        if not -180.0 <= self.east <= 180.0:
            raise ValueError("east longitude must be between -180 and 180")

        raw_width = self.east - self.west
        self.all_longitudes = abs(raw_width) == 360.0
        if raw_width == 0.0:
            raise ValueError("the box must have nonzero longitude width")
        self.longitude_width = 360.0 if self.all_longitudes else raw_width % 360.0

    def contains(
            self, lat: float | np.ndarray, lon: float | np.ndarray
    ) -> bool | np.ndarray:
        if isinstance(lat, np.ndarray) != isinstance(lon, np.ndarray):
            raise ValueError('lat and lon must both be scalars or both be numpy arrays')

        if isinstance(lat, np.ndarray) or isinstance(lon, np.ndarray):
            lat_mask = (lat >= self.south) & (lat <= self.north)
            if self.all_longitudes:
                return lat_mask
            offset = (lon - self.west) % 360.0
            lon_mask = offset <= self.longitude_width
            return lat_mask & lon_mask

        if not self.south <= lat <= self.north:
            return False
        if self.all_longitudes:
            return True
        offset = (lon - self.west) % 360.0
        return offset <= self.longitude_width


EXTENT = Box(north=60.0, south=0.0, east=-50.0, west=-140.0)

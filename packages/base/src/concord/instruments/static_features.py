from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from importlib.resources import files, Anchor
from importlib.resources.abc import Traversable
import json
from pathlib import Path
import struct
from typing import Any
import zipfile

import numpy as np
from rasterio.features import rasterize
from rasterio.transform import Affine, from_bounds
from skimage.morphology import dilation
from skimage.measure import label
from pyproj import Transformer
from pyresample.geometry import SwathDefinition
from pyresample.kd_tree import resample_nearest

from ..grid import Grid, Swath, LCC_GRID, LONLAT_CRS


NATURAL_EARTH = files('timelines.assets') / 'static-features' / 'natural-earth-5.0.0'


class StaticFeatures:
    def __init__(
            self,
            grid: Grid | Swath = LCC_GRID,
            feature_directory: Traversable | Path = NATURAL_EARTH
    ):
        sources = json.loads(
            (feature_directory / 'SOURCES.json').read_text(encoding='utf-8')
        )
        vectors = sources['vectors']
        features = [
            geometry
            for vector in vectors
            for geometry in _read_shapefile_lines(feature_directory / vector['archive'])
        ]

        self.features = _rasterize_static_features(features, grid)

    def __call__(self, detections: np.ndarray, *, threshold: float = 0.9) -> np.ndarray:
        """Remove only 8-connected detections whose feature overlap meets ``threshold``."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError('threshold must be between zero and one')
        if detections.ndim != 2:
            raise ValueError('detections must be two-dimensional')
        if detections.shape != self.features.shape:
            raise ValueError('detections and features must have identical shapes')

        components, component_count = label(detections, return_num=True) # type: ignore
        filtered = detections.copy()
        for component_id in range(1, component_count + 1): # type: ignore
            component = components == component_id
            overlap = (
                int(np.count_nonzero(component & self.features)) /
                int(np.count_nonzero(component))
            )
            if overlap >= threshold:
                filtered[component] = False

        return filtered


def _rasterize_static_features(
    features: Sequence[Mapping[str, Any]], grid: Grid | Swath
) -> np.ndarray:
    """Rasterize longitude/latitude line features to a native grid and dilate them.

    Features use GeoJSON geometry mappings. For a ``Grid``, rasterization uses
    its affine transform. For a swath, features are rasterized on an
    intermediate regular area and resampled to the swath. The result receives a
    one-pixel binary dilation.
    """
    match grid:
        case Grid():
            mask = _rasterize_grid_features(features, grid)
        case Swath():
            mask = _rasterize_swath_features(features, grid)
    return dilation(mask)


def _rasterize_grid_features(
    features: Sequence[Mapping[str, Any]], grid: Grid
) -> np.ndarray:
    transformer = Transformer.from_crs(LONLAT_CRS, grid.crs, always_xy=True)
    geometries = tuple(
        _project_line(feature, transformer)
        for feature in features if feature.get('coordinates')
    )
    if not geometries:
        return np.zeros(grid.shape, dtype=bool)
    mask = rasterize(
        ((geometry, 1) for geometry in geometries),
        out_shape=grid.shape,
        transform=Affine(*grid.transform),
        default_value=1,
        all_touched=True,
        dtype=np.uint8,
    )
    assert mask is not None, 'rasterization failed to produce a mask'
    return mask.astype(bool)


def _rasterize_swath_features(
    features: Sequence[Mapping[str, Any]], swath: Swath
) -> np.ndarray:
    lats = np.asarray(swath.lats)
    lons = np.asarray(swath.lons)

    sw = SwathDefinition(lons=lons, lats=lats)
    area = sw.compute_optimal_bb_area()
    transformer = Transformer.from_crs(LONLAT_CRS, area.crs, always_xy=True)
    geometries = tuple(
        _project_line(feature, transformer)
        for feature in features if feature.get('coordinates')
    )
    if not geometries:
        return np.zeros(lats.shape, dtype=bool)

    regular_mask = rasterize(
        ((geometry, 1) for geometry in geometries),
        out_shape=area.shape,
        transform=from_bounds(
            *area.area_extent,
            width=area.width,
            height=area.height,
        ),
        default_value=1,
        all_touched=True,
        dtype=np.uint8,
    )
    assert regular_mask is not None, 'rasterization failed to produce a mask'

    radius = np.sqrt(2.0) * max(
        area.geocentric_resolution(),
        sw.geocentric_resolution(),
    )
    return resample_nearest(
        area,
        regular_mask,
        sw,
        radius_of_influence=radius,
        fill_value=0,
    ).astype(bool)


def _project_line(geometry: Mapping[str, Any], transformer: Transformer) -> dict[str, Any]:
    if geometry.get('type') != 'LineString':
        raise ValueError(f'unsupported static feature geometry: {geometry.get('type')}')
    return {
        'type': 'LineString',
        'coordinates': [
            transformer.transform(lon, lat)
            for lon, lat in geometry['coordinates']
        ]
    }


def _read_shapefile_lines(archive: Path | Anchor) -> Iterable[dict[str, Any]]:
    """Read PolyLine records from a vendored shapefile ZIP."""
    with zipfile.ZipFile(archive) as bundle:
        name = next(name for name in bundle.namelist() if name.endswith(".shp"))
        data = memoryview(bundle.read(name))
    offset = 100
    while offset < len(data):
        _, content_words = struct.unpack_from(">2i", data, offset)
        content_start = offset + 8
        content_end = content_start + content_words * 2
        shape_type = struct.unpack_from("<i", data, content_start)[0]
        if shape_type in (3, 13, 23):  # PolyLine, PolyLineZ, PolyLineM
            part_count, point_count = struct.unpack_from("<2i", data, content_start + 36)
            parts_start = content_start + 44
            parts = struct.unpack_from(f"<{part_count}i", data, parts_start)
            points_start = parts_start + 4 * part_count
            points = [
                struct.unpack_from("<2d", data, points_start + 16 * index)
                for index in range(point_count)
            ]
            for index, start in enumerate(parts):
                end = parts[index + 1] if index + 1 < part_count else point_count
                if end - start >= 2:
                    yield {"type": "LineString", "coordinates": points[start:end]}
        offset = content_end

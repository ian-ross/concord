import numpy as np
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from skimage.measure import label

from .grid import HasCells


def mask_to_outlines(mask: np.ndarray, geom: HasCells) -> dict:
    labels, count = label(mask, return_num=True) # type: ignore
    if count == 0:
        return {"type": "FeatureCollection", "features": []}
    longitudes, latitudes = geom.cell_corners()
    features = []
    for index in range(1, count + 1): # type: ignore
        rows, columns = np.nonzero(labels == index)
        cells = [
            Polygon(
                [
                    (float(longitudes[row, column]), float(latitudes[row, column])),
                    (float(longitudes[row, column + 1]), float(latitudes[row, column + 1])),
                    (float(longitudes[row + 1, column + 1]), float(latitudes[row + 1, column + 1])),
                    (float(longitudes[row + 1, column]), float(latitudes[row + 1, column])),
                ]
            )
            for row, column in zip(rows, columns, strict=True)
        ]
        geometry = unary_union(cells)
        feature_properties = {"pixel_count": int(len(rows))}
        features.append({
            "type": "Feature",
            "properties": feature_properties,
            "geometry": mapping(geometry)
        })
    return {"type": "FeatureCollection", "features": features}

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from skimage.morphology import erosion


if TYPE_CHECKING:
    from .grid import Grid


class BandArray:
    def __init__(self, grid: Grid | None = None):
        self.grid = grid
        self._data: dict[str, np.ndarray] = {}

    def __getitem__(self, band: str) -> np.ndarray:
        return self._data[band]

    def __setitem__(self, band: str, data: np.ndarray) -> None:
        if self.grid is not None and data.shape != self.grid.shape:
            raise ValueError(
                f'Data shape {data.shape} does not match grid shape {self.grid.shape}'
            )
        self._data[band] = data

    @cached_property
    def valid_mask(self) -> np.ndarray:
        """Return a boolean mask of valid data pixels across all bands."""
        if not self._data:
            raise ValueError('No bands available to compute valid mask')
        masks = [~np.isnan(data) for data in self._data.values()]

        # Use single pixel erosion to remove isolated domain edge pixels.
        return erosion(np.logical_and.reduce(masks))

    def reproject(self, target_grid: Grid) -> BandArray:
        """Reproject all bands to a new grid."""
        if self.grid is None:
            raise ValueError('Cannot reproject BandArray without a grid')
        return self.grid.reproject(self, target_grid)

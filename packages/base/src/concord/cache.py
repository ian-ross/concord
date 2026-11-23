from datetime import datetime
import os
from pathlib import Path
from shutil import rmtree
from typing import Hashable, Callable

from appdirs import user_cache_dir
import numpy as np


PROJECT_NAME = 'concord'


class Cache:
    def __init__(self, name: str, size_limit: int = 128):
        # Save parameters for this cache: sub-directory name and maximum local
        # storage (in Mb).
        self.name = name
        self.size_limit = size_limit

        # Local (writeable) cache directory.
        self.local_cache_dir = Path(user_cache_dir(PROJECT_NAME)) / name

    def get(self, hashables: list[Hashable], fn: Callable):
        # Build filename from hashable objects.
        f = '-'.join([format(abs(hash(h)), 'x') for h in hashables]) + '.npz'
        local_fname = self.local_cache_dir / f

        # If we have a version cached in the local cache, return it,
        # updating modification time for LRU caching.
        if local_fname.exists():
            epoch = datetime.now().timestamp()
            os.utime(local_fname, (epoch, epoch))
            # print('Using cached %s: %s', self.name, local_fname)
            return np.load(local_fname)

        # Calculate value to cache, which should be a dictionary of Numpy
        # arrays suitable for saving with np.savez.
        vals = fn()

        # Save the new values and apply size limit.
        local_fname.parent.mkdir(exist_ok=True, parents=True)
        np.savez(local_fname, **vals)
        self._limit_size()

        # Return computed value.
        return vals

    def purge(self):
        rmtree(self.local_cache_dir, ignore_errors=True)

    @staticmethod
    def purge_all():
        rmtree(user_cache_dir(PROJECT_NAME), ignore_errors=True)

    def _limit_size(self):
        # Manage total cached file size by removing old files until we're
        # below the limit.
        files = []
        total_size = 0
        for f in self.local_cache_dir.iterdir():
            s = f.stat()
            total_size += s.st_size
            files.append((f, s.st_mtime, s.st_size))
        to_remove = total_size - self.size_limit * 1024 * 1024
        if to_remove > 0:
            # Remove files in LRU order.
            files.sort(key=lambda t: t[1], reverse=True)
            while to_remove > 0:
                entry = files.pop()
                entry[0].unlink()
                to_remove -= entry[2]

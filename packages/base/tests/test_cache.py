import os
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from concord import cache as cache_module
from concord.cache import Cache


def make_cache(tmp_path: Path, monkeypatch, name: str = "test", size_limit: int = 128):
    monkeypatch.setattr(cache_module, "user_cache_dir", lambda app: str(tmp_path / app))
    return Cache(name, size_limit=size_limit)


def test_get_computes_saves_and_reuses_values(tmp_path: Path, monkeypatch):
    cache = make_cache(tmp_path, monkeypatch)
    calculate = Mock(return_value={"values": np.array([1, 2, 3])})

    first = cache.get(["key", 4], calculate)
    cache_file = next(cache.local_cache_dir.iterdir())
    old_mtime = 1.0
    os.utime(cache_file, (old_mtime, old_mtime))
    second = cache.get(["key", 4], calculate)

    np.testing.assert_array_equal(first["values"], [1, 2, 3])
    np.testing.assert_array_equal(second["values"], [1, 2, 3])
    assert cache_file.suffix == ".npz"
    assert cache_file.stat().st_mtime > old_mtime
    calculate.assert_called_once_with()
    second.close()


def test_limit_size_removes_oldest_files_first(tmp_path: Path, monkeypatch):
    cache = make_cache(tmp_path, monkeypatch, size_limit=1)
    cache.local_cache_dir.mkdir(parents=True)
    oldest = cache.local_cache_dir / "old.npz"
    newest = cache.local_cache_dir / "new.npz"
    oldest.write_bytes(b"x" * 600_000)
    newest.write_bytes(b"y" * 600_000)
    os.utime(oldest, (10, 10))
    os.utime(newest, (20, 20))

    cache._limit_size()

    assert not oldest.exists()
    assert newest.exists()


def test_purge_removes_only_this_cache(tmp_path: Path, monkeypatch):
    cache = make_cache(tmp_path, monkeypatch, name="one")
    other = make_cache(tmp_path, monkeypatch, name="two")
    cache.local_cache_dir.mkdir(parents=True)
    other.local_cache_dir.mkdir(parents=True)
    (cache.local_cache_dir / "value").touch()
    (other.local_cache_dir / "value").touch()

    cache.purge()

    assert not cache.local_cache_dir.exists()
    assert other.local_cache_dir.exists()


def test_purge_all_removes_the_application_cache(tmp_path: Path, monkeypatch):
    cache = make_cache(tmp_path, monkeypatch)
    cache.local_cache_dir.mkdir(parents=True)
    (cache.local_cache_dir / "value").touch()

    Cache.purge_all()
    Cache.purge_all()

    assert not cache.local_cache_dir.parent.exists()

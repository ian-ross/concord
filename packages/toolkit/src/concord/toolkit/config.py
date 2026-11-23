from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel

from concord.services import gmn, laads, space_track


class ServicesConfig(BaseModel):
    gmn: gmn.Config
    laads: laads.Config
    space_track: space_track.Config


class Config(BaseModel):
    services: ServicesConfig

    @classmethod
    def from_file(cls, path: Path) -> Config:
        """Load configuration from a file."""
        with open(path, 'rb') as f:
            return cls(**tomllib.load(f))

# Concord external services library

This package provides clients for three external data services.

## GMN camera archive

`concord.services.gmn.CameraArchive` lists, searches, and downloads contrail
camera data from the Global Meteor Network archive. It requires an archive
bearer token (which can be generated from the Nimbus web application once you
have login credentials there). The `RunDirectory` class helps to access
downloaded run products, including frame times, flight tracks, and contrail
detections.

## NASA LAADS

`concord.services.laads.Client` lists and downloads satellite data views
(granules) from the NASA LAADS archive. (This is primarily used for data from
the VIIRS instrument.) A LAADS bearer token is required when constructing the
client (you can get a token by registering for an account at
https://urs.earthdata.nasa.gov/). Granule searches accept a product name and
UTC time range.

## Space-Track

`concord.services.space_track.TLEDatabase` stores time-dependent two-line
elements (TLEs) expressing satellite orbital parameters in SQLite. It requires
login credentials to retrieve TLEs from the Space-Track servce, and `lookup()`
returns a `pyorbital.orbital.Orbital` for the nearest available TLE.

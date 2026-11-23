from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest

from concord.services import space_track


PLATFORM = "TESTSAT"
PLATFORM_ID = 99999
LINE1 = "1 99999U 24001A   24001.50000000  .00000000  00000-0  00000-0 0  9991"
LINE2 = "2 99999  98.7000 100.0000 0001000 200.0000 160.0000 14.20000000    01"


@pytest.fixture(autouse=True)
def block_live_http(monkeypatch: pytest.MonkeyPatch):
    def fail(*args, **kwargs):
        raise AssertionError("test attempted a live HTTP request")

    monkeypatch.setattr(space_track.httpx, "Client", fail)
    monkeypatch.setattr(space_track.httpx, "get", fail)


@pytest.fixture
def database(tmp_path: Path):
    db = space_track.TLEDatabase(tmp_path / "tles.sqlite", "user", "password")
    with db.conn:
        db.conn.execute(
            "INSERT INTO platforms (id, name) VALUES (?, ?)",
            (PLATFORM_ID, PLATFORM),
        )
    yield db
    db.conn.close()


def insert_tle(
    database: space_track.TLEDatabase,
    epoch: datetime,
    line1: str = LINE1,
    line2: str = LINE2,
):
    with database.conn:
        database.conn.execute(
            """
            INSERT INTO tles (platform, epoch, line1, line2)
            VALUES (?, ?, ?, ?)
            """,
            (PLATFORM_ID, epoch.timestamp(), line1, line2),
        )


def test_database_initialization_creates_tables_and_enables_foreign_keys(
    database: space_track.TLEDatabase,
):
    tables = {
        row[0]
        for row in database.conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }

    assert {"platforms", "tles"} <= tables
    assert database.conn.execute("PRAGMA foreign_keys").fetchone() == (1,)


def test_space_track_url_accepts_platform_name_or_catalog_id(
    database: space_track.TLEDatabase,
):
    expected = (
        space_track.SPACE_TRACK_BASE_URL
        + "/basicspacedata/query"
        + f"/class/gp_history/NORAD_CAT_ID/{PLATFORM_ID}"
        + "/orderby/TLE_LINE1 ASC"
        + "/EPOCH/2024-01-01--2024-01-03"
        + "/format/tle"
    )

    assert database._space_track_url(
        PLATFORM, date(2024, 1, 1), date(2024, 1, 3)
    ) == expected
    assert database._space_track_url(
        PLATFORM_ID, date(2024, 1, 1), date(2024, 1, 3)
    ) == expected


def test_space_track_url_rejects_unknown_platform(
    database: space_track.TLEDatabase,
):
    with pytest.raises(ValueError, match="Unknown platform: MISSING"):
        database._space_track_url(
            "MISSING", date(2024, 1, 1), date(2024, 1, 2)
        )


def test_login_posts_credentials_and_returns_cookies(
    database: space_track.TLEDatabase,
):
    cookies = httpx.Cookies({"session": "abc"})
    response = Mock(status_code=200, cookies=cookies)
    client = Mock()
    client.post.return_value = response

    assert database._login(client) is cookies
    client.post.assert_called_once_with(
        space_track.SPACE_TRACK_BASE_URL + "/ajaxauth/login",
        data={"identity": "user", "password": "password"},
    )


def test_login_reports_failed_response(database: space_track.TLEDatabase):
    response = Mock(status_code=403, text="forbidden")
    client = Mock()
    client.post.return_value = response

    with pytest.raises(ValueError, match="Login failed: 403 forbidden"):
        database._login(client)


class LoginClient:
    def __init__(self, response):
        self.response = response
        self.post_call = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def post(self, url, data):
        self.post_call = (url, data)
        return self.response


def test_download_logs_in_fetches_tles_and_ignores_duplicates(
    database: space_track.TLEDatabase,
    monkeypatch: pytest.MonkeyPatch,
):
    cookies = httpx.Cookies({"session": "abc"})
    login_response = Mock(status_code=200, cookies=cookies)
    client = LoginClient(login_response)
    client_factory = Mock(return_value=client)
    download_response = Mock(text=f"{LINE1}\r\n{LINE2}\r\n")
    get = Mock(return_value=download_response)
    monkeypatch.setattr(space_track.httpx, "Client", client_factory)
    monkeypatch.setattr(space_track.httpx, "get", get)

    database.download(PLATFORM, date(2024, 1, 1), date(2024, 1, 2))
    database.download(PLATFORM, date(2024, 1, 1), date(2024, 1, 2))

    expected_url = database._space_track_url(
        PLATFORM, date(2024, 1, 1), date(2024, 1, 2)
    )
    assert client.post_call == (
        space_track.SPACE_TRACK_BASE_URL + "/ajaxauth/login",
        {"identity": "user", "password": "password"},
    )
    assert get.call_args_list == [
        ((expected_url,), {"cookies": cookies}),
        ((expected_url,), {"cookies": cookies}),
    ]
    rows = database.conn.execute(
        """
        SELECT platforms.name, tles.epoch, tles.line1, tles.line2
        FROM tles JOIN platforms ON tles.platform = platforms.id
        """
    ).fetchall()
    assert rows == [
        (
            PLATFORM,
            datetime(2024, 1, 1, 12, tzinfo=UTC).timestamp(),
            LINE1,
            LINE2,
        )
    ]


def test_lookup_selects_nearest_tle_within_tolerance(
    database: space_track.TLEDatabase,
):
    requested = datetime(2024, 1, 2, 12, tzinfo=UTC)
    insert_tle(database, requested - timedelta(hours=2), "older", "older-2")
    insert_tle(database, requested + timedelta(minutes=30), "near", "near-2")

    assert database._lookup(
        PLATFORM, requested, timedelta(hours=3)
    ) == ("near", "near-2")
    assert database._lookup(
        PLATFORM, requested, timedelta(minutes=20)
    ) is None


def test_lookup_uses_cached_tle_without_downloading(
    database: space_track.TLEDatabase,
    monkeypatch: pytest.MonkeyPatch,
):
    epoch = datetime(2024, 1, 2, 12, tzinfo=UTC)
    insert_tle(database, epoch - timedelta(hours=1))
    download = Mock()
    database.download = download
    orbital = object()
    orbital_factory = Mock(return_value=orbital)
    monkeypatch.setattr(space_track, "Orbital", orbital_factory)

    assert database.lookup(PLATFORM, epoch) is orbital
    download.assert_not_called()
    orbital_factory.assert_called_once_with(
        PLATFORM, line1=LINE1, line2=LINE2
    )


def test_lookup_downloads_wider_date_range_on_cache_miss(
    database: space_track.TLEDatabase,
    monkeypatch: pytest.MonkeyPatch,
):
    epoch = datetime(2024, 1, 2, 12, tzinfo=UTC)

    def download(platform, start_epoch, end_epoch):
        assert platform == PLATFORM
        insert_tle(database, epoch + timedelta(hours=12))

    database.download = Mock(side_effect=download)
    orbital = object()
    orbital_factory = Mock(return_value=orbital)
    monkeypatch.setattr(space_track, "Orbital", orbital_factory)

    assert database.lookup(PLATFORM, epoch) is orbital
    database.download.assert_called_once_with(
        PLATFORM, date(2024, 1, 1), date(2024, 1, 3)
    )
    orbital_factory.assert_called_once_with(
        PLATFORM, line1=LINE1, line2=LINE2
    )


def test_lookup_raises_when_download_finds_no_tle(
    database: space_track.TLEDatabase,
):
    epoch = datetime(2024, 1, 2, 12, tzinfo=UTC)
    database.download = Mock()

    with pytest.raises(
        ValueError, match=r"No TLE found for TESTSAT at 2024-01-02 12:00:00\+00:00"
    ):
        database.lookup(PLATFORM, epoch)

    database.download.assert_called_once_with(
        PLATFORM, date(2024, 1, 1), date(2024, 1, 3)
    )


@pytest.mark.parametrize(
    ("epoch", "expected"),
    [
        ("24001.50000000", datetime(2024, 1, 1, 12, tzinfo=UTC)),
        ("56001.00000000", datetime(2056, 1, 1, tzinfo=UTC)),
        ("57001.00000000", datetime(1957, 1, 1, tzinfo=UTC)),
    ],
)
def test_epoch_to_timestamp_uses_tle_year_pivot(epoch: str, expected: datetime):
    assert space_track._epoch_to_timestamp(epoch) == expected.timestamp()

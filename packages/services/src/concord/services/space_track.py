import sqlite3
from datetime import UTC, date, datetime, timedelta
from importlib.resources import files
from pathlib import Path

import httpx
from pyorbital.orbital import Orbital


SPACE_TRACK_BASE_URL = 'https://www.space-track.org'


class TLEDatabase:
    """A database for storing and retrieving TLEs from Space-Track."""

    def __init__(self, db_path: Path, login: str, password: str):
        """Initialize the TLE database. Login credentials for Space-Track are
        required."""
        self.db_path = db_path
        self.login = login
        self.password = password
        self.conn = sqlite3.connect(db_path)
        self._ensure_tables()

    def download(self, platform: str, start_epoch: date, end_epoch: date):
        """Download TLEs for a given platform and epoch range from
        Space-Track and insert into database."""
        with httpx.Client() as client:
            cookies = self._login(client)
            url = self._space_track_url(platform, start_epoch, end_epoch)
            r = httpx.get(url, cookies=cookies)
            tle_lines = [
                line.strip()
                for line in r.text.split('\r\n')
                if line.strip() != ''
            ]
            for i in range(0, len(tle_lines), 2):
                line1 = tle_lines[i]
                line2 = tle_lines[i + 1]
                cursor = self.conn.cursor()
                cursor.execute('''
                    INSERT OR IGNORE INTO tles (platform, epoch, line1, line2)
                    VALUES ((SELECT id FROM platforms WHERE name = ?), ?, ?, ?)
                ''', (platform, _epoch_to_timestamp(line1[18:32]), line1, line2))
            self.conn.commit()

    def lookup(self, platform: str, epoch: datetime) -> Orbital:
        """Lookup TLE for a given platform and epoch, downloading data as
        required."""
        lines = self._lookup(platform, epoch, timedelta(hours=6))
        if lines is None:
            self.download(
                platform, epoch.date() - timedelta(days=1), epoch.date() + timedelta(days=1)
            )
            lines = self._lookup(platform, epoch, timedelta(days=1))
            if lines is None:
                raise ValueError(f'No TLE found for {platform} at {epoch}')
        return Orbital(platform, line1=lines[0], line2=lines[1])

    def _ensure_tables(self):
        """Ensure that the required tables exist in the database."""
        with self.conn:
            self.conn.execute('PRAGMA foreign_keys = ON')
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS platforms (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL
                )
            ''')
            def to_row(line: str) -> tuple[int, str]:
                parts = line.strip().split(' ')
                return int(parts[-1]), parts[0]
            platform_data = [
                to_row(line)
                for line in
                open(files('pyorbital.etc') / 'platforms.txt').readlines() # type: ignore
                if line.strip() and not line.startswith('#')
            ]
            self.conn.executemany('''
                INSERT OR IGNORE INTO platforms (id, name) VALUES (?, ?)
            ''', platform_data)
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS tles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    platform INTEGER NOT NULL REFERENCES platforms(id),
                    epoch REAL NOT NULL,
                    line1 TEXT NOT NULL,
                    line2 TEXT NOT NULL,
                    UNIQUE(platform, epoch)
                )
            ''')

    def _login(self, client: httpx.Client) -> httpx.Cookies:
        """Login to Space-Track and return session cookies."""
        r = client.post(
            SPACE_TRACK_BASE_URL + '/ajaxauth/login',
            data={'identity': self.login, 'password': self.password}
        )
        if r.status_code != 200:
            raise ValueError(f'Login failed: {r.status_code} {r.text}')
        return r.cookies

    def _space_track_url(
            self,
            platform: int | str,
            start_epoch: date,
            end_epoch: date
    ) -> str:
        """Construct Space-Track URL for TLE download."""
        if isinstance(platform, str):
            cursor = self.conn.cursor()
            cursor.execute('SELECT id FROM platforms WHERE name = ?', (platform,))
            result = cursor.fetchone()
            if result:
                platform = result[0]
            else:
                raise ValueError(f'Unknown platform: {platform}')
        return (
            SPACE_TRACK_BASE_URL +
            '/basicspacedata/query' +
            f'/class/gp_history/NORAD_CAT_ID/{platform}' +
            '/orderby/TLE_LINE1 ASC' +
            f'/EPOCH/{start_epoch.isoformat()}--{end_epoch.isoformat()}' +
            '/format/tle'
        )

    def _lookup(
            self, platform: str, epoch: datetime, tolerance: timedelta
    ) -> tuple[str, str] | None:
        """Lookup TLE for a given platform and epoch, with a tolerance."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT line1, line2 FROM tles
            WHERE platform = (SELECT id FROM platforms WHERE name = ?)
            AND ABS(epoch - ?) <= ?
            ORDER BY ABS(epoch - ?) ASC
            LIMIT 1
        ''', (platform, epoch.timestamp(), tolerance.total_seconds(), epoch.timestamp()))
        result = cursor.fetchone()
        if result:
            return result[0], result[1]
        else:
            return None


def _epoch_to_timestamp(epoch: str) -> float:
    """Convert TLE epoch string to timestamp.

    Epoch strings are of the form 'yyddd.dddddddd', where 'yy' is the last two
    digits of the year, 'ddd' is the day of the year, and 'dddddddd' is the
    8-digit fractional part of the day.

    See https://en.wikipedia.org/wiki/Two-line_element_set
    """
    year = int(epoch[:2])
    if year < 57:
        year += 2000
    else:
        year += 1900
    day_of_year = int(epoch[2:5])
    fractional_day = float(epoch[5:])
    dt = datetime(year, 1, 1, tzinfo=UTC) + timedelta(days=day_of_year - 1 + fractional_day)
    return dt.timestamp()

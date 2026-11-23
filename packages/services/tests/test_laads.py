from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from concord.services import laads


@pytest.fixture(autouse=True)
def block_live_http(monkeypatch: pytest.MonkeyPatch):
    def fail(*args, **kwargs):
        raise AssertionError("test attempted a live HTTP request")

    monkeypatch.setattr(laads.httpx, "Client", fail)


def test_list_granules_requests_details_and_filters_acquisition_times():
    client = laads.Client("secret-token")
    client._make_request = Mock(return_value={
        "content": [
            {"self": "/archive/VNP02IMG.A2024001.0553.002.old.nc"},
            {"self": "/archive/VNP02IMG.A2024001.0554.002.start.nc"},
            {"self": "/archive/VNP02IMG.A2024001.0600.002.end.nc"},
            {"self": "/archive/VNP02IMG.A2024001.0601.002.new.nc"},
        ]
    })

    names = client.list_granules(
        "VNP02IMG",
        datetime(2024, 1, 1, 5, 54, tzinfo=UTC),
        datetime(2024, 1, 1, 6, 0, tzinfo=UTC),
    )

    assert names == [
        "VNP02IMG.A2024001.0554.002.start.nc",
        "VNP02IMG.A2024001.0600.002.end.nc",
    ]
    client._make_request.assert_called_once_with(
        "content/details",
        products="VNP02IMG",
        temporalRanges="2024-01-01T05:54..2024-01-01T06:00",
    )


def test_time_filter_handles_julian_days_and_inclusive_bounds():
    names = [
        "VNP02IMG.A2023365.2359.002.first.nc",
        "VNP02IMG.A2024001.0000.002.second.nc",
        "VNP02IMG.A2024001.0001.002.third.nc",
    ]

    assert laads.Client.time_filter(
        names,
        datetime(2023, 12, 31, 23, 59, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
    ) == names[:2]


def test_make_request_uses_bearer_token_and_returns_json(
    monkeypatch: pytest.MonkeyPatch,
):
    response = Mock()
    response.json.return_value = {"content": ["result"]}
    http_client = Mock()
    http_client.__enter__ = Mock(return_value=http_client)
    http_client.__exit__ = Mock(return_value=False)
    http_client.get.return_value = response
    client_factory = Mock(return_value=http_client)
    monkeypatch.setattr(laads.httpx, "Client", client_factory)

    result = laads.Client("secret-token")._make_request(
        "content/details", products="VNP02IMG"
    )

    assert result == {"content": ["result"]}
    client_factory.assert_called_once_with(transport=laads.transport)
    http_client.get.assert_called_once_with(
        laads.BASE_URL + "content/details",
        params={"products": "VNP02IMG"},
        headers={"Authorization": "Bearer secret-token"},
    )
    response.raise_for_status.assert_called_once_with()


class DownloadResponse:
    def __init__(self, chunks: list[bytes], content_length: int | None):
        self.chunks = chunks
        self.headers = {}
        if content_length is not None:
            self.headers["content-length"] = str(content_length)
        self.raise_for_status_called = False
        self.chunk_size = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def raise_for_status(self):
        self.raise_for_status_called = True

    def iter_bytes(self, chunk_size: int):
        self.chunk_size = chunk_size
        yield from self.chunks


class DownloadClient:
    def __init__(self, response: DownloadResponse):
        self.response = response
        self.stream_call = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def stream(self, method, url, headers):
        self.stream_call = (method, url, headers)
        return self.response


class Progress:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.updates = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def update(self, size: int):
        self.updates.append(size)


def install_download_fakes(
    monkeypatch: pytest.MonkeyPatch,
    response: DownloadResponse,
):
    http_client = DownloadClient(response)
    client_factory = Mock(return_value=http_client)
    progress_bars = []

    def make_progress(**kwargs):
        progress = Progress(**kwargs)
        progress_bars.append(progress)
        return progress

    monkeypatch.setattr(laads.httpx, "Client", client_factory)
    monkeypatch.setattr(laads, "tqdm", make_progress)
    return client_factory, http_client, progress_bars


def test_retrieve_granule_streams_file_and_logs_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    response = DownloadResponse([b"abc", b"defg"], content_length=7)
    client_factory, http_client, progress_bars = install_download_fakes(
        monkeypatch, response
    )
    logger = Mock()

    size = laads.Client("secret-token").retrieve_granule(
        tmp_path, "granule.nc", logger
    )

    assert size == 7
    assert (tmp_path / "granule.nc").read_bytes() == b"abcdefg"
    client_factory.assert_called_once_with(transport=laads.transport)
    assert http_client.stream_call == (
        "GET",
        laads.BASE_URL + "content/archives/granule.nc",
        {"Authorization": "Bearer secret-token"},
    )
    assert response.raise_for_status_called
    assert response.chunk_size == 64 * 1024
    logger.info.assert_called_once_with("Downloading %s (%s)", "granule.nc", 7)
    assert progress_bars[0].kwargs == {
        "total": 7,
        "unit": "B",
        "unit_scale": True,
        "mininterval": 5,
        "disable": None,
    }
    assert progress_bars[0].updates == [3, 4]


def test_retrieve_granule_allows_missing_content_length(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    response = DownloadResponse([b"content"], content_length=None)
    install_download_fakes(monkeypatch, response)

    size = laads.Client("token").retrieve_granule(tmp_path, "granule.nc")

    assert size == 0
    assert (tmp_path / "granule.nc").read_bytes() == b"content"


def test_retrieve_granule_rejects_incomplete_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    response = DownloadResponse([b"short"], content_length=10)
    install_download_fakes(monkeypatch, response)

    with pytest.raises(
        RuntimeError, match=r"VIIRS data download failed: granule\.nc"
    ):
        laads.Client("token").retrieve_granule(tmp_path, "granule.nc")

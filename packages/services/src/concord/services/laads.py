from datetime import datetime, UTC
import logging
import os
from pathlib import Path

import httpx
from httpx_retries import Retry, RetryTransport
from tqdm import tqdm

logger = logging.getLogger(__name__)


# The LAADS API sometimes returns 401 or 404 errors for valid requests, so we
# retry a few times.

retry = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[401, 404, 502],
    allowed_methods=['GET'],
    respect_retry_after_header=True
)
transport = RetryTransport(retry=retry)


# API specification is here:
# https://ladsweb.modaps.eosdis.nasa.gov/tools-and-services/api-v2/specs/content
BASE_URL = 'https://ladsweb.modaps.eosdis.nasa.gov/api/v2/'


class Client:
    def __init__(self, token: str):
        # Retrieve API token (required for download endpoints).
        self.token = token

    def list_granules(
            self, product: str, start: datetime, end: datetime
    ) -> list[str]:
        """List available granules for product in time range."""
        startstr = start.strftime('%Y-%m-%dT%H:%M')
        endstr = end.strftime('%Y-%m-%dT%H:%M')
        kwargs = {
            'products': product,
            'temporalRanges': f'{startstr}..{endstr}',
        }
        response = self._make_request('content/details', **kwargs)
        names = [os.path.basename(c['self']) for c in response['content']]
        return self.time_filter(names, start, end)

    @staticmethod
    def time_filter(names: list[str], start: datetime, end: datetime) -> list[str]:
        # LAADS filenames have formats made up of a series of dot-separated
        # fields like:
        #
        #   VNP02IMG.A2024001.0554.002.2024006095727.nc
        #
        # where the first field is the product name, the second field gives
        # the acquisition year and Julian day, and the third field the UTC
        # time.
        filtered = []
        for n in names:
            dstr = ''.join(n.split('.')[1:3])[1:]
            d = datetime.strptime(dstr, '%Y%j%H%M').replace(tzinfo=UTC)
            if start <= d <= end:
                filtered.append(n)
        return filtered

    def retrieve_granule(self, outdir: Path, name: str) -> int:
        """Retrieve granule file from LAADS.

        Returns retrieved file size."""
        with httpx.Client(transport=transport) as client:
            with client.stream(
                    'GET',
                    BASE_URL + 'content/archives/' + name,
                    headers={'Authorization': 'Bearer ' + self.token}
            ) as response:
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))

                logger.info('Downloading %s (%s)', name, total_size)
                downloaded = 0
                with tqdm(total=total_size, unit='B', unit_scale=True,
                        mininterval=5, disable=None) as pb:
                    with open(outdir / name, 'wb') as fp:
                        for chunk in response.iter_bytes(64 * 1024):
                            pb.update(len(chunk))
                            downloaded += len(chunk)
                            fp.write(chunk)

                if total_size != 0 and downloaded != total_size:
                    logger.error('Failed to download %s', name)
                    raise RuntimeError('VIIRS data download failed')
                return total_size

    def _make_request(self, path, **kwargs):
        """Make a LAADS API request.

        All requests are GETs, with bearer token authorization, and JSON
        responses.
        """
        with httpx.Client(transport=transport) as client:
            response = client.get(
                BASE_URL + path,
                params=kwargs,
                headers={'Authorization': 'Bearer ' + self.token})
            response.raise_for_status()
            return response.json()

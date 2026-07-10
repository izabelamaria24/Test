import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
# Nominatim's usage policy requires an identifying User-Agent and at most 1 req/sec.
USER_AGENT = "realestate-semantic-search-research/0.1"


@dataclass
class GeocodeResult:
    latitude: float
    longitude: float


def geocode_address(address: str, *, min_interval_seconds: float = 1.0) -> GeocodeResult | None:
    response = requests.get(
        NOMINATIM_URL,
        params={"q": f"{address}, Romania", "format": "json", "limit": 1},
        headers={"User-Agent": USER_AGENT},
        timeout=10,
    )
    response.raise_for_status()
    results = response.json()
    time.sleep(min_interval_seconds)
    if not results:
        return None
    return GeocodeResult(latitude=float(results[0]["lat"]), longitude=float(results[0]["lon"]))


class CachedGeocoder:
    def __init__(
        self,
        cache_path: Path,
        geocode_fn: Callable[[str], GeocodeResult | None] = geocode_address,
    ):
        self._cache_path = cache_path
        self._geocode_fn = geocode_fn
        self._cache: dict[str, dict[str, float] | None] = {}
        if cache_path.exists():
            self._cache = json.loads(cache_path.read_text())

    def geocode(self, address: str) -> GeocodeResult | None:
        if address in self._cache:
            cached = self._cache[address]
            return GeocodeResult(**cached) if cached else None

        result = self._geocode_fn(address)
        self._cache[address] = (
            {"latitude": result.latitude, "longitude": result.longitude} if result else None
        )
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(self._cache))
        return result

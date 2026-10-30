import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

from realestate.enrich.normalizer import normalize_sector

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
# Nominatim's usage policy requires an identifying User-Agent and at most 1 req/sec.
USER_AGENT = "realestate-semantic-search-research/0.1"
# Bucharest plus its immediate Ilfov edge (lon_left,lat_top,lon_right,lat_bottom), so a street
# name that also exists in another town can't resolve there.
BUCHAREST_VIEWBOX = "25.96,44.54,26.23,44.33"
_SECTOR_ADDRESS_KEYS = ("district", "city_district", "suburb", "borough")


@dataclass
class GeocodeResult:
    latitude: float
    longitude: float
    # None outside the six sectors, and for cache entries written before sectors were stored.
    sector: int | None = None


def _sector_from_address(address: dict[str, str]) -> int | None:
    for key in _SECTOR_ADDRESS_KEYS:
        if key in address and (sector := normalize_sector(address[key])) is not None:
            return sector
    return None


def geocode_address(address: str, *, min_interval_seconds: float = 1.0) -> GeocodeResult | None:
    response = requests.get(
        NOMINATIM_URL,
        params={
            "q": f"{address}, Romania",
            "format": "json",
            "limit": 1,
            "viewbox": BUCHAREST_VIEWBOX,
            "bounded": 1,
            "addressdetails": 1,
        },
        headers={"User-Agent": USER_AGENT},
        timeout=10,
    )
    response.raise_for_status()
    results = response.json()
    time.sleep(min_interval_seconds)
    if not results:
        return None
    best = results[0]
    return GeocodeResult(
        latitude=float(best["lat"]),
        longitude=float(best["lon"]),
        sector=_sector_from_address(best.get("address", {})),
    )


class CachedGeocoder:
    def __init__(
        self,
        cache_path: Path,
        geocode_fn: Callable[[str], GeocodeResult | None] = geocode_address,
    ):
        self._cache_path = cache_path
        self._geocode_fn = geocode_fn
        self._cache: dict[str, dict[str, Any] | None] = {}
        if cache_path.exists():
            try:
                self._cache = json.loads(cache_path.read_text())
            except json.JSONDecodeError:
                self._cache = {}

    def geocode(self, address: str) -> GeocodeResult | None:
        if address in self._cache:
            cached = self._cache[address]
            return GeocodeResult(**cached) if cached else None

        try:
            result = self._geocode_fn(address)
        except requests.RequestException:
            return None

        self._cache[address] = asdict(result) if result else None
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._cache_path.with_suffix(self._cache_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(self._cache))
        tmp_path.replace(self._cache_path)
        return result

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
# Nominatim's usage policy requires an identifying User-Agent and at most 1 req/sec.
USER_AGENT = "realestate-semantic-search-research/0.1"
# Bucharest plus its immediate Ilfov edge (lon_left,lat_top,lon_right,lat_bottom), so a street
# name that also exists in another town can't resolve there.
BUCHAREST_VIEWBOX = "25.96,44.54,26.23,44.33"
_MAX_CANDIDATES = 3


@dataclass
class Place:
    latitude: float
    longitude: float
    # GeoJSON: a street's line, a park's or neighbourhood's outline, or just a point.
    geometry: dict[str, Any]

    def shape(self) -> BaseGeometry:
        return shape(self.geometry)


def search_places(query: str, *, min_interval_seconds: float = 1.0) -> list[Place]:
    response = requests.get(
        NOMINATIM_URL,
        params={
            "q": f"{query}, Romania",
            "format": "json",
            "limit": _MAX_CANDIDATES,
            "viewbox": BUCHAREST_VIEWBOX,
            "bounded": 1,
            "polygon_geojson": 1,
        },
        headers={"User-Agent": USER_AGENT},
        timeout=10,
    )
    response.raise_for_status()
    results = response.json()
    time.sleep(min_interval_seconds)
    places = []
    for result in results:
        latitude, longitude = float(result["lat"]), float(result["lon"])
        point = {"type": "Point", "coordinates": [longitude, latitude]}
        places.append(Place(latitude, longitude, result.get("geojson") or point))
    return places


class CachedGeocoder:
    def __init__(
        self,
        cache_path: Path,
        search_fn: Callable[[str], list[Place]] = search_places,
    ):
        self._cache_path = cache_path
        self._search_fn = search_fn
        self._cache: dict[str, list[dict[str, Any]]] = {}
        if cache_path.exists():
            try:
                self._cache = json.loads(cache_path.read_text())
            except json.JSONDecodeError:
                self._cache = {}

    def search(self, query: str) -> list[Place]:
        if query in self._cache:
            return [Place(**cached) for cached in self._cache[query]]

        try:
            places = self._search_fn(query)
        except requests.RequestException:
            return []

        self._cache[query] = [asdict(place) for place in places]
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._cache_path.with_suffix(self._cache_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(self._cache))
        tmp_path.replace(self._cache_path)
        return places

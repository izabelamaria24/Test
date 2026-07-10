import logging
import math
from typing import Callable

import requests

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def fetch_bucharest_subway_stations() -> list[dict]:
    query = """
    [out:json][timeout:25];
    area["name"="Bucuresti"]["admin_level"="4"]->.searchArea;
    (
      node["railway"="station"]["station"="subway"](area.searchArea);
      node["public_transport"="station"]["subway"="yes"](area.searchArea);
    );
    out body;
    """
    response = requests.post(OVERPASS_URL, data={"data": query}, timeout=30)
    response.raise_for_status()
    elements = response.json()["elements"]
    return [
        {"name": el.get("tags", {}).get("name", "unknown"), "lat": el["lat"], "lon": el["lon"]}
        for el in elements
    ]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def osrm_walking_minutes(
    lat1: float, lon1: float, lat2: float, lon2: float, osrm_url: str = "http://localhost:5000"
) -> float:
    url = f"{osrm_url}/route/v1/foot/{lon1},{lat1};{lon2},{lat2}"
    response = requests.get(url, params={"overview": "false"}, timeout=10)
    response.raise_for_status()
    data = response.json()
    if data.get("code") != "Ok":
        raise ValueError(f"OSRM route failed: {data.get('code')}")
    return data["routes"][0]["duration"] / 60.0


def nearest_subway_station(
    lat: float,
    lon: float,
    stations: list[dict],
    *,
    candidates: int = 3,
    walking_fn: Callable[..., float] = osrm_walking_minutes,
    osrm_url: str = "http://localhost:5000",
) -> tuple[str, float] | None:
    if not stations:
        return None

    ranked = sorted(stations, key=lambda s: haversine_km(lat, lon, s["lat"], s["lon"]))[:candidates]

    best: tuple[str, float] | None = None
    for station in ranked:
        try:
            minutes = walking_fn(lat, lon, station["lat"], station["lon"], osrm_url=osrm_url)
        except (requests.RequestException, ValueError) as exc:
            # A single candidate's routing failure -- whether a transient OSRM/network
            # error (RequestException) or a routine routing failure such as OSRM's
            # "NoRoute"/"NoSegment" responses (ValueError) -- shouldn't abort the whole
            # lookup. Fall back to the remaining candidates, but log it so a systemic
            # outage (all candidates failing) is visible instead of looking identical
            # to "no nearby subway stations".
            logger.warning("walking_fn failed for station %s: %s", station["name"], exc)
            continue
        if best is None or minutes < best[1]:
            best = (station["name"], minutes)
    return best

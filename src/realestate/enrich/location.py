import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from realestate.enrich.geocoder import GeocodeResult
from realestate.enrich.location_clues import LocationClues
from realestate.enrich.normalizer import normalize_sector
from realestate.enrich.poi import Station
from realestate.models import RawListing
from realestate.sectors import SECTOR_WITH_NEIGHBOURS

# Rough radius, in metres, within which the apartment probably lies for each kind of clue.
# A listing "near metro X" or on a long street can be some distance from the matched point.
_UNCERTAINTY_M = {
    "building": 100,
    "street": 1000,
    "metro": 1000,
    "landmark": 1500,
    "neighbourhood": 2000,
}
# Used when OLX blurs the map point without saying by how much, and for a sector-level address.
_COARSE_UNCERTAINTY_M = 5000


class Geocoder(Protocol):
    def geocode(self, address: str) -> GeocodeResult | None: ...


@dataclass(frozen=True)
class Location:
    latitude: float
    longitude: float
    source: str
    uncertainty_m: float


def _station_key(name: str) -> str:
    # "Stația de metrou Piața Unirii" and OSM's "Piața Unirii 2" (line-numbered) both -> "piata unirii".
    text = unicodedata.normalize("NFD", name.lower()).encode("ascii", "ignore").decode()
    text = re.sub(r"^\s*(?:statia de\s+)?metro(?:ul|u)?\s+", "", text)
    return re.sub(r"\s+\d+\s*$", "", text).strip()


def _find_station(name: str, stations: list[Station]) -> Station | None:
    key = _station_key(name)
    return next((station for station in stations if _station_key(station["name"]) == key), None)


def _geocoded(
    source: str, query: str, geocoder: Geocoder, allowed_sectors: tuple[int, ...] | None
) -> list[Location]:
    result = geocoder.geocode(query)
    if result is None:
        return []
    # A match far from the listing's sector is most likely a same-named place elsewhere in the
    # city. Neighbouring sectors are allowed: borders run along streets, so a building is easily
    # listed under the sector next door.
    if allowed_sectors is not None and result.sector is not None:
        if result.sector not in allowed_sectors:
            return []
    return [Location(result.latitude, result.longitude, source, _UNCERTAINTY_M[source])]


def _clue_locations(
    clues: LocationClues,
    geocoder: Geocoder,
    stations: list[Station],
    allowed_sectors: tuple[int, ...] | None,
) -> Iterator[Location]:
    """Lazily resolve clues from most to least precise, so lookups stop at the first hit."""
    if clues.street and clues.street_number:
        query = f"{clues.street} {clues.street_number}, București"
        yield from _geocoded("building", query, geocoder, allowed_sectors)
    if clues.street:
        yield from _geocoded("street", f"{clues.street}, București", geocoder, allowed_sectors)
    if clues.metro_station and (station := _find_station(clues.metro_station, stations)):
        yield Location(station["lat"], station["lon"], "metro", _UNCERTAINTY_M["metro"])
    for landmark in clues.landmarks:
        yield from _geocoded("landmark", f"{landmark}, București", geocoder, allowed_sectors)
    if clues.neighbourhood:
        query = f"{clues.neighbourhood}, București"
        yield from _geocoded("neighbourhood", query, geocoder, allowed_sectors)


def locate_listing(
    raw: RawListing, clues: LocationClues, *, geocoder: Geocoder, stations: list[Station]
) -> Location | None:
    """Best available position for a listing, from the most to the least precise clue."""
    listing_sector = normalize_sector(raw.address_raw)
    allowed_sectors = SECTOR_WITH_NEIGHBOURS[listing_sector] if listing_sector else None
    location = next(_clue_locations(clues, geocoder, stations, allowed_sectors), None)
    if location is not None:
        return location

    if raw.map_lat is not None and raw.map_lon is not None:
        uncertainty_m = (
            raw.map_radius_km * 1000 if raw.map_radius_km is not None else _COARSE_UNCERTAINTY_M
        )
        return Location(raw.map_lat, raw.map_lon, "olx_approximate", uncertainty_m)

    result = geocoder.geocode(raw.address_raw)
    if result is None:
        return None
    return Location(result.latitude, result.longitude, "address", _COARSE_UNCERTAINTY_M)

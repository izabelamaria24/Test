import re
import unicodedata
from typing import Protocol

from shapely.geometry import Point

from realestate.enrich.fusion import Evidence, FusedLocation, fuse
from realestate.enrich.geocoder import Place
from realestate.enrich.location_clues import LocationClues
from realestate.enrich.normalizer import normalize_sector
from realestate.enrich.poi import Station
from realestate.models import RawListing

# How far from the matched place the apartment may be, in metres, and how often the clue is
# right at all (the LLM misreads, or the name resolves to a different place). Ads list
# "nearby" malls and parks within driving distance and use neighbourhood names loosely, while
# an explicit street and number is rarely wrong.
_BUILDING_REACH_M, _BUILDING_RELIABILITY = 75, 0.9
_STREET_REACH_M, _STREET_RELIABILITY = 75, 0.85
_METRO_REACH_M, _METRO_RELIABILITY = 1000, 0.85
_LANDMARK_REACH_M, _LANDMARK_RELIABILITY = 1500, 0.6
_NEIGHBOURHOOD_OUTLINE_REACH_M, _NEIGHBOURHOOD_POINT_REACH_M = 1000, 2000
_NEIGHBOURHOOD_RELIABILITY = 0.7
# Sellers name the sector next door for border streets, and sector 6 for new blocks just over
# the city limit (e.g. Militari Residence), so the sector outline is only loosely enforced.
_SECTOR_REACH_M, _SECTOR_RELIABILITY = 1500, 0.95
_OLX_RELIABILITY = 0.9
_OLX_REACH_WITHOUT_RADIUS_M = 5000
_WALKING_M_PER_MINUTE = 80


class PlaceSearcher(Protocol):
    def search(self, query: str) -> list[Place]: ...


def _station_key(name: str) -> str:
    # "Stația de metrou Piața Unirii" and OSM's "Piața Unirii 2" (line-numbered) both -> "piata unirii".
    text = unicodedata.normalize("NFD", name.lower()).encode("ascii", "ignore").decode()
    text = re.sub(r"^\s*(?:statia de\s+)?metro(?:ul|u)?\s+", "", text)
    return re.sub(r"\s+\d+\s*$", "", text).strip()


def _find_station(name: str, stations: list[Station]) -> Station | None:
    key = _station_key(name)
    return next((station for station in stations if _station_key(station["name"]) == key), None)


def _searched(
    kind: str, query: str, geocoder: PlaceSearcher, reach_m: float, reliability: float
) -> list[Evidence]:
    places = geocoder.search(f"{query}, București")
    if not places and kind in ("building", "street"):
        # New blocks just over the city limit (e.g. Militari Residence) are listed by sector;
        # the search is still bounded to the Bucharest area.
        places = geocoder.search(query)
    if not places:
        return []
    return [Evidence(kind, tuple(place.shape() for place in places), reach_m, reliability)]


def _neighbourhood(name: str, geocoder: PlaceSearcher) -> list[Evidence]:
    shapes = [place.shape() for place in geocoder.search(f"{name}, București")]
    outlines = tuple(shape for shape in shapes if shape.geom_type in ("Polygon", "MultiPolygon"))
    if outlines:
        reach_m = _NEIGHBOURHOOD_OUTLINE_REACH_M
        return [Evidence("neighbourhood", outlines, reach_m, _NEIGHBOURHOOD_RELIABILITY)]
    if shapes:
        reach_m = _NEIGHBOURHOOD_POINT_REACH_M
        return [Evidence("neighbourhood", tuple(shapes), reach_m, _NEIGHBOURHOOD_RELIABILITY)]
    return []


def gather_evidence(
    raw: RawListing,
    clues: LocationClues,
    *,
    geocoder: PlaceSearcher,
    stations: list[Station],
    olx_point_is_default: bool = False,
) -> list[Evidence]:
    evidence: list[Evidence] = []

    sector = normalize_sector(raw.address_raw)
    if sector is not None:
        query = f"Sector {sector}"
        evidence += _searched("area", query, geocoder, _SECTOR_REACH_M, _SECTOR_RELIABILITY)

    # A point OLX shares across many listings is its per-sector default, not the apartment.
    if raw.map_lat is not None and raw.map_lon is not None and not olx_point_is_default:
        reach_m = (
            raw.map_radius_km * 1000
            if raw.map_radius_km is not None
            else _OLX_REACH_WITHOUT_RADIUS_M
        )
        point = Point(raw.map_lon, raw.map_lat)
        evidence.append(Evidence("olx_point", (point,), reach_m, _OLX_RELIABILITY))

    building: list[Evidence] = []
    if clues.street and clues.street_number:
        query = f"{clues.street} {clues.street_number}"
        building = _searched("building", query, geocoder, _BUILDING_REACH_M, _BUILDING_RELIABILITY)
    evidence += building
    if clues.street and not building:
        query = clues.street
        evidence += _searched("street", query, geocoder, _STREET_REACH_M, _STREET_RELIABILITY)

    if clues.metro_station and (station := _find_station(clues.metro_station, stations)):
        reach_m = (
            clues.metro_walk_minutes * _WALKING_M_PER_MINUTE
            if clues.metro_walk_minutes
            else _METRO_REACH_M
        )
        point = Point(station["lon"], station["lat"])
        evidence.append(Evidence("metro", (point,), reach_m, _METRO_RELIABILITY))

    for landmark in clues.landmarks:
        evidence += _searched(
            "landmark", landmark, geocoder, _LANDMARK_REACH_M, _LANDMARK_RELIABILITY
        )

    if clues.neighbourhood:
        evidence += _neighbourhood(clues.neighbourhood, geocoder)

    return evidence


def locate_listing(
    raw: RawListing,
    clues: LocationClues,
    *,
    geocoder: PlaceSearcher,
    stations: list[Station],
    olx_point_is_default: bool = False,
) -> FusedLocation | None:
    evidence = gather_evidence(
        raw,
        clues,
        geocoder=geocoder,
        stations=stations,
        olx_point_is_default=olx_point_is_default,
    )
    return fuse(evidence)

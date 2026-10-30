from collections.abc import Callable

from realestate.enrich.geocoder import CachedGeocoder
from realestate.enrich.location import locate_listing
from realestate.enrich.location_clues import LocationClueExtractor, LocationClues
from realestate.enrich.normalizer import (
    normalize_price_eur,
    normalize_rooms,
    normalize_sector,
    normalize_specs,
)
from realestate.enrich.poi import Station, nearest_subway_station
from realestate.models import EnrichedListing, RawListing

# Walking time to the subway is meaningless if the start point could be further off than this.
MAX_UNCERTAINTY_FOR_SUBWAY_M = 1000


def enrich_listing(
    raw: RawListing,
    *,
    geocoder: CachedGeocoder,
    stations: list[Station],
    clue_extractor: LocationClueExtractor | None = None,
    olx_point_is_default: bool = False,
    nearest_station_fn: Callable[..., tuple[str, float] | None] = nearest_subway_station,
) -> EnrichedListing:
    specs = normalize_specs(raw.specs)

    clues = (
        clue_extractor.extract(raw.title, raw.description) if clue_extractor else LocationClues()
    )
    location = locate_listing(
        raw,
        clues,
        geocoder=geocoder,
        stations=stations,
        olx_point_is_default=olx_point_is_default,
    )
    is_precise = location is not None and location.uncertainty_m <= MAX_UNCERTAINTY_FOR_SUBWAY_M

    station_name: str | None = None
    walking_minutes: float | None = None
    if location is not None and is_precise:
        nearest = nearest_station_fn(location.latitude, location.longitude, stations)
        if nearest is not None:
            station_name, walking_minutes = nearest

    return EnrichedListing(
        external_id=raw.external_id,
        title=raw.title,
        description=raw.description,
        price_eur=normalize_price_eur(raw.price_raw),
        rooms=normalize_rooms(raw.title),
        built_area_sqm=specs.get("built_area_sqm"),
        floor_number=specs.get("floor_number"),
        construction_year_range=specs.get("construction_year_range"),
        layout_type=specs.get("layout_type"),
        address_text=raw.address_raw,
        sector=normalize_sector(raw.address_raw),
        latitude=location.latitude if location else None,
        longitude=location.longitude if location else None,
        location_confidence="ok" if is_precise else "low_confidence_location",
        location_evidence=list(location.agreeing) if location else [],
        location_conflicts=list(location.conflicting) if location else [],
        location_uncertainty_m=location.uncertainty_m if location else None,
        nearest_subway_station=station_name,
        subway_walking_minutes=walking_minutes,
        image_urls=raw.image_urls,
    )

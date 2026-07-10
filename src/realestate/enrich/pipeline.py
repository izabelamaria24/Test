from typing import Callable

from realestate.enrich.geocoder import CachedGeocoder
from realestate.enrich.normalizer import normalize_price_eur, normalize_rooms, normalize_specs
from realestate.enrich.poi import nearest_subway_station
from realestate.models import EnrichedListing, RawListing


def enrich_listing(
    raw: RawListing,
    *,
    geocoder: CachedGeocoder,
    stations: list[dict],
    nearest_station_fn: Callable[..., tuple[str, float] | None] = nearest_subway_station,
) -> EnrichedListing:
    specs = normalize_specs(raw.specs)

    if raw.map_lat is not None and raw.map_lon is not None:
        latitude, longitude = raw.map_lat, raw.map_lon
    else:
        geocode_result = geocoder.geocode(raw.address_raw)
        latitude = geocode_result.latitude if geocode_result else None
        longitude = geocode_result.longitude if geocode_result else None

    location_confidence = "ok" if latitude is not None else "low_confidence_location"

    station_name: str | None = None
    walking_minutes: float | None = None
    if latitude is not None:
        nearest = nearest_station_fn(latitude, longitude, stations)
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
        latitude=latitude,
        longitude=longitude,
        location_confidence=location_confidence,
        nearest_subway_station=station_name,
        subway_walking_minutes=walking_minutes,
        image_urls=raw.image_urls,
    )

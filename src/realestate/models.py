from pydantic import BaseModel


class RawListing(BaseModel):
    external_id: str
    title: str
    description: str
    price_raw: str
    address_raw: str
    specs: dict[str, str]
    image_urls: list[str]
    source_file: str
    # OLX's own JSON gives an approximate lat/lon per listing (see ad.map in the JSON schema) —
    # privacy-fuzzed (radius + show_detailed:false observed on every real listing sampled so far),
    # but free (no network call) and often finer-grained than geocoding the sparse text address.
    # None when a listing's JSON omits map data (not observed yet, but the schema doesn't guarantee it).
    map_lat: float | None
    map_lon: float | None
    # Blur radius OLX draws around the map point; km, judging by the values (2-5) next to zoom 11-13.
    map_radius_km: float | None = None


class EnrichedListing(BaseModel):
    external_id: str
    title: str
    description: str
    price_eur: float
    rooms: int | None
    built_area_sqm: float | None
    floor_number: int | None
    construction_year_range: str | None
    layout_type: str | None
    address_text: str
    # Defaults to None so payloads stored before this field existed still load.
    sector: int | None = None
    latitude: float | None
    longitude: float | None
    location_confidence: str
    # Which clue the coordinates came from (e.g. "building", "olx_approximate") and roughly how
    # far off they may be. Default to None so payloads stored before these fields still load.
    location_source: str | None = None
    location_uncertainty_m: float | None = None
    nearest_subway_station: str | None
    subway_walking_minutes: float | None
    image_urls: list[str]

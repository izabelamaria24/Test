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
    latitude: float | None
    longitude: float | None
    location_confidence: str
    nearest_subway_station: str | None
    subway_walking_minutes: float | None
    image_urls: list[str]

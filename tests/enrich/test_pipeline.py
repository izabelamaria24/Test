import math

from realestate.enrich.geocoder import CachedGeocoder, Place
from realestate.enrich.location_clues import LocationClues
from realestate.enrich.pipeline import enrich_listing
from realestate.models import RawListing

RAW_WITH_MAP = RawListing(
    external_id="304473136",
    title="Vand apartament 2 camere TITAN",
    description="Direct proprietar, apartament 2 camere decomandat.",
    price_raw="100.000 €",
    address_raw="Bucuresti - Ilfov, Bucuresti, Sectorul 3",
    specs={"Suprafata utila": "48 m²", "Etaj": "3"},
    image_urls=["https://example.com/a.jpg"],
    source_file="304473136.html",
    map_lat=44.42,
    map_lon=26.1,
    map_radius_km=3,
)

STATIONS = [{"name": "Universitate", "lat": 44.4356, "lon": 26.1023}]
BUILDING = Place(44.4325, 26.1013, {"type": "Point", "coordinates": [26.1013, 44.4325]})


class FakeClueExtractor:
    def __init__(self, clues: LocationClues):
        self._clues = clues
        self.calls: list[tuple[str, str]] = []

    def extract(self, title: str, description: str) -> LocationClues:
        self.calls.append((title, description))
        return self._clues


def _geocoder(tmp_path, results: dict[str, list[Place]]) -> CachedGeocoder:
    return CachedGeocoder(tmp_path / "places.json", search_fn=lambda q: results.get(q, []))


def _fake_nearest_station(lat, lon, stations, **kwargs):
    return ("Universitate", 6.5)


def _fail_nearest_station(*args, **kwargs):
    raise AssertionError("subway time should not be computed for an imprecise location")


def test_enrich_listing_uses_precise_location_from_listing_text(tmp_path):
    geocoder = _geocoder(tmp_path, {"Strada Sibiu 12, București": [BUILDING]})
    extractor = FakeClueExtractor(LocationClues(street="Strada Sibiu", street_number="12"))

    enriched = enrich_listing(
        RAW_WITH_MAP,
        geocoder=geocoder,
        stations=STATIONS,
        clue_extractor=extractor,
        nearest_station_fn=_fake_nearest_station,
    )

    assert extractor.calls == [(RAW_WITH_MAP.title, RAW_WITH_MAP.description)]
    assert enriched.price_eur == 100000.0
    assert enriched.rooms == 2
    assert enriched.built_area_sqm == 48.0
    assert enriched.floor_number == 3
    assert enriched.sector == 3
    assert enriched.latitude is not None and enriched.longitude is not None
    assert math.isclose(enriched.latitude, 44.4325, abs_tol=0.001)
    assert math.isclose(enriched.longitude, 26.1013, abs_tol=0.001)
    assert enriched.location_evidence == ["olx_point", "building"]
    assert enriched.location_conflicts == []
    assert enriched.location_uncertainty_m is not None
    assert enriched.location_uncertainty_m < 300
    assert enriched.location_confidence == "ok"
    assert enriched.nearest_subway_station == "Universitate"
    assert enriched.subway_walking_minutes == 6.5


def test_enrich_listing_skips_subway_time_for_blurred_olx_point(tmp_path):
    enriched = enrich_listing(
        RAW_WITH_MAP,
        geocoder=_geocoder(tmp_path, {}),
        stations=STATIONS,
        nearest_station_fn=_fail_nearest_station,
    )

    assert enriched.location_evidence == ["olx_point"]
    assert enriched.location_uncertainty_m is not None
    assert enriched.location_uncertainty_m > 1000
    assert enriched.location_confidence == "low_confidence_location"
    assert enriched.nearest_subway_station is None
    assert enriched.subway_walking_minutes is None


def test_listed_street_address_beats_the_street_guessed_from_the_text(tmp_path):
    raw = RAW_WITH_MAP.model_copy(
        update={
            "listing_type": "rent",
            "street": "Strada Sibiu",
            "street_number": "12",
            "neighbourhood": "Drumul Taberei",
        }
    )
    searched: list[str] = []

    def search(query: str) -> list[Place]:
        searched.append(query)
        return [BUILDING] if query == "Strada Sibiu 12, București" else []

    extractor = FakeClueExtractor(
        LocationClues(street="Strada Gresita", street_number="99", metro_station="Dristor")
    )

    enriched = enrich_listing(
        raw,
        geocoder=CachedGeocoder(tmp_path / "places.json", search_fn=search),
        stations=STATIONS,
        clue_extractor=extractor,
        nearest_station_fn=_fake_nearest_station,
    )

    assert enriched.listing_type == "rent"
    assert "building" in enriched.location_evidence
    assert "Strada Gresita 99, București" not in searched
    assert "Drumul Taberei, București" in searched


def test_enrich_listing_has_no_location_without_any_usable_evidence(tmp_path):
    enriched = enrich_listing(
        RAW_WITH_MAP,
        geocoder=_geocoder(tmp_path, {}),
        stations=STATIONS,
        olx_point_is_default=True,
        nearest_station_fn=_fail_nearest_station,
    )

    assert enriched.latitude is None
    assert enriched.longitude is None
    assert enriched.location_evidence == []
    assert enriched.location_uncertainty_m is None
    assert enriched.location_confidence == "low_confidence_location"

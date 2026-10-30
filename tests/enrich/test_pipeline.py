from realestate.enrich.geocoder import CachedGeocoder, GeocodeResult
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
)

RAW_WITHOUT_MAP = RAW_WITH_MAP.model_copy(update={"map_lat": None, "map_lon": None})

STATIONS = [{"name": "Universitate", "lat": 44.4356, "lon": 26.1023}]


class FakeClueExtractor:
    def __init__(self, clues: LocationClues):
        self._clues = clues
        self.calls: list[tuple[str, str]] = []

    def extract(self, title: str, description: str) -> LocationClues:
        self.calls.append((title, description))
        return self._clues


def _fake_nearest_station(lat, lon, stations, **kwargs):
    return ("Universitate", 6.5)


def test_enrich_listing_uses_precise_location_from_listing_text(tmp_path):
    geocoder = CachedGeocoder(
        cache_path=tmp_path / "cache.json",
        geocode_fn=lambda address: GeocodeResult(latitude=44.4325, longitude=26.1013, sector=3),
    )
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
    assert enriched.latitude == 44.4325
    assert enriched.longitude == 26.1013
    assert enriched.location_source == "building"
    assert enriched.location_uncertainty_m == 100
    assert enriched.location_confidence == "ok"
    assert enriched.nearest_subway_station == "Universitate"
    assert enriched.subway_walking_minutes == 6.5


def test_enrich_listing_skips_subway_time_for_blurred_olx_point(tmp_path):
    def fail_if_called(address):
        raise AssertionError("geocoder should not be called without text clues")

    geocoder = CachedGeocoder(cache_path=tmp_path / "cache.json", geocode_fn=fail_if_called)

    def fail_nearest_station(*args, **kwargs):
        raise AssertionError("subway time should not be computed for an imprecise location")

    enriched = enrich_listing(
        RAW_WITH_MAP, geocoder=geocoder, stations=STATIONS, nearest_station_fn=fail_nearest_station
    )

    assert enriched.latitude == 44.42
    assert enriched.longitude == 26.1
    assert enriched.location_source == "olx_approximate"
    assert enriched.location_confidence == "low_confidence_location"
    assert enriched.nearest_subway_station is None
    assert enriched.subway_walking_minutes is None


def test_enrich_listing_falls_back_to_geocoding_the_address_when_map_absent(tmp_path):
    geocoder = CachedGeocoder(
        cache_path=tmp_path / "cache.json",
        geocode_fn=lambda address: GeocodeResult(latitude=44.4325, longitude=26.1013),
    )

    enriched = enrich_listing(
        RAW_WITHOUT_MAP,
        geocoder=geocoder,
        stations=STATIONS,
        nearest_station_fn=_fake_nearest_station,
    )

    assert enriched.latitude == 44.4325
    assert enriched.location_source == "address"
    assert enriched.location_confidence == "low_confidence_location"
    assert enriched.nearest_subway_station is None


def test_enrich_listing_flags_low_confidence_location_when_map_absent_and_geocode_fails(tmp_path):
    geocoder = CachedGeocoder(
        cache_path=tmp_path / "cache.json",
        geocode_fn=lambda address: None,
    )

    enriched = enrich_listing(
        RAW_WITHOUT_MAP,
        geocoder=geocoder,
        stations=STATIONS,
        nearest_station_fn=lambda *a, **k: None,
    )

    assert enriched.latitude is None
    assert enriched.location_source is None
    assert enriched.location_uncertainty_m is None
    assert enriched.location_confidence == "low_confidence_location"
    assert enriched.nearest_subway_station is None

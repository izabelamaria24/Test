from realestate.enrich.geocoder import CachedGeocoder, GeocodeResult
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


def test_enrich_listing_uses_map_coordinates_without_geocoding(tmp_path):
    def fail_if_called(address):
        raise AssertionError("geocoder should not be called when map_lat/map_lon are present")

    geocoder = CachedGeocoder(cache_path=tmp_path / "cache.json", geocode_fn=fail_if_called)

    def fake_nearest_station(lat, lon, stations, **kwargs):
        return ("Universitate", 6.5)

    enriched = enrich_listing(
        RAW_WITH_MAP, geocoder=geocoder, stations=STATIONS, nearest_station_fn=fake_nearest_station
    )

    assert enriched.price_eur == 100000.0
    assert enriched.rooms == 2
    assert enriched.built_area_sqm == 48.0
    assert enriched.floor_number == 3
    assert enriched.latitude == 44.42
    assert enriched.longitude == 26.1
    assert enriched.location_confidence == "ok"
    assert enriched.nearest_subway_station == "Universitate"
    assert enriched.subway_walking_minutes == 6.5


def test_enrich_listing_falls_back_to_geocoding_when_map_absent(tmp_path):
    geocoder = CachedGeocoder(
        cache_path=tmp_path / "cache.json",
        geocode_fn=lambda address: GeocodeResult(latitude=44.4325, longitude=26.1013),
    )

    def fake_nearest_station(lat, lon, stations, **kwargs):
        return ("Universitate", 6.5)

    enriched = enrich_listing(
        RAW_WITHOUT_MAP, geocoder=geocoder, stations=STATIONS, nearest_station_fn=fake_nearest_station
    )

    assert enriched.latitude == 44.4325
    assert enriched.location_confidence == "ok"
    assert enriched.nearest_subway_station == "Universitate"


def test_enrich_listing_flags_low_confidence_location_when_map_absent_and_geocode_fails(tmp_path):
    geocoder = CachedGeocoder(
        cache_path=tmp_path / "cache.json",
        geocode_fn=lambda address: None,
    )

    enriched = enrich_listing(
        RAW_WITHOUT_MAP, geocoder=geocoder, stations=STATIONS, nearest_station_fn=lambda *a, **k: None
    )

    assert enriched.latitude is None
    assert enriched.location_confidence == "low_confidence_location"
    assert enriched.nearest_subway_station is None

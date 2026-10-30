from realestate.enrich.geocoder import GeocodeResult
from realestate.enrich.location import Location, locate_listing
from realestate.enrich.location_clues import LocationClues
from realestate.models import RawListing

RAW = RawListing(
    external_id="1",
    title="Apartament 2 camere",
    description="d",
    price_raw="100.000 €",
    address_raw="Bucuresti - Ilfov, Bucuresti, Sectorul 3",
    specs={},
    image_urls=[],
    source_file="1.html",
    map_lat=44.41,
    map_lon=26.16,
    map_radius_km=3,
)
RAW_WITHOUT_MAP = RAW.model_copy(update={"map_lat": None, "map_lon": None, "map_radius_km": None})
STATIONS = [
    {"name": "Dristor 1", "lat": 44.42, "lon": 26.14},
    {"name": "Piața Unirii 2", "lat": 44.43, "lon": 26.10},
]


class FakeGeocoder:
    def __init__(self, results: dict[str, GeocodeResult]):
        self._results = results
        self.queries: list[str] = []

    def geocode(self, address: str) -> GeocodeResult | None:
        self.queries.append(address)
        return self._results.get(address)


def _in_sector(sector: int | None, latitude: float = 44.42) -> GeocodeResult:
    return GeocodeResult(latitude=latitude, longitude=26.1, sector=sector)


def test_prefers_building_address_over_coarser_clues():
    clues = LocationClues(
        street="Strada Sibiu",
        street_number="12",
        neighbourhood="Titan",
        metro_station="Dristor",
        landmarks=["Parcul IOR"],
    )
    geocoder = FakeGeocoder(
        {
            "Strada Sibiu 12, București": _in_sector(3, 44.1),
            "Titan, București": _in_sector(3, 44.2),
        }
    )

    location = locate_listing(RAW, clues, geocoder=geocoder, stations=STATIONS)

    assert location == Location(latitude=44.1, longitude=26.1, source="building", uncertainty_m=100)
    assert geocoder.queries == ["Strada Sibiu 12, București"]


def test_tries_clues_from_most_to_least_precise():
    clues = LocationClues(
        street="Strada Sibiu",
        neighbourhood="Titan",
        metro_station="Gara de Nord",
        landmarks=["Parcul IOR", "Mall Vitan"],
    )
    geocoder = FakeGeocoder({"Titan, București": _in_sector(3)})

    location = locate_listing(RAW, clues, geocoder=geocoder, stations=STATIONS)

    assert location is not None
    assert location.source == "neighbourhood"
    assert geocoder.queries == [
        "Strada Sibiu, București",
        "Parcul IOR, București",
        "Mall Vitan, București",
        "Titan, București",
    ]


def test_metro_clue_is_matched_against_known_stations_ignoring_diacritics_and_case():
    clues = LocationClues(metro_station="piata unirii", landmarks=["Parcul IOR"])
    geocoder = FakeGeocoder({"Parcul IOR, București": _in_sector(3)})

    location = locate_listing(RAW, clues, geocoder=geocoder, stations=STATIONS)

    assert location == Location(latitude=44.43, longitude=26.10, source="metro", uncertainty_m=1000)
    assert geocoder.queries == []


def test_street_metro_landmark_and_neighbourhood_have_growing_uncertainty():
    sources = {}
    for source, clues, query in [
        ("street", LocationClues(street="Strada Sibiu"), "Strada Sibiu, București"),
        ("metro", LocationClues(metro_station="Dristor"), "unused"),
        ("landmark", LocationClues(landmarks=["Parcul IOR"]), "Parcul IOR, București"),
        ("neighbourhood", LocationClues(neighbourhood="Titan"), "Titan, București"),
    ]:
        geocoder = FakeGeocoder({query: _in_sector(3)})
        location = locate_listing(RAW, clues, geocoder=geocoder, stations=STATIONS)
        assert location is not None
        assert location.source == source
        sources[source] = location.uncertainty_m

    assert sources["street"] <= sources["metro"] < sources["landmark"] < sources["neighbourhood"]


def test_accepts_a_match_in_a_neighbouring_sector():
    # Sector borders run along streets, so a building can be listed under the sector next door.
    clues = LocationClues(street="Bulevardul Geniului", street_number="2-4")
    geocoder = FakeGeocoder({"Bulevardul Geniului 2-4, București": _in_sector(2)})

    location = locate_listing(RAW, clues, geocoder=geocoder, stations=STATIONS)

    assert location is not None
    assert location.source == "building"


def test_skips_a_match_in_a_sector_not_bordering_the_listing():
    clues = LocationClues(neighbourhood="Universitate", landmarks=["Parcul IOR"])
    geocoder = FakeGeocoder(
        {"Universitate, București": _in_sector(1), "Parcul IOR, București": _in_sector(3)}
    )

    location = locate_listing(RAW, clues, geocoder=geocoder, stations=STATIONS)

    assert location is not None
    assert location.source == "landmark"


def test_accepts_a_match_whose_sector_is_unknown():
    clues = LocationClues(street="Strada Rezervelor", street_number="65")
    geocoder = FakeGeocoder({"Strada Rezervelor 65, București": _in_sector(None)})

    location = locate_listing(RAW, clues, geocoder=geocoder, stations=STATIONS)

    assert location is not None
    assert location.source == "building"


def test_accepts_any_sector_when_the_listing_has_none():
    raw = RAW.model_copy(update={"address_raw": "Bucuresti - Ilfov, Chiajna"})
    geocoder = FakeGeocoder({"Titan, București": _in_sector(3)})

    location = locate_listing(
        raw, LocationClues(neighbourhood="Titan"), geocoder=geocoder, stations=STATIONS
    )

    assert location is not None
    assert location.source == "neighbourhood"


def test_falls_back_to_the_blurred_olx_point_using_its_radius():
    location = locate_listing(RAW, LocationClues(), geocoder=FakeGeocoder({}), stations=STATIONS)

    assert location == Location(
        latitude=44.41, longitude=26.16, source="olx_approximate", uncertainty_m=3000
    )


def test_olx_point_without_radius_is_treated_as_very_uncertain():
    raw = RAW.model_copy(update={"map_radius_km": None})

    location = locate_listing(raw, LocationClues(), geocoder=FakeGeocoder({}), stations=STATIONS)

    assert location is not None
    assert location.uncertainty_m == 5000


def test_falls_back_to_geocoding_the_sector_address_without_olx_point():
    geocoder = FakeGeocoder({RAW_WITHOUT_MAP.address_raw: _in_sector(3)})

    location = locate_listing(
        RAW_WITHOUT_MAP, LocationClues(), geocoder=geocoder, stations=STATIONS
    )

    assert location is not None
    assert location.source == "address"
    assert location.uncertainty_m == 5000


def test_returns_none_when_nothing_can_be_located():
    location = locate_listing(
        RAW_WITHOUT_MAP, LocationClues(), geocoder=FakeGeocoder({}), stations=STATIONS
    )

    assert location is None

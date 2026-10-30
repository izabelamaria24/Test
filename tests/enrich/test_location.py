import math

from realestate.enrich.geocoder import Place
from realestate.enrich.location import gather_evidence, locate_listing
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
STATIONS = [
    {"name": "Dristor 1", "lat": 44.42, "lon": 26.14},
    {"name": "Piața Unirii 2", "lat": 44.43, "lon": 26.10},
]
SECTOR_3 = {
    "type": "Polygon",
    "coordinates": [
        [[26.09, 44.38], [26.22, 44.38], [26.22, 44.44], [26.09, 44.44], [26.09, 44.38]]
    ],
}


def _point(lon: float, lat: float) -> Place:
    return Place(lat, lon, {"type": "Point", "coordinates": [lon, lat]})


class FakeGeocoder:
    def __init__(self, results: dict[str, list[Place]]):
        self._results = results
        self.queries: list[str] = []

    def search(self, query: str) -> list[Place]:
        self.queries.append(query)
        return self._results.get(query, [])


def _kinds(evidence) -> list[str]:
    return [item.kind for item in evidence]


def test_sector_and_olx_point_are_evidence_even_without_text_clues():
    geocoder = FakeGeocoder({"Sector 3, București": [Place(44.41, 26.15, SECTOR_3)]})

    evidence = gather_evidence(RAW, LocationClues(), geocoder=geocoder, stations=STATIONS)

    assert _kinds(evidence) == ["area", "olx_point"]
    assert evidence[1].reach_m == 3000


def test_olx_default_sector_point_is_not_evidence():
    evidence = gather_evidence(
        RAW,
        LocationClues(),
        geocoder=FakeGeocoder({}),
        stations=STATIONS,
        olx_point_is_default=True,
    )

    assert evidence == []


def test_olx_point_without_radius_is_wide():
    raw = RAW.model_copy(update={"map_radius_km": None})

    evidence = gather_evidence(raw, LocationClues(), geocoder=FakeGeocoder({}), stations=[])

    assert evidence[0].reach_m == 5000


def test_listing_without_a_sector_has_no_area_evidence():
    raw = RAW.model_copy(update={"address_raw": "Bucuresti - Ilfov, Chiajna", "map_lat": None})

    evidence = gather_evidence(raw, LocationClues(), geocoder=FakeGeocoder({}), stations=[])

    assert evidence == []


def test_every_text_clue_becomes_evidence_with_all_candidates():
    clues = LocationClues(
        street="Strada Sibiu",
        street_number="12",
        neighbourhood="Titan",
        metro_station="Dristor",
        landmarks=["Parcul IOR", "Mall Vitan"],
    )
    geocoder = FakeGeocoder(
        {
            "Strada Sibiu 12, București": [_point(26.15, 44.42), _point(26.0, 44.5)],
            "Parcul IOR, București": [_point(26.16, 44.41)],
            "Titan, București": [_point(26.17, 44.42)],
        }
    )
    raw = RAW.model_copy(update={"map_lat": None})

    evidence = gather_evidence(raw, clues, geocoder=geocoder, stations=STATIONS)

    assert _kinds(evidence) == ["building", "metro", "landmark", "neighbourhood"]
    assert len(evidence[0].candidates) == 2
    assert "Strada Sibiu, București" not in geocoder.queries


def test_building_just_outside_the_city_limits_is_still_found():
    # New blocks on the Ilfov edge (e.g. Militari Residence) are listed under sector 6.
    clues = LocationClues(street="Strada Rezervelor", street_number="65")
    geocoder = FakeGeocoder({"Strada Rezervelor 65": [_point(25.97, 44.43)]})
    raw = RAW.model_copy(update={"map_lat": None})

    evidence = gather_evidence(raw, clues, geocoder=geocoder, stations=[])

    assert _kinds(evidence) == ["building"]
    assert geocoder.queries == [
        "Sector 3, București",
        "Strada Rezervelor 65, București",
        "Strada Rezervelor 65",
    ]


def test_street_is_used_when_the_building_is_not_found():
    clues = LocationClues(street="Strada Sibiu", street_number="12")
    line = {"type": "LineString", "coordinates": [[26.15, 44.42], [26.16, 44.42]]}
    geocoder = FakeGeocoder({"Strada Sibiu, București": [Place(44.42, 26.155, line)]})
    raw = RAW.model_copy(update={"map_lat": None})

    evidence = gather_evidence(raw, clues, geocoder=geocoder, stations=[])

    assert _kinds(evidence) == ["street"]
    assert evidence[0].candidates[0].geom_type == "LineString"


def test_stated_walking_time_tightens_the_metro_clue():
    raw = RAW.model_copy(update={"map_lat": None})

    def metro_reach(clues: LocationClues) -> float:
        [evidence] = gather_evidence(raw, clues, geocoder=FakeGeocoder({}), stations=STATIONS)
        return evidence.reach_m

    vague = metro_reach(LocationClues(metro_station="piata unirii"))
    stated = metro_reach(LocationClues(metro_station="Piata Unirii", metro_walk_minutes=5))

    assert stated == 400
    assert stated < vague


def test_neighbourhood_outline_is_trusted_more_than_its_centre_point():
    raw = RAW.model_copy(update={"map_lat": None})

    def reach(place: Place) -> float:
        geocoder = FakeGeocoder({"Titan, București": [place]})
        clues = LocationClues(neighbourhood="Titan")
        return gather_evidence(raw, clues, geocoder=geocoder, stations=[])[0].reach_m

    assert reach(Place(44.41, 26.15, SECTOR_3)) < reach(_point(26.15, 44.41))


def test_locate_listing_combines_the_clues():
    clues = LocationClues(street="Strada Sibiu", street_number="12", metro_station="Dristor")
    geocoder = FakeGeocoder(
        {
            "Sector 3, București": [Place(44.41, 26.15, SECTOR_3)],
            # The second candidate is a same-named building across the city.
            "Strada Sibiu 12, București": [_point(26.05, 44.50), _point(26.145, 44.418)],
        }
    )

    location = locate_listing(RAW, clues, geocoder=geocoder, stations=STATIONS)

    assert location is not None
    assert math.isclose(location.latitude, 44.418, abs_tol=0.002)
    assert math.isclose(location.longitude, 26.145, abs_tol=0.002)
    assert location.uncertainty_m < 300
    assert set(location.agreeing) == {"area", "olx_point", "building", "metro"}


def test_locate_listing_returns_none_without_any_evidence():
    raw = RAW.model_copy(update={"map_lat": None, "address_raw": ""})

    assert locate_listing(raw, LocationClues(), geocoder=FakeGeocoder({}), stations=[]) is None

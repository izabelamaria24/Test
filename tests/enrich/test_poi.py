import functools

import requests

from realestate.enrich import poi
from realestate.enrich.geocoder import USER_AGENT
from realestate.enrich.poi import (
    fetch_bucharest_subway_stations,
    haversine_km,
    load_subway_stations,
    nearest_subway_station,
    parse_overpass_stations,
)

DRISTOR = {"name": "Dristor 1", "lat": 44.42, "lon": 26.14}


def test_station_list_is_fetched_once_then_read_from_cache(tmp_path):
    cache_path = tmp_path / "stations.json"
    calls = []

    def fetch():
        calls.append(1)
        return [DRISTOR]

    assert load_subway_stations(cache_path, fetch_fn=fetch) == [DRISTOR]
    assert load_subway_stations(cache_path, fetch_fn=fetch) == [DRISTOR]
    assert len(calls) == 1


def test_empty_station_list_is_not_cached(tmp_path):
    cache_path = tmp_path / "stations.json"

    assert load_subway_stations(cache_path, fetch_fn=list) == []
    assert not cache_path.exists()


def test_fetch_stations_identifies_itself_to_overpass(monkeypatch):
    # Overpass answers 406 to the default python-requests User-Agent.
    sent_headers: list[dict] = []

    class _Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"elements": [{"lat": 44.42, "lon": 26.14, "tags": {"name": "Dristor 1"}}]}

    def fake_post(url, data, headers, timeout):
        sent_headers.append(headers)
        return _Response()

    monkeypatch.setattr(poi.requests, "post", fake_post)

    stations = fetch_bucharest_subway_stations()

    assert sent_headers == [{"User-Agent": USER_AGENT}]
    assert stations == [{"name": "Dristor 1", "lat": 44.42, "lon": 26.14}]


STATIONS = [
    {"name": "Piata Unirii", "lat": 44.4278, "lon": 26.1030},
    {"name": "Universitate", "lat": 44.4356, "lon": 26.1023},
    {"name": "Dristor", "lat": 44.4197, "lon": 26.1489},
]


def test_haversine_km_zero_for_same_point():
    assert haversine_km(44.43, 26.10, 44.43, 26.10) == 0.0


def test_haversine_km_matches_known_approx_distance():
    # Bucharest Piata Unirii to Universitate, roughly ~0.9km apart.
    distance = haversine_km(44.4278, 26.1030, 44.4356, 26.1023)
    assert 0.5 < distance < 1.5


def test_nearest_subway_station_picks_shortest_walking_time():
    def fake_walking_fn(lat1, lon1, lat2, lon2):
        # Simulate: Universitate is closest by straight-line but has a slow walking route;
        # Piata Unirii is farther by straight-line but has the fastest actual walk.
        if (lat2, lon2) == (44.4356, 26.1023):
            return 25.0
        if (lat2, lon2) == (44.4278, 26.1030):
            return 8.0
        return 40.0

    result = nearest_subway_station(
        44.430, 26.102, STATIONS, candidates=3, walking_fn=fake_walking_fn
    )

    assert result == ("Piata Unirii", 8.0)


def test_nearest_subway_station_returns_none_for_empty_station_list():
    assert nearest_subway_station(44.43, 26.10, [], walking_fn=lambda *a, **k: 0.0) is None


def test_nearest_subway_station_skips_candidate_when_walking_fn_raises():
    # Universitate is closest by straight-line but OSRM fails to route to it (e.g. transient
    # network error); the orchestrator should fall back to the next-best candidate instead of
    # crashing the whole lookup.
    def flaky_walking_fn(lat1, lon1, lat2, lon2):
        if (lat2, lon2) == (44.4356, 26.1023):
            raise requests.RequestException("OSRM unreachable")
        if (lat2, lon2) == (44.4278, 26.1030):
            return 8.0
        return 40.0

    result = nearest_subway_station(
        44.430, 26.102, STATIONS, candidates=3, walking_fn=flaky_walking_fn
    )

    assert result == ("Piata Unirii", 8.0)


def test_nearest_subway_station_skips_candidate_when_walking_fn_raises_value_error():
    # Universitate is closest by straight-line but OSRM genuinely can't find a walking
    # route to it (e.g. "NoRoute"), which osrm_walking_minutes surfaces as a ValueError;
    # the orchestrator should fall back to the next-best candidate instead of crashing.
    def no_route_walking_fn(lat1, lon1, lat2, lon2):
        if (lat2, lon2) == (44.4356, 26.1023):
            raise ValueError("OSRM route failed: NoRoute")
        if (lat2, lon2) == (44.4278, 26.1030):
            return 8.0
        return 40.0

    result = nearest_subway_station(
        44.430, 26.102, STATIONS, candidates=3, walking_fn=no_route_walking_fn
    )

    assert result == ("Piata Unirii", 8.0)


def test_nearest_subway_station_returns_none_when_all_candidates_fail():
    def always_fails(lat1, lon1, lat2, lon2):
        raise requests.RequestException("OSRM unreachable")

    result = nearest_subway_station(44.430, 26.102, STATIONS, candidates=3, walking_fn=always_fails)

    assert result is None


def test_nearest_subway_station_accepts_partially_bound_walking_fn():
    # Custom OSRM URLs are now supplied by binding the arg (functools.partial),
    # not via a nearest_subway_station parameter.
    seen_urls: list[str] = []

    def walking_fn_with_url(lat1, lon1, lat2, lon2, *, osrm_url):
        seen_urls.append(osrm_url)
        return 5.0 if (lat2, lon2) == (44.4356, 26.1023) else 30.0

    bound = functools.partial(walking_fn_with_url, osrm_url="http://osrm.example:5000")
    result = nearest_subway_station(44.435, 26.102, STATIONS, candidates=3, walking_fn=bound)

    assert result == ("Universitate", 5.0)
    assert seen_urls and all(u == "http://osrm.example:5000" for u in seen_urls)


def test_parse_overpass_stations_builds_stations_from_valid_records():
    elements = [
        {"lat": 44.4278, "lon": 26.1030, "tags": {"name": "Piata Unirii"}},
        {"lat": 44, "lon": 26},  # ints are valid; missing name falls back
    ]

    assert parse_overpass_stations(elements) == [
        {"name": "Piata Unirii", "lat": 44.4278, "lon": 26.1030},
        {"name": "unknown", "lat": 44, "lon": 26},
    ]


def test_parse_overpass_stations_skips_records_with_missing_or_invalid_coordinates():
    valid = {"lat": 44.4278, "lon": 26.1030, "tags": {"name": "Piata Unirii"}}
    elements = [
        {"lon": 26.1, "tags": {"name": "no lat"}},
        {"lat": 44.4, "tags": {"name": "no lon"}},
        {"lat": "44.4", "lon": 26.1, "tags": {"name": "string lat"}},
        {"lat": None, "lon": 26.1, "tags": {"name": "null lat"}},
        {"lat": True, "lon": 26.1, "tags": {"name": "bool lat"}},
        {"lat": float("nan"), "lon": 26.1, "tags": {"name": "nan lat"}},
        valid,
    ]

    stations = parse_overpass_stations(elements)

    assert [s["name"] for s in stations] == ["Piata Unirii"]

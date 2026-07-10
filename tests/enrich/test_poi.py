import math

import requests

from realestate.enrich.poi import haversine_km, nearest_subway_station

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
    def fake_walking_fn(lat1, lon1, lat2, lon2, osrm_url="http://localhost:5000"):
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
    def flaky_walking_fn(lat1, lon1, lat2, lon2, osrm_url="http://localhost:5000"):
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
    def no_route_walking_fn(lat1, lon1, lat2, lon2, osrm_url="http://localhost:5000"):
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
    def always_fails(lat1, lon1, lat2, lon2, osrm_url="http://localhost:5000"):
        raise requests.RequestException("OSRM unreachable")

    result = nearest_subway_station(
        44.430, 26.102, STATIONS, candidates=3, walking_fn=always_fails
    )

    assert result is None

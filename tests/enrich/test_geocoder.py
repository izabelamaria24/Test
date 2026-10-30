import json
from pathlib import Path

import pytest
import requests

from realestate.enrich import geocoder as geocoder_module
from realestate.enrich.geocoder import CachedGeocoder, Place, search_places

POINT = {"type": "Point", "coordinates": [26.1, 44.43]}
LINE = {"type": "LineString", "coordinates": [[26.1, 44.43], [26.11, 44.43]]}


class _FakeResponse:
    def __init__(self, body: list[dict]):
        self._body = body

    def raise_for_status(self) -> None:
        pass

    def json(self) -> list[dict]:
        return self._body


@pytest.fixture
def nominatim(monkeypatch):
    calls: list[dict] = []
    responses: list[list[dict]] = []

    def fake_get(url, params, headers, timeout):
        calls.append(params)
        return _FakeResponse(responses.pop(0))

    monkeypatch.setattr(geocoder_module.requests, "get", fake_get)
    monkeypatch.setattr(geocoder_module.time, "sleep", lambda seconds: None)
    return calls, responses


def test_search_is_bounded_to_bucharest_and_asks_for_shapes(nominatim):
    calls, responses = nominatim
    responses.append([])

    search_places("Strada Sibiu 12, București")

    params = calls[0]
    assert params["q"] == "Strada Sibiu 12, București, Romania"
    assert params["bounded"] == 1
    assert params["polygon_geojson"] == 1
    assert params["limit"] == 3
    assert params["viewbox"] == "25.96,44.54,26.23,44.33"


def test_search_returns_every_candidate_with_its_shape(nominatim):
    _, responses = nominatim
    responses.append(
        [
            {"lat": "44.43", "lon": "26.105", "geojson": LINE},
            {"lat": "44.40", "lon": "26.0"},
        ]
    )

    places = search_places("Strada Sibiu, București")

    assert places == [
        Place(latitude=44.43, longitude=26.105, geometry=LINE),
        Place(
            latitude=44.40, longitude=26.0, geometry={"type": "Point", "coordinates": [26.0, 44.4]}
        ),
    ]


def test_place_shape_is_a_shapely_geometry():
    assert Place(latitude=44.43, longitude=26.105, geometry=LINE).shape().length > 0


def test_cache_miss_searches_and_persists(tmp_path: Path):
    cache_path = tmp_path / "places.json"
    calls: list[str] = []
    place = Place(latitude=44.43, longitude=26.1, geometry=POINT)

    def fake_search(query: str) -> list[Place]:
        calls.append(query)
        return [place]

    assert CachedGeocoder(cache_path, search_fn=fake_search).search("Titan") == [place]
    assert CachedGeocoder(cache_path, search_fn=fake_search).search("Titan") == [place]
    assert calls == ["Titan"]
    assert json.loads(cache_path.read_text())["Titan"] == [
        {"latitude": 44.43, "longitude": 26.1, "geometry": POINT}
    ]


def test_caches_empty_results(tmp_path: Path):
    calls: list[str] = []

    def fake_search(query: str) -> list[Place]:
        calls.append(query)
        return []

    geocoder = CachedGeocoder(tmp_path / "places.json", search_fn=fake_search)

    assert geocoder.search("Nowhere") == []
    assert geocoder.search("Nowhere") == []
    assert calls == ["Nowhere"]


def test_corrupted_cache_file_falls_back_to_empty_cache(tmp_path: Path):
    cache_path = tmp_path / "places.json"
    cache_path.write_text("{not valid json")

    geocoder = CachedGeocoder(cache_path, search_fn=lambda query: [])

    assert geocoder.search("x") == []


def test_network_failure_returns_nothing_and_is_not_cached(tmp_path: Path):
    cache_path = tmp_path / "places.json"

    def fake_search(query: str) -> list[Place]:
        raise requests.RequestException("boom")

    assert CachedGeocoder(cache_path, search_fn=fake_search).search("x") == []
    assert not cache_path.exists()

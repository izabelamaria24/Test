import json
from pathlib import Path

import pytest
import requests

from realestate.enrich import geocoder as geocoder_module
from realestate.enrich.geocoder import CachedGeocoder, GeocodeResult, geocode_address


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


def test_geocode_address_is_bounded_to_bucharest_and_requests_address_details(nominatim):
    calls, responses = nominatim
    responses.append([])

    geocode_address("Strada Sibiu 12, București")

    params = calls[0]
    assert params["q"] == "Strada Sibiu 12, București, Romania"
    assert params["bounded"] == 1
    assert params["addressdetails"] == 1
    assert params["viewbox"] == "25.96,44.54,26.23,44.33"


@pytest.mark.parametrize("key", ["district", "city_district", "suburb", "borough"])
def test_geocode_address_reads_sector_from_address_details(nominatim, key):
    _, responses = nominatim
    responses.append([{"lat": "44.42", "lon": "26.1", "address": {key: "Sector 3"}}])

    assert geocode_address("x") == GeocodeResult(latitude=44.42, longitude=26.1, sector=3)


def test_geocode_address_leaves_sector_unknown_outside_the_sectors(nominatim):
    _, responses = nominatim
    responses.append([{"lat": "44.4", "lon": "26.0", "address": {"village": "Chiajna"}}])

    assert geocode_address("x") == GeocodeResult(latitude=44.4, longitude=26.0, sector=None)


def test_geocode_address_returns_none_without_results(nominatim):
    _, responses = nominatim
    responses.append([])

    assert geocode_address("x") is None


def test_cache_miss_calls_geocode_fn_and_persists(tmp_path: Path):
    cache_path = tmp_path / "geocode_cache.json"
    calls: list[str] = []

    def fake_geocode(address: str) -> GeocodeResult | None:
        calls.append(address)
        return GeocodeResult(latitude=44.43, longitude=26.10, sector=3)

    geocoder = CachedGeocoder(cache_path=cache_path, geocode_fn=fake_geocode)
    result = geocoder.geocode("Centrul Istoric, Bucuresti")

    assert result is not None
    assert result.latitude == 44.43
    assert calls == ["Centrul Istoric, Bucuresti"]
    assert json.loads(cache_path.read_text())["Centrul Istoric, Bucuresti"] == {
        "latitude": 44.43,
        "longitude": 26.10,
        "sector": 3,
    }
    assert CachedGeocoder(cache_path=cache_path, geocode_fn=fake_geocode).geocode(
        "Centrul Istoric, Bucuresti"
    ) == GeocodeResult(latitude=44.43, longitude=26.10, sector=3)
    assert len(calls) == 1


def test_cache_hit_does_not_call_geocode_fn_again(tmp_path: Path):
    cache_path = tmp_path / "geocode_cache.json"
    calls: list[str] = []

    def fake_geocode(address: str) -> GeocodeResult | None:
        calls.append(address)
        return GeocodeResult(latitude=44.43, longitude=26.10)

    geocoder = CachedGeocoder(cache_path=cache_path, geocode_fn=fake_geocode)
    geocoder.geocode("Centrul Istoric, Bucuresti")
    geocoder.geocode("Centrul Istoric, Bucuresti")

    assert calls == ["Centrul Istoric, Bucuresti"]


def test_caches_negative_results(tmp_path: Path):
    cache_path = tmp_path / "geocode_cache.json"
    calls: list[str] = []

    def fake_geocode(address: str) -> GeocodeResult | None:
        calls.append(address)
        return None

    geocoder = CachedGeocoder(cache_path=cache_path, geocode_fn=fake_geocode)
    first = geocoder.geocode("Nonexistent Place")
    second = geocoder.geocode("Nonexistent Place")

    assert first is None
    assert second is None
    assert calls == ["Nonexistent Place"]


def test_loads_existing_cache_from_disk_without_calling_geocode_fn(tmp_path: Path):
    cache_path = tmp_path / "geocode_cache.json"
    cache_path.write_text(json.dumps({"some address": {"latitude": 44.43, "longitude": 26.10}}))

    def fake_geocode(address: str) -> GeocodeResult | None:
        raise AssertionError("geocode_fn should not be called for a cached address")

    geocoder = CachedGeocoder(cache_path=cache_path, geocode_fn=fake_geocode)
    result = geocoder.geocode("some address")

    assert result == GeocodeResult(latitude=44.43, longitude=26.10)


def test_corrupted_cache_file_falls_back_to_empty_cache(tmp_path: Path):
    cache_path = tmp_path / "geocode_cache.json"
    cache_path.write_text("{not valid json")
    calls: list[str] = []

    def fake_geocode(address: str) -> GeocodeResult | None:
        calls.append(address)
        return GeocodeResult(latitude=44.43, longitude=26.10)

    geocoder = CachedGeocoder(cache_path=cache_path, geocode_fn=fake_geocode)
    result = geocoder.geocode("some address")

    assert result == GeocodeResult(latitude=44.43, longitude=26.10)
    assert calls == ["some address"]


def test_network_failure_returns_none_and_does_not_cache(tmp_path: Path):
    cache_path = tmp_path / "geocode_cache.json"

    def fake_geocode(address: str) -> GeocodeResult | None:
        raise requests.RequestException("boom")

    geocoder = CachedGeocoder(cache_path=cache_path, geocode_fn=fake_geocode)
    result = geocoder.geocode("some address")

    assert result is None
    assert not cache_path.exists()

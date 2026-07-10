import json
from pathlib import Path

import requests

from realestate.enrich.geocoder import CachedGeocoder, GeocodeResult


def test_cache_miss_calls_geocode_fn_and_persists(tmp_path: Path):
    cache_path = tmp_path / "geocode_cache.json"
    calls: list[str] = []

    def fake_geocode(address: str) -> GeocodeResult | None:
        calls.append(address)
        return GeocodeResult(latitude=44.43, longitude=26.10)

    geocoder = CachedGeocoder(cache_path=cache_path, geocode_fn=fake_geocode)
    result = geocoder.geocode("Centrul Istoric, Bucuresti")

    assert result is not None
    assert result.latitude == 44.43
    assert calls == ["Centrul Istoric, Bucuresti"]
    assert json.loads(cache_path.read_text())["Centrul Istoric, Bucuresti"] == {
        "latitude": 44.43,
        "longitude": 26.10,
    }


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

import json
from pathlib import Path
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

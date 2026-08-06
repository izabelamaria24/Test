import copy
from pathlib import Path

from realestate.enrich.geocoder import CachedGeocoder, GeocodeResult
from realestate.ingestion_pipeline import run_ingestion
from tests.ingest.test_olx_parser import SAMPLE_AD, SAMPLE_HTML, _build_sample_html


class FakeEmbedder:
    def embed_passage(self, text: str) -> list[float]:
        return [0.1] * 768

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 768


class FakeStore:
    def __init__(self):
        self.upserted: list[tuple[str, list[float], dict]] = []

    def upsert(self, external_id, vector, payload):
        self.upserted.append((external_id, vector, payload))

    def query(self, *args, **kwargs):
        raise NotImplementedError("not exercised in this test")


def test_run_ingestion_embeds_and_stores_valid_listings(tmp_path: Path):
    (tmp_path / "304473136.html").write_text(SAMPLE_HTML, encoding="utf-8")
    (tmp_path / "bad.html").write_text("<html></html>", encoding="utf-8")

    geocoder = CachedGeocoder(
        cache_path=tmp_path / "cache.json",
        geocode_fn=lambda address: GeocodeResult(latitude=44.43, longitude=26.10),
    )
    store = FakeStore()

    report = run_ingestion(
        tmp_path,
        geocoder=geocoder,
        stations=[],
        embedder=FakeEmbedder(),
        store=store,
    )

    assert report.succeeded == 1
    assert len(report.parse_failures) == 1
    assert store.upserted[0][0] == "304473136"
    assert store.upserted[0][2]["price_eur"] == 100000.0


def test_run_ingestion_quarantines_enrich_failures_without_aborting_batch(tmp_path: Path):
    # A listing that parses fine (well-formed PRERENDERED_STATE) but whose price is not in
    # the expected "<digits> €" format - mirrors real OLX listings priced "La cerere" (on
    # request) that fail normalize_price_eur's regex during enrichment, not during parsing.
    bad_price_ad = copy.deepcopy(SAMPLE_AD)
    bad_price_ad["id"] = 999999999
    bad_price_ad["price"]["regularPrice"] = {
        "value": 100000,
        "currencyCode": "EUR",
        "currencySymbol": "La cerere",
    }

    (tmp_path / "304473136.html").write_text(SAMPLE_HTML, encoding="utf-8")
    (tmp_path / "999999999.html").write_text(_build_sample_html(bad_price_ad), encoding="utf-8")

    geocoder = CachedGeocoder(
        cache_path=tmp_path / "cache.json",
        geocode_fn=lambda address: GeocodeResult(latitude=44.43, longitude=26.10),
    )
    store = FakeStore()

    report = run_ingestion(
        tmp_path,
        geocoder=geocoder,
        stations=[],
        embedder=FakeEmbedder(),
        store=store,
    )

    assert report.succeeded == 1
    assert len(report.enrich_failures) == 1
    assert report.enrich_failures[0][0] == "999999999"
    assert store.upserted[0][0] == "304473136"

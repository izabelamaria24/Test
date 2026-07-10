from pathlib import Path

from realestate.enrich.geocoder import CachedGeocoder, GeocodeResult
from realestate.ingestion_pipeline import run_ingestion
from tests.ingest.test_olx_parser import SAMPLE_HTML


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

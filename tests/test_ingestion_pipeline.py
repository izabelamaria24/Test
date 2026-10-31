import copy
from pathlib import Path

from realestate.enrich.geocoder import CachedGeocoder, Place
from realestate.enrich.location_clues import LocationClues
from realestate.ingestion_pipeline import ingest_listings, run_ingestion
from realestate.models import RawListing
from tests.ingest.test_olx_parser import SAMPLE_AD, SAMPLE_HTML, _build_sample_html

STREET = Place(44.43, 26.10, {"type": "Point", "coordinates": [26.10, 44.43]})


def _geocoder(tmp_path: Path, results: dict[str, list[Place]] | None = None) -> CachedGeocoder:
    found = results or {}
    return CachedGeocoder(tmp_path / "places.json", search_fn=lambda q: found.get(q, []))


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

    geocoder = _geocoder(tmp_path)
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

    geocoder = _geocoder(tmp_path)
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


def test_run_ingestion_locates_listings_from_their_text(tmp_path: Path):
    (tmp_path / "304473136.html").write_text(SAMPLE_HTML, encoding="utf-8")

    class StreetExtractor:
        def extract(self, title: str, description: str) -> LocationClues:
            return LocationClues(street="Strada Sibiu")

    geocoder = _geocoder(tmp_path, {"Strada Sibiu, București": [STREET]})
    store = FakeStore()

    run_ingestion(
        tmp_path,
        geocoder=geocoder,
        stations=[],
        embedder=FakeEmbedder(),
        store=store,
        clue_extractor=StreetExtractor(),
    )

    payload = store.upserted[0][2]
    assert payload["location_evidence"] == ["olx_point", "street"]
    assert payload["location_uncertainty_m"] < 300


def test_ingest_listings_stores_listings_from_any_source(tmp_path: Path):
    raw = RawListing(
        external_id="imobiliare-1",
        title="Apartament cu 2 camere",
        description="d",
        price_raw="650 EUR",
        address_raw="Strada Sibiu 12, Drumul Taberei, Sector 6",
        specs={},
        image_urls=[],
        source_file="rent.json",
        map_lat=None,
        map_lon=None,
        listing_type="rent",
    )
    store = FakeStore()

    report = ingest_listings(
        [raw], geocoder=_geocoder(tmp_path), stations=[], embedder=FakeEmbedder(), store=store
    )

    assert report.succeeded == 1
    external_id, _, payload = store.upserted[0]
    assert external_id == "imobiliare-1"
    assert payload["listing_type"] == "rent"
    assert payload["price_eur"] == 650.0


def test_run_ingestion_ignores_a_map_point_shared_by_many_listings(tmp_path: Path):
    for listing_id in ("1", "2", "3"):
        ad = copy.deepcopy(SAMPLE_AD)
        ad["id"] = int(listing_id)
        (tmp_path / f"{listing_id}.html").write_text(_build_sample_html(ad), encoding="utf-8")
    store = FakeStore()

    run_ingestion(
        tmp_path,
        geocoder=_geocoder(tmp_path),
        stations=[],
        embedder=FakeEmbedder(),
        store=store,
    )

    assert [payload["location_evidence"] for _, _, payload in store.upserted] == [[], [], []]


def test_run_ingestion_keeps_a_map_point_used_by_fewer_listings(tmp_path: Path):
    (tmp_path / "304473136.html").write_text(SAMPLE_HTML, encoding="utf-8")
    store = FakeStore()

    run_ingestion(
        tmp_path,
        geocoder=_geocoder(tmp_path),
        stations=[],
        embedder=FakeEmbedder(),
        store=store,
    )

    assert store.upserted[0][2]["location_evidence"] == ["olx_point"]

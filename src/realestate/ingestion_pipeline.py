from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from realestate.embed.base import Embedder
from realestate.enrich.geocoder import CachedGeocoder
from realestate.enrich.location_clues import LocationClueExtractor
from realestate.enrich.pipeline import enrich_listing
from realestate.enrich.poi import Station
from realestate.ingest.olx_loader import load_olx_html_directory
from realestate.models import RawListing
from realestate.store.base import VectorStore


@dataclass
class IngestionReport:
    succeeded: int = 0
    parse_failures: list[tuple[Path, str]] = field(default_factory=list)
    enrich_failures: list[tuple[str, str]] = field(default_factory=list)


# Distinct apartments never share an exact point; OLX shows a per-sector default for listings
# without a location, so a point repeated this often carries no information.
_DEFAULT_POINT_MIN_LISTINGS = 3


def _shared_map_points(raw_listings: list[RawListing]) -> set[tuple[float, float]]:
    counts = Counter(
        (raw.map_lat, raw.map_lon)
        for raw in raw_listings
        if raw.map_lat is not None and raw.map_lon is not None
    )
    return {point for point, count in counts.items() if count >= _DEFAULT_POINT_MIN_LISTINGS}


def run_ingestion(
    html_dir: Path,
    *,
    geocoder: CachedGeocoder,
    stations: list[Station],
    embedder: Embedder,
    store: VectorStore,
    clue_extractor: LocationClueExtractor | None = None,
) -> IngestionReport:
    raw_listings, parse_failures = load_olx_html_directory(html_dir)
    report = ingest_listings(
        raw_listings,
        geocoder=geocoder,
        stations=stations,
        embedder=embedder,
        store=store,
        clue_extractor=clue_extractor,
    )
    report.parse_failures = parse_failures
    return report


def ingest_listings(
    raw_listings: list[RawListing],
    *,
    geocoder: CachedGeocoder,
    stations: list[Station],
    embedder: Embedder,
    store: VectorStore,
    clue_extractor: LocationClueExtractor | None = None,
) -> IngestionReport:
    default_points = _shared_map_points(raw_listings)

    succeeded = 0
    enrich_failures: list[tuple[str, str]] = []
    for raw in raw_listings:
        try:
            enriched = enrich_listing(
                raw,
                geocoder=geocoder,
                stations=stations,
                clue_extractor=clue_extractor,
                olx_point_is_default=(raw.map_lat, raw.map_lon) in default_points,
            )
            vector = embedder.embed_passage(enriched.description)
            store.upsert(
                enriched.external_id,
                vector,
                payload=enriched.model_dump(),
            )
            succeeded += 1
        except Exception as exc:  # noqa: BLE001 - quarantine, don't crash the batch
            enrich_failures.append((raw.external_id, str(exc)))

    return IngestionReport(succeeded=succeeded, enrich_failures=enrich_failures)

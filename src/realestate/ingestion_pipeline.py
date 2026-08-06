from dataclasses import dataclass, field
from pathlib import Path

from realestate.embed.base import Embedder
from realestate.enrich.geocoder import CachedGeocoder
from realestate.enrich.pipeline import enrich_listing
from realestate.ingest.olx_loader import load_olx_html_directory
from realestate.store.base import VectorStore


@dataclass
class IngestionReport:
    succeeded: int = 0
    parse_failures: list[tuple[Path, str]] = field(default_factory=list)
    enrich_failures: list[tuple[str, str]] = field(default_factory=list)


def run_ingestion(
    html_dir: Path,
    *,
    geocoder: CachedGeocoder,
    stations: list[dict],
    embedder: Embedder,
    store: VectorStore,
) -> IngestionReport:
    raw_listings, parse_failures = load_olx_html_directory(html_dir)

    succeeded = 0
    enrich_failures: list[tuple[str, str]] = []
    for raw in raw_listings:
        try:
            enriched = enrich_listing(raw, geocoder=geocoder, stations=stations)
            vector = embedder.embed_passage(enriched.description)
            store.upsert(
                enriched.external_id,
                vector,
                payload=enriched.model_dump(),
            )
            succeeded += 1
        except Exception as exc:  # noqa: BLE001 - quarantine, don't crash the batch
            enrich_failures.append((raw.external_id, str(exc)))

    return IngestionReport(
        succeeded=succeeded, parse_failures=parse_failures, enrich_failures=enrich_failures
    )

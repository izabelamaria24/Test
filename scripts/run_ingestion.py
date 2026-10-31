"""Run the full ingestion pipeline for one source.

Usage:
  python scripts/run_ingestion.py olx 100      # scrape OLX if fewer than 100 pages on disk
  python scripts/run_ingestion.py imobiliare   # records saved by scripts/fetch_imobiliare.py
"""

import sys
from pathlib import Path

import requests

from realestate.embed.sentence_embedder import MultilingualE5Embedder
from realestate.enrich.geocoder import CachedGeocoder
from realestate.enrich.location_clues import OllamaLocationClueExtractor
from realestate.enrich.poi import load_subway_stations
from realestate.ingest.imobiliare_loader import load_imobiliare_directory
from realestate.ingest.olx_loader import load_olx_html_directory
from realestate.ingest.olx_scraper import download_listings
from realestate.ingestion_pipeline import ingest_listings
from realestate.store.qdrant_store import QdrantListingStore

if __name__ == "__main__":
    source = sys.argv[1]
    if source == "olx":
        target_count = int(sys.argv[2])
        html_dir = Path("data/raw/olx_html")
        newly_downloaded = download_listings(target_count, html_dir)
        print(f"Downloaded {len(newly_downloaded)} new listings (target: {target_count} on disk).")
        raw_listings, parse_failures = load_olx_html_directory(html_dir)
    elif source == "imobiliare":
        raw_listings, parse_failures = load_imobiliare_directory(Path("data/raw/imobiliare"))
    else:
        sys.exit(f"unknown source {source!r}; use 'olx' or 'imobiliare'")

    geocoder = CachedGeocoder(cache_path=Path("data/cache/places_cache.json"))
    try:
        stations = load_subway_stations(Path("data/cache/subway_stations.json"))
    except requests.RequestException as exc:
        print(f"Subway stations unavailable ({exc}); metro clues and walking times are skipped.")
        stations = []
    embedder = MultilingualE5Embedder(device="mps")
    store = QdrantListingStore()

    report = ingest_listings(
        raw_listings,
        geocoder=geocoder,
        stations=stations,
        embedder=embedder,
        store=store,
        clue_extractor=OllamaLocationClueExtractor(),
    )

    print(f"Ingested {report.succeeded} listings.")
    if parse_failures:
        print(f"{len(parse_failures)} records failed to parse:")
        for path, error in parse_failures:
            print(f"  {path}: {error}")
    if report.enrich_failures:
        print(f"{len(report.enrich_failures)} listings failed to enrich/embed/store:")
        for external_id, error in report.enrich_failures:
            print(f"  {external_id}: {error}")

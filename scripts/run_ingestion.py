"""Scrape OLX.ro listings (if needed) and run the full ingestion pipeline.

Usage: python scripts/run_ingestion.py 100
"""

import sys
from pathlib import Path

from realestate.embed.sentence_embedder import MultilingualE5Embedder
from realestate.enrich.geocoder import CachedGeocoder
from realestate.enrich.poi import fetch_bucharest_subway_stations
from realestate.ingest.olx_scraper import download_listings
from realestate.ingestion_pipeline import run_ingestion
from realestate.store.qdrant_store import QdrantListingStore

if __name__ == "__main__":
    target_count = int(sys.argv[1])
    html_dir = Path("data/raw/olx_html")

    newly_downloaded = download_listings(target_count, html_dir)
    print(f"Downloaded {len(newly_downloaded)} new listings (target: {target_count} total on disk).")

    geocoder = CachedGeocoder(cache_path=Path("data/cache/geocode_cache.json"))
    stations = fetch_bucharest_subway_stations()
    embedder = MultilingualE5Embedder(device="mps")
    store = QdrantListingStore()

    report = run_ingestion(
        html_dir, geocoder=geocoder, stations=stations, embedder=embedder, store=store
    )

    print(f"Ingested {report.succeeded} listings.")
    if report.parse_failures:
        print(f"{len(report.parse_failures)} files failed to parse:")
        for path, error in report.parse_failures:
            print(f"  {path}: {error}")
    if report.enrich_failures:
        print(f"{len(report.enrich_failures)} listings failed to enrich/embed/store:")
        for external_id, error in report.enrich_failures:
            print(f"  {external_id}: {error}")

# src/realestate/ingest/olx_loader.py
from pathlib import Path
from realestate.ingest.olx_parser import parse_olx_listing_html
from realestate.models import RawListing


def load_olx_html_directory(
    directory: Path,
) -> tuple[list[RawListing], list[tuple[Path, str]]]:
    listings: list[RawListing] = []
    failures: list[tuple[Path, str]] = []

    for html_file in sorted(directory.glob("*.html")):
        external_id = html_file.stem
        try:
            html = html_file.read_text(encoding="utf-8")
            listings.append(parse_olx_listing_html(html, external_id))
        except Exception as exc:  # noqa: BLE001 - quarantine, don't crash the batch
            failures.append((html_file, str(exc)))

    return listings, failures

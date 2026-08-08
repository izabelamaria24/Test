# src/realestate/ingest/olx_scraper.py
import re
import time
from pathlib import Path

import requests

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
CATEGORY_URL = "https://www.olx.ro/imobiliare/apartamente-garsoniere-de-vanzare/bucuresti/"

_LISTING_HREF_RE = re.compile(r'href="(/d/oferta/[a-zA-Z0-9_-]+\.html)')
_LISTING_ID_RE = re.compile(r"-ID([a-zA-Z0-9]+)\.html$")


def fetch_listing_urls_from_category(
    page: int, *, session: requests.Session, rate_limit_seconds: float = 2.0
) -> list[str]:
    response = session.get(
        CATEGORY_URL, params={"page": page}, headers={"User-Agent": USER_AGENT}, timeout=15
    )
    time.sleep(rate_limit_seconds)
    if response.status_code != 200:
        raise RuntimeError(f"category page fetch failed: status={response.status_code} page={page}")
    hrefs = sorted(set(_LISTING_HREF_RE.findall(response.text)))
    return [f"https://www.olx.ro{href}" for href in hrefs]


def listing_id_from_url(url: str) -> str:
    match = _LISTING_ID_RE.search(url)
    if not match:
        raise ValueError(f"could not extract listing id from URL: {url}")
    return match.group(1)


def download_listings(
    target_count: int,
    output_dir: Path,
    *,
    session: requests.Session | None = None,
    rate_limit_seconds: float = 2.0,
    max_pages: int = 50,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    session = session or requests.Session()

    downloaded_ids: list[str] = []
    for page in range(1, max_pages + 1):
        if len(list(output_dir.glob("*.html"))) >= target_count:
            break

        listing_urls = fetch_listing_urls_from_category(
            page, session=session, rate_limit_seconds=rate_limit_seconds
        )
        if not listing_urls:
            break

        for url in listing_urls:
            if len(list(output_dir.glob("*.html"))) >= target_count:
                break
            try:
                listing_id = listing_id_from_url(url)
            except ValueError:
                continue
            file_path = output_dir / f"{listing_id}.html"
            if file_path.exists():
                continue

            response = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
            time.sleep(rate_limit_seconds)
            if response.status_code != 200:
                continue
            file_path.write_text(response.text, encoding="utf-8")
            downloaded_ids.append(listing_id)

    return downloaded_ids

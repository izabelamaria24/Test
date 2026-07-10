# tests/ingest/test_olx_scraper.py
from pathlib import Path
from realestate.ingest.olx_scraper import (
    download_listings,
    fetch_listing_urls_from_category,
    listing_id_from_url,
)

CATEGORY_PAGE_HTML = """
<html><body>
<a href="/d/oferta/apartament-2-camere-titan-IDkBxn2.html?search_reason=organic">Ad 1</a>
<a href="/d/oferta/apartament-2-camere-titan-IDkBxn2.html?search_reason=organic">Ad 1 dup</a>
<a href="/d/oferta/apartament-3-camere-ghencea-IDkHrfc.html">Ad 2</a>
</body></html>
"""

LISTING_HTML_TEMPLATE = "<html><body>listing {listing_id}</body></html>"


class FakeResponse:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError(f"status {self.status_code}")


class FakeSession:
    def __init__(self, category_pages: dict[int, str], listing_pages: dict[str, str]):
        self._category_pages = category_pages
        self._listing_pages = listing_pages

    def get(self, url, params=None, headers=None, timeout=None):
        if params and "page" in params:
            return FakeResponse(self._category_pages.get(params["page"], ""))
        return FakeResponse(self._listing_pages.get(url, ""), status_code=200 if url in self._listing_pages else 404)


def test_listing_id_from_url_extracts_trailing_id():
    assert listing_id_from_url("https://www.olx.ro/d/oferta/apartament-IDkBxn2.html") == "kBxn2"


def test_fetch_listing_urls_from_category_dedupes_and_builds_full_urls():
    session = FakeSession(category_pages={1: CATEGORY_PAGE_HTML}, listing_pages={})
    urls = fetch_listing_urls_from_category(1, session=session)
    assert urls == [
        "https://www.olx.ro/d/oferta/apartament-2-camere-titan-IDkBxn2.html",
        "https://www.olx.ro/d/oferta/apartament-3-camere-ghencea-IDkHrfc.html",
    ]


def test_download_listings_writes_files_and_stops_at_target_count(tmp_path: Path):
    session = FakeSession(
        category_pages={1: CATEGORY_PAGE_HTML},
        listing_pages={
            "https://www.olx.ro/d/oferta/apartament-2-camere-titan-IDkBxn2.html": LISTING_HTML_TEMPLATE.format(listing_id="kBxn2"),
            "https://www.olx.ro/d/oferta/apartament-3-camere-ghencea-IDkHrfc.html": LISTING_HTML_TEMPLATE.format(listing_id="kHrfc"),
        },
    )

    downloaded = download_listings(1, tmp_path, session=session, rate_limit_seconds=0)

    assert len(downloaded) == 1
    assert len(list(tmp_path.glob("*.html"))) == 1


def test_download_listings_is_idempotent_and_skips_existing_files(tmp_path: Path):
    (tmp_path / "kBxn2.html").write_text("already here", encoding="utf-8")
    session = FakeSession(
        category_pages={1: CATEGORY_PAGE_HTML},
        listing_pages={
            "https://www.olx.ro/d/oferta/apartament-2-camere-titan-IDkBxn2.html": LISTING_HTML_TEMPLATE.format(listing_id="kBxn2"),
            "https://www.olx.ro/d/oferta/apartament-3-camere-ghencea-IDkHrfc.html": LISTING_HTML_TEMPLATE.format(listing_id="kHrfc"),
        },
    )

    downloaded = download_listings(2, tmp_path, session=session, rate_limit_seconds=0)

    assert "kBxn2" not in downloaded
    assert (tmp_path / "kBxn2.html").read_text(encoding="utf-8") == "already here"

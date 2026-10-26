# tests/ingest/test_olx_scraper.py
from pathlib import Path
from unittest.mock import patch

import pytest

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
        self.closed = False

    def get(self, url, params=None, headers=None, timeout=None):
        if params and "page" in params:
            return FakeResponse(self._category_pages.get(params["page"], ""))
        return FakeResponse(
            self._listing_pages.get(url, ""), status_code=200 if url in self._listing_pages else 404
        )

    def close(self):
        self.closed = True


def test_listing_id_from_url_extracts_trailing_id():
    assert listing_id_from_url("https://www.olx.ro/d/oferta/apartament-IDkBxn2.html") == "kBxn2"


def test_fetch_listing_urls_from_category_dedupes_and_builds_full_urls():
    session = FakeSession(category_pages={1: CATEGORY_PAGE_HTML}, listing_pages={})
    urls = fetch_listing_urls_from_category(1, session=session, rate_limit_seconds=0)
    assert urls == [
        "https://www.olx.ro/d/oferta/apartament-2-camere-titan-IDkBxn2.html",
        "https://www.olx.ro/d/oferta/apartament-3-camere-ghencea-IDkHrfc.html",
    ]


def test_download_listings_writes_files_and_stops_at_target_count(tmp_path: Path):
    session = FakeSession(
        category_pages={1: CATEGORY_PAGE_HTML},
        listing_pages={
            "https://www.olx.ro/d/oferta/apartament-2-camere-titan-IDkBxn2.html": LISTING_HTML_TEMPLATE.format(
                listing_id="kBxn2"
            ),
            "https://www.olx.ro/d/oferta/apartament-3-camere-ghencea-IDkHrfc.html": LISTING_HTML_TEMPLATE.format(
                listing_id="kHrfc"
            ),
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
            "https://www.olx.ro/d/oferta/apartament-2-camere-titan-IDkBxn2.html": LISTING_HTML_TEMPLATE.format(
                listing_id="kBxn2"
            ),
            "https://www.olx.ro/d/oferta/apartament-3-camere-ghencea-IDkHrfc.html": LISTING_HTML_TEMPLATE.format(
                listing_id="kHrfc"
            ),
        },
    )

    downloaded = download_listings(2, tmp_path, session=session, rate_limit_seconds=0)

    assert "kBxn2" not in downloaded
    assert (tmp_path / "kBxn2.html").read_text(encoding="utf-8") == "already here"


def test_download_listings_does_not_close_injected_session(tmp_path: Path):
    session = FakeSession(category_pages={1: ""}, listing_pages={})
    download_listings(1, tmp_path, session=session, rate_limit_seconds=0)
    assert session.closed is False


def test_download_listings_closes_self_created_session(tmp_path: Path):
    created = FakeSession(category_pages={1: ""}, listing_pages={})
    with patch("realestate.ingest.olx_scraper.requests.Session", return_value=created):
        download_listings(1, tmp_path, rate_limit_seconds=0)
    assert created.closed is True


class _FalsySession(FakeSession):
    # A caller-provided session whose truthiness is False (edge case the old
    # `session or requests.Session()` would silently replace).
    def __bool__(self) -> bool:
        return False


def test_download_listings_uses_falsy_injected_session_without_replacing_it(tmp_path: Path):
    session = _FalsySession(
        category_pages={1: CATEGORY_PAGE_HTML},
        listing_pages={
            "https://www.olx.ro/d/oferta/apartament-2-camere-titan-IDkBxn2.html": LISTING_HTML_TEMPLATE.format(
                listing_id="kBxn2"
            ),
        },
    )

    downloaded = download_listings(1, tmp_path, session=session, rate_limit_seconds=0)

    # The injected (falsy) session was used, not replaced by a real requests.Session,
    assert downloaded == ["kBxn2"]
    # and an injected session is never closed by download_listings.
    assert session.closed is False


def test_download_listings_closes_self_created_session_when_scan_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    created = FakeSession(category_pages={1: ""}, listing_pages={})

    def boom(self, pattern):
        raise OSError("directory scan failed")

    monkeypatch.setattr(Path, "glob", boom)
    with (
        patch("realestate.ingest.olx_scraper.requests.Session", return_value=created),
        pytest.raises(OSError),
    ):
        download_listings(1, tmp_path, rate_limit_seconds=0)
    assert created.closed is True


def test_fetch_listing_urls_from_category_sleeps_for_rate_limit():
    session = FakeSession(category_pages={1: CATEGORY_PAGE_HTML}, listing_pages={})

    with patch("realestate.ingest.olx_scraper.time.sleep") as mock_sleep:
        fetch_listing_urls_from_category(1, session=session, rate_limit_seconds=3)

    mock_sleep.assert_called_once_with(3)


def test_download_listings_rate_limits_category_page_fetches_too(tmp_path: Path):
    # All listings on page 1 are already downloaded, and page 2 has none, so no
    # listing-page GET ever happens -- only category-page fetches occur, and those
    # must still be rate limited on their own (an idempotent re-run shouldn't
    # hammer the category endpoint with zero delay between pages).
    (tmp_path / "kBxn2.html").write_text("already here", encoding="utf-8")
    (tmp_path / "kHrfc.html").write_text("already here", encoding="utf-8")
    session = FakeSession(
        category_pages={1: CATEGORY_PAGE_HTML, 2: ""},
        listing_pages={},
    )

    with patch("realestate.ingest.olx_scraper.time.sleep") as mock_sleep:
        download_listings(10, tmp_path, session=session, rate_limit_seconds=4)

    assert mock_sleep.call_count == 2
    mock_sleep.assert_called_with(4)


def test_download_listings_skips_malformed_listing_url_and_continues(tmp_path: Path):
    category_html_with_malformed_href = """
    <html><body>
    <a href="/d/oferta/apartament-fara-id.html">No ID</a>
    <a href="/d/oferta/apartament-3-camere-ghencea-IDkHrfc.html">Ad 2</a>
    </body></html>
    """
    session = FakeSession(
        category_pages={1: category_html_with_malformed_href},
        listing_pages={
            "https://www.olx.ro/d/oferta/apartament-3-camere-ghencea-IDkHrfc.html": LISTING_HTML_TEMPLATE.format(
                listing_id="kHrfc"
            ),
        },
    )

    downloaded = download_listings(5, tmp_path, session=session, rate_limit_seconds=0)

    assert downloaded == ["kHrfc"]
    assert (tmp_path / "kHrfc.html").exists()


def test_download_listings_raises_on_non_200_category_page(tmp_path: Path):
    class FailingCategorySession(FakeSession):
        def get(self, url, params=None, headers=None, timeout=None):
            if params and "page" in params:
                return FakeResponse("", status_code=500)
            return super().get(url, params=params, headers=headers, timeout=timeout)

    session = FailingCategorySession(category_pages={}, listing_pages={})

    with pytest.raises(RuntimeError):
        download_listings(5, tmp_path, session=session, rate_limit_seconds=0)

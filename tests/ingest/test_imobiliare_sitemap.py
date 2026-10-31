import pytest

from realestate.ingest.imobiliare_sitemap import (
    listing_urls,
    sample_per_sector,
    sector_of_url,
    sitemap_url,
)

_BASE = "https://www.imobiliare.ro/oferta/apartament-de-vanzare"


def test_listing_urls_reads_ad_locations_only():
    xml = (
        '<?xml version="1.0"?><urlset>'
        f"<url><loc>{_BASE}-sector-3-titan-2-camere-1</loc></url>"
        "<url><loc>https://www.imobiliare.ro/vanzare-apartamente/bucuresti</loc></url>"
        f"<url><loc>{_BASE}-sector-1-a&amp;b-2-camere-2</loc></url>"
        "</urlset>"
    )

    assert listing_urls(xml) == [
        f"{_BASE}-sector-3-titan-2-camere-1",
        f"{_BASE}-sector-1-a&b-2-camere-2",
    ]


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (f"{_BASE}-sector-3-titan-2-camere-275887313", 3),
        ("https://www.imobiliare.ro/oferta/apartament-de-inchiriat-sector-6-militari-mobilat-1", 6),
        (f"{_BASE}-periferie-3-camere-275887313", None),
        (f"{_BASE}-sector-9-x-1", None),
    ],
)
def test_sector_of_url(url, expected):
    assert sector_of_url(url) == expected


def _urls(sector: int, count: int) -> list[str]:
    return [f"{_BASE}-sector-{sector}-x-{sector}{i:04d}" for i in range(count)]


def test_sample_per_sector_takes_an_equal_share_from_each_sector():
    urls = _urls(1, 50) + _urls(3, 10) + _urls(5, 2) + [f"{_BASE}-periferie-1"]

    sample = sample_per_sector(urls, per_sector=5, seed=7)

    sectors = [sector_of_url(url) for url in sample]
    assert sectors.count(1) == 5
    assert sectors.count(3) == 5
    assert sectors.count(5) == 2
    assert None not in sectors
    assert len(set(sample)) == len(sample)


def test_sample_per_sector_is_reproducible_and_ignores_input_order():
    urls = _urls(1, 50) + _urls(2, 50)

    first = sample_per_sector(urls, per_sector=10, seed=7)

    assert sample_per_sector(list(reversed(urls)), per_sector=10, seed=7) == first
    assert sample_per_sector(urls, per_sector=10, seed=8) != first


@pytest.mark.parametrize(
    ("offer_type", "expected"),
    [
        ("sell", "https://www.imobiliare.ro/sitemap-listings-apartments-for-sale-bucuresti.xml"),
        ("rent", "https://www.imobiliare.ro/sitemap-listings-apartments-for-rent-bucuresti.xml"),
    ],
)
def test_sitemap_url(offer_type, expected):
    assert sitemap_url(offer_type) == expected


def test_sitemap_url_rejects_unknown_offer_types():
    with pytest.raises(ValueError):
        sitemap_url("buy")

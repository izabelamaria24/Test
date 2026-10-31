"""Pick imobiliare.ro ads from the site's published sitemaps of every live Bucharest apartment ad.

Sampling the sitemap avoids the bias of the site's own search, which puts promoted and new ads
first.
"""

import html
import random
import re
from collections import defaultdict

_SITEMAP_TYPES = {"sell": "sale", "rent": "rent"}
_LOC = re.compile(r"<loc>([^<]+)</loc>")
_SECTOR = re.compile(r"-sector-([1-6])-")


def sitemap_url(offer_type: str) -> str:
    if offer_type not in _SITEMAP_TYPES:
        raise ValueError(f"offer_type must be one of {tuple(_SITEMAP_TYPES)}, got {offer_type!r}")
    return (
        "https://www.imobiliare.ro/sitemap-listings-apartments-for-"
        f"{_SITEMAP_TYPES[offer_type]}-bucuresti.xml"
    )


def listing_urls(sitemap_xml: str) -> list[str]:
    urls = (html.unescape(match) for match in _LOC.findall(sitemap_xml))
    return [url for url in urls if "/oferta/" in url]


def sector_of_url(url: str) -> int | None:
    match = _SECTOR.search(url)
    return int(match.group(1)) if match else None


def sample_per_sector(urls: list[str], *, per_sector: int, seed: int) -> list[str]:
    """Up to `per_sector` random ads from each sector; ads with no sector in the URL are skipped."""
    by_sector: dict[int, list[str]] = defaultdict(list)
    for url in set(urls):
        sector = sector_of_url(url)
        if sector is not None:
            by_sector[sector].append(url)
    rng = random.Random(seed)
    sample: list[str] = []
    for sector in sorted(by_sector):
        candidates = sorted(by_sector[sector])
        sample.extend(rng.sample(candidates, min(per_sector, len(candidates))))
    return sample

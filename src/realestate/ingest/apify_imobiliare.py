"""Buy imobiliare.ro listings through the third-party Apify actor `imobiliare-ro-scraper`.

Paid per run; the pilot cost about $1.50 per 1,000 listings with details. Every run is capped
with `max_charge_usd`.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import requests

_API = "https://api.apify.com/v2"
_ACTOR = "unfenced-group~imobiliare-ro-scraper"
_OFFER_TYPES = ("sell", "rent")
_TERMINAL_STATUSES = ("SUCCEEDED", "FAILED", "TIMED-OUT", "ABORTED")
_SERVER_WAIT_SECONDS = 60


@dataclass(frozen=True)
class FetchResult:
    listings: list[dict]
    # As reported right after the run ends; Apify may still adjust it slightly.
    cost_usd: float | None


def fetch_imobiliare_listings(
    token: str,
    *,
    offer_type: str,
    max_items: int,
    max_charge_usd: float,
    location: str = "bucuresti",
    start_urls: list[str] | None = None,
    post_fn: Callable[..., Any] = requests.post,
    get_fn: Callable[..., Any] = requests.get,
) -> FetchResult:
    if offer_type not in _OFFER_TYPES:
        raise ValueError(f"offer_type must be one of {_OFFER_TYPES}, got {offer_type!r}")
    headers = {"Authorization": f"Bearer {token}"}
    run_input: dict[str, Any] = {
        "offerType": offer_type,
        "propertyType": "apartamente",
        "location": location,
        "maxItems": max_items,
        # Coordinates, street address and the full description are only on detail pages.
        "fetchDetails": True,
    }
    if start_urls is not None:
        # Ad (/oferta/) URLs only: the actor returns one empty record for a search page URL.
        run_input["startUrls"] = [{"url": url} for url in start_urls]

    response = post_fn(
        f"{_API}/acts/{_ACTOR}/runs",
        params={"maxItems": max_items, "maxTotalChargeUsd": max_charge_usd},
        json=run_input,
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()
    run = response.json()["data"]

    while run["status"] not in _TERMINAL_STATUSES:
        response = get_fn(
            f"{_API}/actor-runs/{run['id']}",
            params={"waitForFinish": _SERVER_WAIT_SECONDS},
            headers=headers,
            timeout=_SERVER_WAIT_SECONDS + 30,
        )
        response.raise_for_status()
        run = response.json()["data"]
    if run["status"] != "SUCCEEDED":
        raise RuntimeError(f"Apify run {run['id']} ended with status {run['status']}")

    response = get_fn(
        f"{_API}/datasets/{run['defaultDatasetId']}/items",
        params={"format": "json", "clean": "true"},
        headers=headers,
        timeout=120,
    )
    response.raise_for_status()
    return FetchResult(listings=response.json(), cost_usd=run.get("usageTotalUsd"))

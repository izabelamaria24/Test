"""Buy imobiliare.ro listings via Apify and save the raw JSON to data/raw/imobiliare/.

Usage (APIFY_TOKEN must be set in your shell):
  python scripts/fetch_imobiliare.py {sell|rent} MAX_ITEMS MAX_CHARGE_USD
  python scripts/fetch_imobiliare.py sample PER_SECTOR MAX_CHARGE_USD_PER_RUN
The second form picks PER_SECTOR random ads per sector and type from imobiliare.ro's sitemaps of
every live ad, instead of the top of the site's ranked search. The sample is saved on first use so
re-runs fetch the same ads, and batches already saved are skipped.
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import requests

from realestate.enrich.geocoder import USER_AGENT
from realestate.ingest.apify_imobiliare import FetchResult, fetch_imobiliare_listings
from realestate.ingest.imobiliare_sitemap import listing_urls, sample_per_sector, sitemap_url

OUT_DIR = Path("data/raw/imobiliare")
# Kept apart from OUT_DIR, whose JSON files the loader reads as listings.
SAMPLE_DIR = Path("data/raw/imobiliare_samples")
SEED = 20261031
BATCH_SIZE = 100
# The free plan refused a 6th simultaneous run with 402 Payment Required.
PARALLEL_RUNS = 3


def _save(result: FetchResult, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = OUT_DIR / f"{name}_{stamp}.json"
    out_path.write_text(json.dumps(result.listings, ensure_ascii=False, indent=1), encoding="utf-8")
    cost = f"~${result.cost_usd:.4f}" if result.cost_usd is not None else "unknown"
    print(f"{name}: saved {len(result.listings)} listings to {out_path} (cost {cost})", flush=True)


def _sample(offer_type: str, per_sector: int) -> list[str]:
    path = SAMPLE_DIR / f"{offer_type}_{per_sector}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8").split()
    response = requests.get(sitemap_url(offer_type), headers={"User-Agent": USER_AGENT}, timeout=60)
    response.raise_for_status()
    urls = sample_per_sector(listing_urls(response.text), per_sector=per_sector, seed=SEED)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(urls) + "\n", encoding="utf-8")
    print(f"{offer_type}: sampled {len(urls)} ads, saved to {path}")
    return urls


if __name__ == "__main__":
    token = os.environ.get("APIFY_TOKEN")
    if not token:
        sys.exit(
            "Set APIFY_TOKEN in your shell first (Apify Console > Settings > API & Integrations)."
        )
    mode, max_items, max_charge_usd = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])

    if mode in ("sell", "rent"):
        result = fetch_imobiliare_listings(
            token, offer_type=mode, max_items=max_items, max_charge_usd=max_charge_usd
        )
        _save(result, mode)
        sys.exit(0)
    if mode != "sample":
        sys.exit(f"unknown mode {mode!r}; use sell, rent or sample")

    jobs: dict[str, tuple[str, list[str]]] = {}
    for offer_type in ("sell", "rent"):
        urls = _sample(offer_type, max_items)
        for start in range(0, len(urls), BATCH_SIZE):
            name = f"{offer_type}_sample{max_items}_batch{start // BATCH_SIZE:02d}"
            jobs[name] = (offer_type, urls[start : start + BATCH_SIZE])
    done = [name for name in jobs if any(OUT_DIR.glob(f"{name}_*.json"))]
    for name in done:
        print(f"{name}: already saved, skipping")
        del jobs[name]
    print(f"Starting {len(jobs)} runs, at most ${max_charge_usd * len(jobs):.2f} in total.")
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=PARALLEL_RUNS) as pool:
        futures = {
            pool.submit(
                fetch_imobiliare_listings,
                token,
                offer_type=offer_type,
                max_items=len(batch),
                max_charge_usd=max_charge_usd,
                start_urls=batch,
            ): name
            for name, (offer_type, batch) in jobs.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                _save(future.result(), name)
            except Exception as exc:  # noqa: BLE001 - report and keep the other runs' data
                failed.append(name)
                response = getattr(exc, "response", None)
                detail = f": {response.text[:300]}" if response is not None else ""
                print(f"{name}: FAILED ({exc}){detail}", flush=True)
    if failed:
        sys.exit(f"{len(failed)} run(s) failed: {', '.join(sorted(failed))}")

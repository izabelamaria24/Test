"""Buy imobiliare.ro listings via Apify and save the raw JSON to data/raw/imobiliare/.

Usage (APIFY_TOKEN must be set in your shell):
  python scripts/fetch_imobiliare.py {sell|rent} MAX_ITEMS MAX_CHARGE_USD
  python scripts/fetch_imobiliare.py sectors PER_SECTOR MAX_CHARGE_USD_PER_RUN
The second form runs sale and rent for each of Bucharest's six sectors in parallel, so the
sample is spread across the city instead of the top of one ranked search.
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from realestate.ingest.apify_imobiliare import (
    FetchResult,
    fetch_imobiliare_listings,
    sector_search_url,
)

OUT_DIR = Path("data/raw/imobiliare")
PARALLEL_RUNS = 6


def _save(result: FetchResult, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = OUT_DIR / f"{name}_{stamp}.json"
    out_path.write_text(json.dumps(result.listings, ensure_ascii=False, indent=1), encoding="utf-8")
    cost = f"~${result.cost_usd:.4f}" if result.cost_usd is not None else "unknown"
    print(f"{name}: saved {len(result.listings)} listings to {out_path} (cost {cost})", flush=True)


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
    if mode != "sectors":
        sys.exit(f"unknown mode {mode!r}; use sell, rent or sectors")

    jobs = {
        f"{offer_type}_sector{sector}": (offer_type, sector_search_url(offer_type, sector))
        for offer_type in ("sell", "rent")
        for sector in range(1, 7)
    }
    print(f"Starting {len(jobs)} runs, at most ${max_charge_usd * len(jobs):.2f} in total.")
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=PARALLEL_RUNS) as pool:
        futures = {
            pool.submit(
                fetch_imobiliare_listings,
                token,
                offer_type=offer_type,
                max_items=max_items,
                max_charge_usd=max_charge_usd,
                start_url=url,
            ): name
            for name, (offer_type, url) in jobs.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                _save(future.result(), name)
            except Exception as exc:  # noqa: BLE001 - report and keep the other runs' data
                failed.append(name)
                print(f"{name}: FAILED ({exc})", flush=True)
    if failed:
        sys.exit(f"{len(failed)} run(s) failed: {', '.join(sorted(failed))}")

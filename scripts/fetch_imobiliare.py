"""Buy a batch of imobiliare.ro listings via Apify and save the raw JSON.

Usage: APIFY_TOKEN=... python scripts/fetch_imobiliare.py {sell|rent} MAX_ITEMS MAX_CHARGE_USD
"""

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from realestate.ingest.apify_imobiliare import fetch_imobiliare_listings

if __name__ == "__main__":
    offer_type, max_items, max_charge_usd = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    token = os.environ.get("APIFY_TOKEN")
    if not token:
        sys.exit(
            "Set APIFY_TOKEN in your shell first (Apify Console > Settings > API & Integrations)."
        )

    result = fetch_imobiliare_listings(
        token, offer_type=offer_type, max_items=max_items, max_charge_usd=max_charge_usd
    )

    out_dir = Path("data/raw/imobiliare")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"{offer_type}_{stamp}.json"
    out_path.write_text(json.dumps(result.listings, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Saved {len(result.listings)} listings to {out_path}")
    if result.cost_usd is not None and result.listings:
        per_1000 = result.cost_usd / len(result.listings) * 1000
        print(f"Run cost ~${result.cost_usd:.4f} (~${per_1000:.2f} per 1,000 listings)")

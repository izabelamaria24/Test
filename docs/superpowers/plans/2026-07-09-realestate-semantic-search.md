# Real Estate Semantic Search (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 1 foundation from the design spec — ingest real Bucharest listings (scraped from OLX.ro, which robots.txt explicitly permits and which has no bot-protection challenge, unlike imobiliare.ro/storia.ro — see Global Constraints), enrich them with geocoding and subway-walking-distance, embed and store them in a vector DB, and answer natural-language queries via hybrid (structured filter + geospatial + dense vector) retrieval, with an offline evaluation harness.

**Architecture:** A modular-monolith Python package (`realestate`) with five interface-bound layers — `ingest`, `enrich`, `embed`, `store`, `query` — plus an `eval` layer for offline IR metrics. Each layer depends only on the abstractions of its neighbors (see Dependency Inversion in the spec), so components are swappable and independently testable.

**Tech Stack:** Python 3.11, pydantic (schemas), BeautifulSoup4 + lxml (HTML parsing), requests (Nominatim/Overpass/OSRM/Ollama HTTP calls), sentence-transformers + PyTorch/MPS (embeddings, `intfloat/multilingual-e5-base`), Qdrant (self-hosted via Docker) + qdrant-client, Ollama (local LLM, query parsing), pytest.

## Global Constraints

- No automated scraping of imobiliare.ro/storia.ro — both actively block automated requests with a Cloudflare managed challenge (confirmed empirically: a plain request to an individual listing page returns a "Just a moment..." JS challenge page, not just a 403). Bypassing that requires browser-automation evasion techniques that are out of scope, full stop.
- OLX.ro is the real-data source instead: its `robots.txt` explicitly allows crawling (`Allow: /`, no listing-page disallow), it returns plain 200 responses with no bot-protection challenge, and each listing page embeds a clean JSON blob (`window.__PRERENDERED_STATE__`) with structured fields — no fragile CSS-selector scraping needed. Scraping is still rate-limited (2s between requests) as ordinary good practice, not because it's required to get past a defense.
- Geocoding uses OpenStreetMap Nominatim's public instance — max 1 request/second, custom `User-Agent` header required, results cached to disk (Nominatim usage policy).
- Subway-distance uses OpenStreetMap Overpass (station locations) + a self-hosted OSRM instance (walking routes) — not Google Maps.
- All LLM inference (query parsing, eval query generation) runs locally via Ollama — no external LLM API calls in this plan.
- Embedding model: `intfloat/multilingual-e5-base` via `sentence-transformers`, run on the `mps` device.

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `docker-compose.yml`
- Create: `.gitignore`
- Create: `src/realestate/__init__.py`
- Test: `tests/test_scaffolding.py`

**Interfaces:**
- Produces: an installable `realestate` package importable from `tests/`, and a `qdrant` service reachable at `http://localhost:6333` once `docker compose up -d` is run.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "realestate"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.7",
    "beautifulsoup4>=4.12",
    "lxml>=5.2",
    "requests>=2.32",
    "sentence-transformers>=3.0",
    "qdrant-client>=1.11",
]

[project.optional-dependencies]
dev = ["pytest>=8.2"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src", "."]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Write `docker-compose.yml`**

```yaml
services:
  qdrant:
    image: qdrant/qdrant:v1.11.0
    ports:
      - "6333:6333"
    volumes:
      - ./data/qdrant_storage:/qdrant/storage
  osrm:
    image: osrm/osrm-backend:v5.27.1
    ports:
      - "5000:5000"
    volumes:
      - ./data/osrm:/data
    command: osrm-routed --algorithm mld /data/romania-latest.osrm
```

- [ ] **Step 3: Write `.gitignore`**

```
__pycache__/
*.pyc
.venv/
data/raw/
data/cache/
data/qdrant_storage/
data/osrm/
*.egg-info/
.worktrees/
.tokensave
```

- [ ] **Step 4: Create the package and install in editable mode**

```bash
mkdir -p src/realestate tests
touch src/realestate/__init__.py
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

- [ ] **Step 5: Write the smoke test**

```python
# tests/test_scaffolding.py
import realestate


def test_package_importable():
    assert realestate is not None
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_scaffolding.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml docker-compose.yml .gitignore src/realestate/__init__.py tests/test_scaffolding.py
git commit -m "chore: scaffold realestate package"
```

---

### Task 2: Core schema — `RawListing` and `EnrichedListing`

**Files:**
- Create: `src/realestate/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `RawListing(external_id: str, title: str, description: str, price_raw: str, address_raw: str, specs: dict[str, str], image_urls: list[str], source_file: str, map_lat: float | None, map_lon: float | None)`
- Produces: `EnrichedListing(external_id: str, title: str, description: str, price_eur: float, rooms: int | None, built_area_sqm: float | None, floor_number: int | None, construction_year_range: str | None, layout_type: str | None, address_text: str, latitude: float | None, longitude: float | None, location_confidence: str, nearest_subway_station: str | None, subway_walking_minutes: float | None, image_urls: list[str])`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
from realestate.models import RawListing, EnrichedListing


def test_raw_listing_construction():
    listing = RawListing(
        external_id="304473136",
        title="Vand apartament 2 camere TITAN",
        description="Direct proprietar, vand apartament 2 camere...",
        price_raw="100.000 €",
        address_raw="Bucuresti - Ilfov, Bucuresti, Sectorul 3",
        specs={"Suprafata utila": "48 m²"},
        image_urls=["https://frankfurt.apollo.olxcdn.com:443/v1/files/example/image"],
        source_file="304473136.html",
        map_lat=44.42,
        map_lon=26.10,
    )
    assert listing.external_id == "304473136"
    assert listing.specs["Suprafata utila"] == "48 m²"
    assert listing.map_lat == 44.42


def test_enriched_listing_defaults_for_missing_location():
    listing = EnrichedListing(
        external_id="304473136",
        title="Vand apartament 2 camere TITAN",
        description="Direct proprietar, vand apartament 2 camere...",
        price_eur=100000.0,
        rooms=2,
        built_area_sqm=48.0,
        floor_number=3,
        construction_year_range="1977 – 1990",
        layout_type="Decomandat",
        address_text="Bucuresti - Ilfov, Bucuresti, Sectorul 3",
        latitude=None,
        longitude=None,
        location_confidence="low_confidence_location",
        nearest_subway_station=None,
        subway_walking_minutes=None,
        image_urls=[],
    )
    assert listing.location_confidence == "low_confidence_location"
    assert listing.latitude is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'realestate.models'`

- [ ] **Step 3: Write the implementation**

```python
# src/realestate/models.py
from pydantic import BaseModel


class RawListing(BaseModel):
    external_id: str
    title: str
    description: str
    price_raw: str
    address_raw: str
    specs: dict[str, str]
    image_urls: list[str]
    source_file: str
    # OLX's own JSON gives an approximate lat/lon per listing (see ad.map in the JSON schema) —
    # privacy-fuzzed (radius + show_detailed:false observed on every real listing sampled so far),
    # but free (no network call) and often finer-grained than geocoding the sparse text address.
    # None when a listing's JSON omits map data (not observed yet, but the schema doesn't guarantee it).
    map_lat: float | None
    map_lon: float | None


class EnrichedListing(BaseModel):
    external_id: str
    title: str
    description: str
    price_eur: float
    rooms: int | None
    built_area_sqm: float | None
    floor_number: int | None
    construction_year_range: str | None
    layout_type: str | None
    address_text: str
    latitude: float | None
    longitude: float | None
    location_confidence: str
    nearest_subway_station: str | None
    subway_walking_minutes: float | None
    image_urls: list[str]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/realestate/models.py tests/test_models.py
git commit -m "feat: add RawListing and EnrichedListing schemas"
```

---

### Task 3: OLX.ro listing JSON parser

OLX embeds each listing's full structured data as JSON in a `window.__PRERENDERED_STATE__= "...";`
assignment in the page `<script>` — a JSON string literal containing escaped JSON (decode twice).
This was verified directly against a live listing page, not guessed: real fields are `id`, `title`,
`description` (HTML-formatted), `price.regularPrice.{value,currencySymbol}`, `location.pathName`,
`photos` (list of image URLs), and `params` (list of `{name, value}` spec rows — e.g. `Suprafata
utila`, `Etaj`, `An constructie`, `Compartimentare`).

**Files:**
- Create: `src/realestate/ingest/__init__.py`
- Create: `src/realestate/ingest/olx_parser.py`
- Test: `tests/ingest/test_olx_parser.py`

**Interfaces:**
- Consumes: `RawListing` from `realestate.models`
- Produces: `parse_olx_listing_html(html: str, external_id: str) -> RawListing`

- [ ] **Step 1: Write the failing test using a real, verified schema**

```python
# tests/ingest/test_olx_parser.py
import json
import pytest
from realestate.ingest.olx_parser import parse_olx_listing_html

# Fields below mirror the real structure of window.__PRERENDERED_STATE__ verified against
# a live OLX.ro listing page — not a guessed schema. Building the fixture via json.dumps
# (rather than hand-writing escaped JSON) keeps the double-encoding correct.
SAMPLE_AD = {
    "id": 304473136,
    "title": "Vand apartament 2 camere TITAN",
    "description": (
        "Direct proprietar !<br />\nVand apartament 2 camere decomandat<br />\n"
        "* Suprafata utila 48 utili"
    ),
    "price": {
        "regularPrice": {"value": 100000, "currencyCode": "EUR", "currencySymbol": "€"}
    },
    "location": {"pathName": "Bucuresti - Ilfov, Bucuresti, Sectorul 3"},
    "params": [
        {"key": "compartimentare", "name": "Compartimentare", "value": "Decomandat"},
        {"key": "m", "name": "Suprafata utila", "value": "48 m²"},
        {"key": "constructie", "name": "An constructie", "value": "1977 – 1990"},
        {"key": "floor", "name": "Etaj", "value": "3"},
    ],
    "photos": ["https://frankfurt.apollo.olxcdn.com:443/v1/files/dlbik2gpbb3j1-RO/image;s=750x1000"],
    "map": {"lat": 44.42, "lon": 26.1, "radius": 3, "show_detailed": False, "zoom": 12},
}


def _build_sample_html(ad_data: dict) -> str:
    inner_json_text = json.dumps({"ad": {"ad": ad_data}}, ensure_ascii=False)
    js_string_literal = json.dumps(inner_json_text, ensure_ascii=False)
    return (
        "<html><head><script>window.__PRERENDERED_STATE__= "
        f"{js_string_literal};\n</script></head><body></body></html>"
    )


SAMPLE_HTML = _build_sample_html(SAMPLE_AD)


def test_parses_price():
    listing = parse_olx_listing_html(SAMPLE_HTML, external_id="304473136")
    assert listing.price_raw == "100.000 €"


def test_parses_title_and_strips_html_from_description():
    listing = parse_olx_listing_html(SAMPLE_HTML, external_id="304473136")
    assert listing.title == "Vand apartament 2 camere TITAN"
    assert "Suprafata utila 48 utili" in listing.description
    assert "<br" not in listing.description


def test_parses_address_from_location_pathname():
    listing = parse_olx_listing_html(SAMPLE_HTML, external_id="304473136")
    assert listing.address_raw == "Bucuresti - Ilfov, Bucuresti, Sectorul 3"


def test_parses_specs_by_display_name():
    listing = parse_olx_listing_html(SAMPLE_HTML, external_id="304473136")
    assert listing.specs["Suprafata utila"] == "48 m²"
    assert listing.specs["Etaj"] == "3"
    assert listing.specs["Compartimentare"] == "Decomandat"


def test_parses_image_urls():
    listing = parse_olx_listing_html(SAMPLE_HTML, external_id="304473136")
    assert listing.image_urls == [
        "https://frankfurt.apollo.olxcdn.com:443/v1/files/dlbik2gpbb3j1-RO/image;s=750x1000"
    ]


def test_raises_when_prerendered_state_missing():
    with pytest.raises(ValueError, match="PRERENDERED_STATE"):
        parse_olx_listing_html("<html><body>no state here</body></html>", external_id="000")


def test_parses_map_coordinates():
    listing = parse_olx_listing_html(SAMPLE_HTML, external_id="304473136")
    assert listing.map_lat == 44.42
    assert listing.map_lon == 26.1


def test_map_coordinates_default_to_none_when_absent():
    ad_without_map = {k: v for k, v in SAMPLE_AD.items() if k != "map"}
    html = _build_sample_html(ad_without_map)
    listing = parse_olx_listing_html(html, external_id="304473136")
    assert listing.map_lat is None
    assert listing.map_lon is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ingest/test_olx_parser.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'realestate.ingest.olx_parser'`

- [ ] **Step 3: Write the implementation**

```python
# src/realestate/ingest/olx_parser.py
import json

from bs4 import BeautifulSoup
from realestate.models import RawListing

_STATE_PREFIX = "window.__PRERENDERED_STATE__= "


def _extract_ad_json(html: str) -> dict:
    start = html.find(_STATE_PREFIX)
    if start == -1:
        raise ValueError("PRERENDERED_STATE script assignment not found")
    start += len(_STATE_PREFIX)
    end = html.find(";\n", start)
    if end == -1:
        raise ValueError("could not find end of PRERENDERED_STATE assignment")
    js_string_literal = html[start:end]
    inner_json_text = json.loads(js_string_literal)  # un-escape the JS string
    return json.loads(inner_json_text)  # parse the actual JSON payload


def _clean_description(raw_html_description: str) -> str:
    return BeautifulSoup(raw_html_description, "lxml").get_text(separator="\n").strip()


def parse_olx_listing_html(html: str, external_id: str) -> RawListing:
    state = _extract_ad_json(html)
    ad = state["ad"]["ad"]

    price_info = ad.get("price", {}).get("regularPrice")
    if not price_info:
        raise ValueError(f"listing {external_id}: no regular price found")
    price_raw = f"{price_info['value']:,.0f} {price_info['currencySymbol']}".replace(",", ".")

    specs = {param["name"]: param["value"] for param in ad.get("params", [])}
    map_data = ad.get("map") or {}

    return RawListing(
        external_id=external_id,
        title=ad.get("title", ""),
        description=_clean_description(ad.get("description", "")),
        price_raw=price_raw,
        address_raw=ad.get("location", {}).get("pathName", ""),
        specs=specs,
        image_urls=ad.get("photos", []),
        source_file=f"{external_id}.html",
        map_lat=map_data.get("lat"),
        map_lon=map_data.get("lon"),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ingest/test_olx_parser.py -v`
Expected: PASS (8 passed)

Note: OLX's `map` data is privacy-fuzzed in practice — every real listing sampled during this
project had `show_detailed: false` with a `radius`, and several distinct listings shared the exact
same coordinates (neighborhood-level precision, not per-building). Still worth using directly
(Task 8 prefers it over geocoding, see below) since it's free and at least as precise as the sparse
text address, but don't treat `map_lat`/`map_lon` as an exact location.

- [ ] **Step 5: Commit**

```bash
mkdir -p tests/ingest
touch tests/ingest/__init__.py
git add src/realestate/ingest/ tests/ingest/
git commit -m "feat: parse OLX.ro listing JSON into RawListing"
```

---

### Task 4: OLX.ro category scraper + directory loader

Two responsibilities: (a) paginate OLX's Bucharest apartment-sale category pages (confirmed
working via plain HTTP GET, `?page=N` pagination verified against live pages) to discover listing
URLs and download each one's HTML to disk, idempotently and rate-limited; (b) load a directory of
already-downloaded HTML files and parse them, quarantining failures rather than crashing.

**Files:**
- Create: `src/realestate/ingest/olx_scraper.py`
- Create: `src/realestate/ingest/olx_loader.py`
- Test: `tests/ingest/test_olx_scraper.py`
- Test: `tests/ingest/test_olx_loader.py`

**Interfaces:**
- Consumes: `parse_olx_listing_html` from Task 3
- Produces: `fetch_listing_urls_from_category(page: int, *, session) -> list[str]`
- Produces: `listing_id_from_url(url: str) -> str`
- Produces: `download_listings(target_count: int, output_dir: Path, *, session=None, rate_limit_seconds=2.0, max_pages=50) -> list[str]` (returns newly downloaded listing ids; skips ones already on disk)
- Produces: `load_olx_html_directory(directory: Path) -> tuple[list[RawListing], list[tuple[Path, str]]]`

- [ ] **Step 1: Write the failing scraper test (network calls injected via a fake session)**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ingest/test_olx_scraper.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'realestate.ingest.olx_scraper'`

- [ ] **Step 3: Write the scraper implementation**

```python
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


def fetch_listing_urls_from_category(page: int, *, session: requests.Session) -> list[str]:
    response = session.get(
        CATEGORY_URL, params={"page": page}, headers={"User-Agent": USER_AGENT}, timeout=15
    )
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

        listing_urls = fetch_listing_urls_from_category(page, session=session)
        if not listing_urls:
            break

        for url in listing_urls:
            if len(list(output_dir.glob("*.html"))) >= target_count:
                break
            listing_id = listing_id_from_url(url)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ingest/test_olx_scraper.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Write the failing loader test**

```python
# tests/ingest/test_olx_loader.py
from pathlib import Path
from realestate.ingest.olx_loader import load_olx_html_directory
from tests.ingest.test_olx_parser import SAMPLE_HTML

MALFORMED_HTML = "<html><body>no prerendered state here</body></html>"


def test_loads_valid_html_files(tmp_path: Path):
    (tmp_path / "304473136.html").write_text(SAMPLE_HTML, encoding="utf-8")

    listings, failures = load_olx_html_directory(tmp_path)

    assert len(listings) == 1
    assert listings[0].external_id == "304473136"
    assert failures == []


def test_quarantines_malformed_files_instead_of_raising(tmp_path: Path):
    (tmp_path / "304473136.html").write_text(SAMPLE_HTML, encoding="utf-8")
    (tmp_path / "999.html").write_text(MALFORMED_HTML, encoding="utf-8")

    listings, failures = load_olx_html_directory(tmp_path)

    assert len(listings) == 1
    assert len(failures) == 1
    assert failures[0][0].name == "999.html"
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/ingest/test_olx_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'realestate.ingest.olx_loader'`

- [ ] **Step 7: Write the loader implementation**

```python
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
```

- [ ] **Step 8: Run both test files to verify everything passes**

Run: `pytest tests/ingest/ -v`
Expected: PASS (all tests across Task 3 and Task 4)

- [ ] **Step 9: Commit**

```bash
git add src/realestate/ingest/olx_scraper.py src/realestate/ingest/olx_loader.py tests/ingest/test_olx_scraper.py tests/ingest/test_olx_loader.py
git commit -m "feat: scrape OLX.ro category pages and load downloaded listings"
```

- [ ] **Step 10: Manual smoke test — download real listings**

Run:
```python
from pathlib import Path
from realestate.ingest.olx_scraper import download_listings

downloaded = download_listings(100, Path("data/raw/olx_html"))
print(f"downloaded {len(downloaded)} new listings")
```
Expected: after a couple minutes (rate-limited at 2s/request), `data/raw/olx_html/` contains ~100
`.html` files. If OLX's markup has changed since this was verified, some downloads may fail to
parse later in Task 11 — that's what the quarantine mechanism is for.

---

### Task 5: Field normalizer

**Files:**
- Create: `src/realestate/enrich/__init__.py`
- Create: `src/realestate/enrich/normalizer.py`
- Test: `tests/enrich/test_normalizer.py`

**Interfaces:**
- Produces: `normalize_price_eur(price_raw: str) -> float`
- Produces: `normalize_rooms(title: str) -> int | None`
- Produces: `normalize_specs(specs: dict[str, str]) -> dict[str, object]` (keys: `built_area_sqm: float | None`, `floor_number: int | None`, `construction_year_range: str | None`, `layout_type: str | None`)

- [ ] **Step 1: Write the failing test**

```python
# tests/enrich/test_normalizer.py
from realestate.enrich.normalizer import (
    normalize_price_eur,
    normalize_rooms,
    normalize_specs,
)


def test_normalize_price_eur_parses_dotted_thousands():
    assert normalize_price_eur("100.000 €") == 100000.0


def test_normalize_rooms_detects_studio():
    assert normalize_rooms("Garsoniera 40mp in Centrul Istoric") == 1


def test_normalize_rooms_detects_explicit_count():
    assert normalize_rooms("Apartament 3 camere de vanzare") == 3


def test_normalize_rooms_returns_none_when_unknown():
    assert normalize_rooms("Teren de vanzare") is None


def test_normalize_specs_maps_known_labels():
    raw_specs = {
        "Suprafata utila": "48 m²",
        "Etaj": "3",
        "An constructie": "1977 – 1990",
        "Compartimentare": "Decomandat",
    }
    normalized = normalize_specs(raw_specs)
    assert normalized["built_area_sqm"] == 48.0
    assert normalized["floor_number"] == 3
    assert normalized["construction_year_range"] == "1977 – 1990"
    assert normalized["layout_type"] == "Decomandat"


def test_normalize_specs_ignores_unknown_labels():
    normalized = normalize_specs({"Un label necunoscut": "valoare"})
    assert normalized == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/enrich/test_normalizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'realestate.enrich.normalizer'`

- [ ] **Step 3: Write the implementation**

```python
# src/realestate/enrich/normalizer.py
import re

SPEC_LABEL_MAP: dict[str, str] = {
    "Suprafata utila": "built_area_sqm",
    "Etaj": "floor_number",
    "An constructie": "construction_year_range",
    "Compartimentare": "layout_type",
}

_NUMERIC_FIELDS = {"built_area_sqm", "floor_number"}


def normalize_price_eur(price_raw: str) -> float:
    match = re.search(r"([\d.]+)\s*€", price_raw)
    if not match:
        raise ValueError(f"unrecognized price format: {price_raw!r}")
    digits = match.group(1).replace(".", "")
    return float(digits)


def normalize_rooms(title: str) -> int | None:
    lowered = title.lower()
    match = re.search(r"(\d+)\s*camere", lowered)
    if match:
        return int(match.group(1))
    if "garsonier" in lowered:
        return 1
    return None


def normalize_specs(specs: dict[str, str]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for raw_label, raw_value in specs.items():
        field = SPEC_LABEL_MAP.get(raw_label)
        if field is None:
            continue
        if field == "built_area_sqm":
            match = re.search(r"([\d.,]+)", raw_value)
            normalized[field] = float(match.group(1).replace(",", ".")) if match else None
        elif field == "floor_number":
            match = re.search(r"(\d+)", raw_value)
            normalized[field] = int(match.group(1)) if match else None
        else:
            normalized[field] = raw_value
    return normalized
```

Note: `SPEC_LABEL_MAP` keys are the exact `name` values OLX's own JSON gives per param (verified
directly against a live listing) — not translated/guessed labels, so this should hold up better
than a scrape targeting rendered HTML text would. Still worth a spot-check against a handful of
real downloaded listings in Task 11: OLX may use different `params` keys for other property types
(houses, land) than the apartment listing this was verified against.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/enrich/test_normalizer.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
mkdir -p tests/enrich
touch tests/enrich/__init__.py
git add src/realestate/enrich/ tests/enrich/
git commit -m "feat: normalize raw price/rooms/specs into typed fields"
```

---

### Task 6: Geocoder (Nominatim, cached)

**Files:**
- Create: `src/realestate/enrich/geocoder.py`
- Test: `tests/enrich/test_geocoder.py`

**Interfaces:**
- Produces: `GeocodeResult(latitude: float, longitude: float)`
- Produces: `CachedGeocoder(cache_path: Path, geocode_fn: Callable[[str], GeocodeResult | None] = geocode_address)` with method `.geocode(address: str) -> GeocodeResult | None`
- Produces: `geocode_address(address: str) -> GeocodeResult | None` (hits the real Nominatim API — not unit tested directly; `CachedGeocoder` is tested with a fake `geocode_fn` so tests don't make network calls)

- [ ] **Step 1: Write the failing test**

```python
# tests/enrich/test_geocoder.py
import json
from pathlib import Path
from realestate.enrich.geocoder import CachedGeocoder, GeocodeResult


def test_cache_miss_calls_geocode_fn_and_persists(tmp_path: Path):
    cache_path = tmp_path / "geocode_cache.json"
    calls: list[str] = []

    def fake_geocode(address: str) -> GeocodeResult | None:
        calls.append(address)
        return GeocodeResult(latitude=44.43, longitude=26.10)

    geocoder = CachedGeocoder(cache_path=cache_path, geocode_fn=fake_geocode)
    result = geocoder.geocode("Centrul Istoric, Bucuresti")

    assert result is not None
    assert result.latitude == 44.43
    assert calls == ["Centrul Istoric, Bucuresti"]
    assert json.loads(cache_path.read_text())["Centrul Istoric, Bucuresti"] == {
        "latitude": 44.43,
        "longitude": 26.10,
    }


def test_cache_hit_does_not_call_geocode_fn_again(tmp_path: Path):
    cache_path = tmp_path / "geocode_cache.json"
    calls: list[str] = []

    def fake_geocode(address: str) -> GeocodeResult | None:
        calls.append(address)
        return GeocodeResult(latitude=44.43, longitude=26.10)

    geocoder = CachedGeocoder(cache_path=cache_path, geocode_fn=fake_geocode)
    geocoder.geocode("Centrul Istoric, Bucuresti")
    geocoder.geocode("Centrul Istoric, Bucuresti")

    assert calls == ["Centrul Istoric, Bucuresti"]


def test_caches_negative_results(tmp_path: Path):
    cache_path = tmp_path / "geocode_cache.json"
    calls: list[str] = []

    def fake_geocode(address: str) -> GeocodeResult | None:
        calls.append(address)
        return None

    geocoder = CachedGeocoder(cache_path=cache_path, geocode_fn=fake_geocode)
    first = geocoder.geocode("Nonexistent Place")
    second = geocoder.geocode("Nonexistent Place")

    assert first is None
    assert second is None
    assert calls == ["Nonexistent Place"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/enrich/test_geocoder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'realestate.enrich.geocoder'`

- [ ] **Step 3: Write the implementation**

```python
# src/realestate/enrich/geocoder.py
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
# Nominatim's usage policy requires an identifying User-Agent and at most 1 req/sec.
USER_AGENT = "realestate-semantic-search-research/0.1"


@dataclass
class GeocodeResult:
    latitude: float
    longitude: float


def geocode_address(address: str, *, min_interval_seconds: float = 1.0) -> GeocodeResult | None:
    response = requests.get(
        NOMINATIM_URL,
        params={"q": f"{address}, Romania", "format": "json", "limit": 1},
        headers={"User-Agent": USER_AGENT},
        timeout=10,
    )
    response.raise_for_status()
    results = response.json()
    time.sleep(min_interval_seconds)
    if not results:
        return None
    return GeocodeResult(latitude=float(results[0]["lat"]), longitude=float(results[0]["lon"]))


class CachedGeocoder:
    def __init__(
        self,
        cache_path: Path,
        geocode_fn: Callable[[str], GeocodeResult | None] = geocode_address,
    ):
        self._cache_path = cache_path
        self._geocode_fn = geocode_fn
        self._cache: dict[str, dict[str, float] | None] = {}
        if cache_path.exists():
            self._cache = json.loads(cache_path.read_text())

    def geocode(self, address: str) -> GeocodeResult | None:
        if address in self._cache:
            cached = self._cache[address]
            return GeocodeResult(**cached) if cached else None

        result = self._geocode_fn(address)
        self._cache[address] = (
            {"latitude": result.latitude, "longitude": result.longitude} if result else None
        )
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(self._cache))
        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/enrich/test_geocoder.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/realestate/enrich/geocoder.py tests/enrich/test_geocoder.py
git commit -m "feat: add cached Nominatim geocoder"
```

---

### Task 7: Subway walking-distance (Overpass + OSRM)

**Files:**
- Create: `src/realestate/enrich/poi.py`
- Test: `tests/enrich/test_poi.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (standalone geospatial utility)
- Produces: `haversine_km(lat1, lon1, lat2, lon2) -> float`
- Produces: `fetch_bucharest_subway_stations() -> list[dict]` (hits Overpass live — not unit tested directly)
- Produces: `osrm_walking_minutes(lat1, lon1, lat2, lon2, osrm_url: str) -> float` (hits OSRM live — not unit tested directly)
- Produces: `nearest_subway_station(lat, lon, stations, *, candidates=3, walking_fn=osrm_walking_minutes) -> tuple[str, float] | None`

- [ ] **Step 1: Write the failing test**

```python
# tests/enrich/test_poi.py
import math
from realestate.enrich.poi import haversine_km, nearest_subway_station

STATIONS = [
    {"name": "Piata Unirii", "lat": 44.4278, "lon": 26.1030},
    {"name": "Universitate", "lat": 44.4356, "lon": 26.1023},
    {"name": "Dristor", "lat": 44.4197, "lon": 26.1489},
]


def test_haversine_km_zero_for_same_point():
    assert haversine_km(44.43, 26.10, 44.43, 26.10) == 0.0


def test_haversine_km_matches_known_approx_distance():
    # Bucharest Piata Unirii to Universitate, roughly ~0.9km apart.
    distance = haversine_km(44.4278, 26.1030, 44.4356, 26.1023)
    assert 0.5 < distance < 1.5


def test_nearest_subway_station_picks_shortest_walking_time():
    def fake_walking_fn(lat1, lon1, lat2, lon2, osrm_url="http://localhost:5000"):
        # Simulate: Universitate is closest by straight-line but has a slow walking route;
        # Piata Unirii is farther by straight-line but has the fastest actual walk.
        if (lat2, lon2) == (44.4356, 26.1023):
            return 25.0
        if (lat2, lon2) == (44.4278, 26.1030):
            return 8.0
        return 40.0

    result = nearest_subway_station(
        44.430, 26.102, STATIONS, candidates=3, walking_fn=fake_walking_fn
    )

    assert result == ("Piata Unirii", 8.0)


def test_nearest_subway_station_returns_none_for_empty_station_list():
    assert nearest_subway_station(44.43, 26.10, [], walking_fn=lambda *a, **k: 0.0) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/enrich/test_poi.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'realestate.enrich.poi'`

- [ ] **Step 3: Write the implementation**

```python
# src/realestate/enrich/poi.py
import math
from typing import Callable

import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def fetch_bucharest_subway_stations() -> list[dict]:
    query = """
    [out:json][timeout:25];
    area["name"="Bucuresti"]["admin_level"="4"]->.searchArea;
    (
      node["railway"="station"]["station"="subway"](area.searchArea);
      node["public_transport"="station"]["subway"="yes"](area.searchArea);
    );
    out body;
    """
    response = requests.post(OVERPASS_URL, data={"data": query}, timeout=30)
    response.raise_for_status()
    elements = response.json()["elements"]
    return [
        {"name": el.get("tags", {}).get("name", "unknown"), "lat": el["lat"], "lon": el["lon"]}
        for el in elements
    ]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def osrm_walking_minutes(
    lat1: float, lon1: float, lat2: float, lon2: float, osrm_url: str = "http://localhost:5000"
) -> float:
    url = f"{osrm_url}/route/v1/foot/{lon1},{lat1};{lon2},{lat2}"
    response = requests.get(url, params={"overview": "false"}, timeout=10)
    response.raise_for_status()
    data = response.json()
    if data.get("code") != "Ok":
        raise ValueError(f"OSRM route failed: {data.get('code')}")
    return data["routes"][0]["duration"] / 60.0


def nearest_subway_station(
    lat: float,
    lon: float,
    stations: list[dict],
    *,
    candidates: int = 3,
    walking_fn: Callable[..., float] = osrm_walking_minutes,
    osrm_url: str = "http://localhost:5000",
) -> tuple[str, float] | None:
    if not stations:
        return None

    ranked = sorted(stations, key=lambda s: haversine_km(lat, lon, s["lat"], s["lon"]))[:candidates]

    best: tuple[str, float] | None = None
    for station in ranked:
        minutes = walking_fn(lat, lon, station["lat"], station["lon"], osrm_url=osrm_url)
        if best is None or minutes < best[1]:
            best = (station["name"], minutes)
    return best
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/enrich/test_poi.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/realestate/enrich/poi.py tests/enrich/test_poi.py
git commit -m "feat: compute nearest-subway walking distance via Overpass+OSRM"
```

---

### Task 8: Enrichment pipeline orchestrator

**Files:**
- Create: `src/realestate/enrich/pipeline.py`
- Test: `tests/enrich/test_pipeline.py`

**Interfaces:**
- Consumes: `RawListing` (Task 2), `normalize_price_eur`/`normalize_rooms`/`normalize_specs` (Task 5), `CachedGeocoder` (Task 6), `nearest_subway_station` (Task 7)
- Produces: `enrich_listing(raw: RawListing, *, geocoder: CachedGeocoder, stations: list[dict], nearest_station_fn=nearest_subway_station) -> EnrichedListing`

**Location precedence:** OLX's own `map.lat`/`map.lon` (already on `RawListing` as `map_lat`/`map_lon`
per Task 3) is used directly when present — no geocoding call at all, since every real listing
sampled during this project had map data. Nominatim geocoding via `CachedGeocoder` is only a
fallback for the rare/hypothetical case where a listing's JSON omits `map`. This matters for cost
and reliability, not just precision: it means the network-dependent, rate-limited geocoder path is
barely exercised in practice.

- [ ] **Step 1: Write the failing test**

```python
# tests/enrich/test_pipeline.py
from realestate.enrich.geocoder import CachedGeocoder, GeocodeResult
from realestate.enrich.pipeline import enrich_listing
from realestate.models import RawListing

RAW_WITH_MAP = RawListing(
    external_id="304473136",
    title="Vand apartament 2 camere TITAN",
    description="Direct proprietar, apartament 2 camere decomandat.",
    price_raw="100.000 €",
    address_raw="Bucuresti - Ilfov, Bucuresti, Sectorul 3",
    specs={"Suprafata utila": "48 m²", "Etaj": "3"},
    image_urls=["https://example.com/a.jpg"],
    source_file="304473136.html",
    map_lat=44.42,
    map_lon=26.1,
)

RAW_WITHOUT_MAP = RAW_WITH_MAP.model_copy(update={"map_lat": None, "map_lon": None})

STATIONS = [{"name": "Universitate", "lat": 44.4356, "lon": 26.1023}]


def test_enrich_listing_uses_map_coordinates_without_geocoding(tmp_path):
    def fail_if_called(address):
        raise AssertionError("geocoder should not be called when map_lat/map_lon are present")

    geocoder = CachedGeocoder(cache_path=tmp_path / "cache.json", geocode_fn=fail_if_called)

    def fake_nearest_station(lat, lon, stations, **kwargs):
        return ("Universitate", 6.5)

    enriched = enrich_listing(
        RAW_WITH_MAP, geocoder=geocoder, stations=STATIONS, nearest_station_fn=fake_nearest_station
    )

    assert enriched.price_eur == 100000.0
    assert enriched.rooms == 2
    assert enriched.built_area_sqm == 48.0
    assert enriched.floor_number == 3
    assert enriched.latitude == 44.42
    assert enriched.longitude == 26.1
    assert enriched.location_confidence == "ok"
    assert enriched.nearest_subway_station == "Universitate"
    assert enriched.subway_walking_minutes == 6.5


def test_enrich_listing_falls_back_to_geocoding_when_map_absent(tmp_path):
    geocoder = CachedGeocoder(
        cache_path=tmp_path / "cache.json",
        geocode_fn=lambda address: GeocodeResult(latitude=44.4325, longitude=26.1013),
    )

    def fake_nearest_station(lat, lon, stations, **kwargs):
        return ("Universitate", 6.5)

    enriched = enrich_listing(
        RAW_WITHOUT_MAP, geocoder=geocoder, stations=STATIONS, nearest_station_fn=fake_nearest_station
    )

    assert enriched.latitude == 44.4325
    assert enriched.location_confidence == "ok"
    assert enriched.nearest_subway_station == "Universitate"


def test_enrich_listing_flags_low_confidence_location_when_map_absent_and_geocode_fails(tmp_path):
    geocoder = CachedGeocoder(
        cache_path=tmp_path / "cache.json",
        geocode_fn=lambda address: None,
    )

    enriched = enrich_listing(
        RAW_WITHOUT_MAP, geocoder=geocoder, stations=STATIONS, nearest_station_fn=lambda *a, **k: None
    )

    assert enriched.latitude is None
    assert enriched.location_confidence == "low_confidence_location"
    assert enriched.nearest_subway_station is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/enrich/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'realestate.enrich.pipeline'`

- [ ] **Step 3: Write the implementation**

```python
# src/realestate/enrich/pipeline.py
from typing import Callable

from realestate.enrich.geocoder import CachedGeocoder
from realestate.enrich.normalizer import normalize_price_eur, normalize_rooms, normalize_specs
from realestate.enrich.poi import nearest_subway_station
from realestate.models import EnrichedListing, RawListing


def enrich_listing(
    raw: RawListing,
    *,
    geocoder: CachedGeocoder,
    stations: list[dict],
    nearest_station_fn: Callable[..., tuple[str, float] | None] = nearest_subway_station,
) -> EnrichedListing:
    specs = normalize_specs(raw.specs)

    if raw.map_lat is not None and raw.map_lon is not None:
        latitude, longitude = raw.map_lat, raw.map_lon
    else:
        geocode_result = geocoder.geocode(raw.address_raw)
        latitude = geocode_result.latitude if geocode_result else None
        longitude = geocode_result.longitude if geocode_result else None

    location_confidence = "ok" if latitude is not None else "low_confidence_location"

    station_name: str | None = None
    walking_minutes: float | None = None
    if latitude is not None:
        nearest = nearest_station_fn(latitude, longitude, stations)
        if nearest is not None:
            station_name, walking_minutes = nearest

    return EnrichedListing(
        external_id=raw.external_id,
        title=raw.title,
        description=raw.description,
        price_eur=normalize_price_eur(raw.price_raw),
        rooms=normalize_rooms(raw.title),
        built_area_sqm=specs.get("built_area_sqm"),
        floor_number=specs.get("floor_number"),
        construction_year_range=specs.get("construction_year_range"),
        layout_type=specs.get("layout_type"),
        address_text=raw.address_raw,
        latitude=latitude,
        longitude=longitude,
        location_confidence=location_confidence,
        nearest_subway_station=station_name,
        subway_walking_minutes=walking_minutes,
        image_urls=raw.image_urls,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/enrich/test_pipeline.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/realestate/enrich/pipeline.py tests/enrich/test_pipeline.py
git commit -m "feat: orchestrate normalization+geocoding+POI into EnrichedListing"
```

---

### Task 9: Embedder (multilingual-e5, MPS)

**Files:**
- Create: `src/realestate/embed/__init__.py`
- Create: `src/realestate/embed/base.py`
- Create: `src/realestate/embed/sentence_embedder.py`
- Test: `tests/embed/test_sentence_embedder.py`

**Interfaces:**
- Produces: `Embedder` protocol with `.embed_passage(text: str) -> list[float]` and `.embed_query(text: str) -> list[float]`
- Produces: `MultilingualE5Embedder(model_name="intfloat/multilingual-e5-base", device="mps")` implementing `Embedder`

- [ ] **Step 1: Write `base.py` (the interface)**

```python
# src/realestate/embed/base.py
from typing import Protocol


class Embedder(Protocol):
    def embed_passage(self, text: str) -> list[float]: ...
    def embed_query(self, text: str) -> list[float]: ...
```

- [ ] **Step 2: Write the failing test**

This test downloads the real model on first run (network required once, then cached by
`sentence-transformers` locally) and runs actual inference — there's no meaningful fake for "does
this produce a usable embedding," so it's an integration test, not mocked.

```python
# tests/embed/test_sentence_embedder.py
from realestate.embed.sentence_embedder import MultilingualE5Embedder


def test_embed_passage_and_query_return_same_dimensionality():
    embedder = MultilingualE5Embedder(device="cpu")  # cpu here so CI/non-Mac runs work too
    passage_vec = embedder.embed_passage("Garsoniera luminoasa in Centrul Istoric")
    query_vec = embedder.embed_query("apartament luminos in centru")

    assert len(passage_vec) == len(query_vec)
    assert len(passage_vec) == 768  # multilingual-e5-base hidden size


def test_similar_texts_have_higher_cosine_similarity_than_dissimilar():
    import numpy as np

    embedder = MultilingualE5Embedder(device="cpu")
    query_vec = np.array(embedder.embed_query("apartament luminos cu parchet"))
    similar_vec = np.array(embedder.embed_passage("Apartament luminos, parchet masiv, mult soare"))
    dissimilar_vec = np.array(embedder.embed_passage("Teren agricol de vanzare in Ilfov"))

    similar_score = query_vec @ similar_vec
    dissimilar_score = query_vec @ dissimilar_vec
    assert similar_score > dissimilar_score
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/embed/test_sentence_embedder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'realestate.embed.sentence_embedder'`

- [ ] **Step 4: Write the implementation**

```python
# src/realestate/embed/sentence_embedder.py
from sentence_transformers import SentenceTransformer


class MultilingualE5Embedder:
    """Wraps intfloat/multilingual-e5-base. Note the model requires 'query: '/'passage: '
    prefixes on input text — this is part of how it was trained, not an arbitrary choice."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base", device: str = "mps"):
        self._model = SentenceTransformer(model_name, device=device)

    def embed_passage(self, text: str) -> list[float]:
        return self._model.encode(f"passage: {text}", normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(f"query: {text}", normalize_embeddings=True).tolist()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/embed/test_sentence_embedder.py -v`
Expected: PASS (2 passed) — first run will be slower while the model downloads (~1GB)

- [ ] **Step 6: Commit**

```bash
mkdir -p tests/embed
touch tests/embed/__init__.py
git add src/realestate/embed/ tests/embed/
git commit -m "feat: add multilingual-e5 embedder"
```

---

### Task 10: Qdrant vector store

**Files:**
- Create: `src/realestate/store/__init__.py`
- Create: `src/realestate/store/base.py`
- Create: `src/realestate/store/qdrant_store.py`
- Test: `tests/store/test_qdrant_store.py`

**Interfaces:**
- Produces: `VectorStore` protocol with `.upsert(external_id, vector, payload)` and `.query(vector, *, price_max=None, rooms=None, max_subway_minutes=None, limit=10) -> list`
- Produces: `QdrantListingStore(url="http://localhost:6333", collection="listings", vector_size=768)` implementing `VectorStore`

**Requires:** `docker compose up -d qdrant` running (from Task 1) before these tests can pass — this is an integration test against a real local Qdrant instance, not mocked, since the filter-plus-vector query behavior is exactly what needs verifying.

- [ ] **Step 1: Write `base.py` (the interface)**

```python
# src/realestate/store/base.py
from typing import Protocol


class VectorStore(Protocol):
    def upsert(self, external_id: str, vector: list[float], payload: dict) -> None: ...

    def query(
        self,
        vector: list[float],
        *,
        price_max: float | None = None,
        rooms: int | None = None,
        max_subway_minutes: float | None = None,
        limit: int = 10,
    ) -> list: ...
```

- [ ] **Step 2: Write the failing test**

```python
# tests/store/test_qdrant_store.py
import uuid
import pytest
from realestate.store.qdrant_store import QdrantListingStore


@pytest.fixture
def store():
    # Unique collection per test run so tests don't interfere with each other.
    return QdrantListingStore(collection=f"test_listings_{uuid.uuid4().hex[:8]}")


def test_upsert_and_query_returns_the_point(store):
    vector = [0.1] * 768
    store.upsert("111", vector, {"price_eur": 100000, "rooms": 1, "subway_walking_minutes": 5.0})

    results = store.query(vector, limit=5)

    assert len(results) == 1
    assert results[0].payload["external_id"] == "111"


def test_query_filters_out_listings_above_price_max(store):
    vector = [0.1] * 768
    store.upsert("cheap", vector, {"price_eur": 50000, "rooms": 1, "subway_walking_minutes": 5.0})
    store.upsert("expensive", vector, {"price_eur": 500000, "rooms": 1, "subway_walking_minutes": 5.0})

    results = store.query(vector, price_max=100000, limit=10)

    ids = [r.payload["external_id"] for r in results]
    assert "cheap" in ids
    assert "expensive" not in ids


def test_query_filters_by_max_subway_minutes(store):
    vector = [0.1] * 768
    store.upsert("close", vector, {"price_eur": 100000, "rooms": 1, "subway_walking_minutes": 5.0})
    store.upsert("far", vector, {"price_eur": 100000, "rooms": 1, "subway_walking_minutes": 40.0})

    results = store.query(vector, max_subway_minutes=15.0, limit=10)

    ids = [r.payload["external_id"] for r in results]
    assert "close" in ids
    assert "far" not in ids
```

- [ ] **Step 3: Run test to verify it fails**

Run: `docker compose up -d qdrant && pytest tests/store/test_qdrant_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'realestate.store.qdrant_store'`

- [ ] **Step 4: Write the implementation**

```python
# src/realestate/store/qdrant_store.py
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)


def _external_id_to_point_id(external_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, external_id))


class QdrantListingStore:
    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection: str = "listings",
        vector_size: int = 768,
    ):
        self._client = QdrantClient(url=url)
        self._collection = collection
        if not self._client.collection_exists(collection):
            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def upsert(self, external_id: str, vector: list[float], payload: dict) -> None:
        self._client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(
                    id=_external_id_to_point_id(external_id),
                    vector=vector,
                    payload={**payload, "external_id": external_id},
                )
            ],
        )

    def query(
        self,
        vector: list[float],
        *,
        price_max: float | None = None,
        rooms: int | None = None,
        max_subway_minutes: float | None = None,
        limit: int = 10,
    ) -> list:
        must = []
        if price_max is not None:
            must.append(FieldCondition(key="price_eur", range=Range(lte=price_max)))
        if rooms is not None:
            must.append(FieldCondition(key="rooms", match=MatchValue(value=rooms)))
        if max_subway_minutes is not None:
            must.append(
                FieldCondition(key="subway_walking_minutes", range=Range(lte=max_subway_minutes))
            )
        query_filter = Filter(must=must) if must else None

        response = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            query_filter=query_filter,
            limit=limit,
        )
        return response.points
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/store/test_qdrant_store.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
mkdir -p tests/store
touch tests/store/__init__.py
git add src/realestate/store/ tests/store/
git commit -m "feat: add Qdrant-backed vector store with structured filtering"
```

---

### Task 11: End-to-end ingestion orchestrator

**Files:**
- Create: `src/realestate/ingestion_pipeline.py`
- Create: `scripts/run_ingestion.py`
- Test: `tests/test_ingestion_pipeline.py`

**Interfaces:**
- Consumes: `load_olx_html_directory` (Task 4), `enrich_listing` (Task 8), `Embedder` (Task 9), `VectorStore` (Task 10)
- Produces: `run_ingestion(html_dir: Path, *, geocoder, stations, embedder: Embedder, store: VectorStore) -> IngestionReport` where `IngestionReport` has `.succeeded: int`, `.parse_failures: list[tuple[Path, str]]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingestion_pipeline.py
import copy
from pathlib import Path

from realestate.enrich.geocoder import CachedGeocoder, GeocodeResult
from realestate.ingestion_pipeline import run_ingestion
from tests.ingest.test_olx_parser import SAMPLE_AD, SAMPLE_HTML, _build_sample_html


class FakeEmbedder:
    def embed_passage(self, text: str) -> list[float]:
        return [0.1] * 768

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 768


class FakeStore:
    def __init__(self):
        self.upserted: list[tuple[str, list[float], dict]] = []

    def upsert(self, external_id, vector, payload):
        self.upserted.append((external_id, vector, payload))

    def query(self, *args, **kwargs):
        raise NotImplementedError("not exercised in this test")


def test_run_ingestion_embeds_and_stores_valid_listings(tmp_path: Path):
    (tmp_path / "304473136.html").write_text(SAMPLE_HTML, encoding="utf-8")
    (tmp_path / "bad.html").write_text("<html></html>", encoding="utf-8")

    geocoder = CachedGeocoder(
        cache_path=tmp_path / "cache.json",
        geocode_fn=lambda address: GeocodeResult(latitude=44.43, longitude=26.10),
    )
    store = FakeStore()

    report = run_ingestion(
        tmp_path,
        geocoder=geocoder,
        stations=[],
        embedder=FakeEmbedder(),
        store=store,
    )

    assert report.succeeded == 1
    assert len(report.parse_failures) == 1
    assert store.upserted[0][0] == "304473136"
    assert store.upserted[0][2]["price_eur"] == 100000.0


def test_run_ingestion_quarantines_enrich_failures_without_aborting_batch(tmp_path: Path):
    # A listing that parses fine (well-formed PRERENDERED_STATE) but whose price is not in
    # the expected "<digits> €" format - mirrors real OLX listings priced "La cerere" (on
    # request) that fail normalize_price_eur's regex during enrichment, not during parsing.
    bad_price_ad = copy.deepcopy(SAMPLE_AD)
    bad_price_ad["id"] = 999999999
    bad_price_ad["price"]["regularPrice"] = {
        "value": 100000,
        "currencyCode": "EUR",
        "currencySymbol": "La cerere",
    }

    (tmp_path / "304473136.html").write_text(SAMPLE_HTML, encoding="utf-8")
    (tmp_path / "999999999.html").write_text(_build_sample_html(bad_price_ad), encoding="utf-8")

    geocoder = CachedGeocoder(
        cache_path=tmp_path / "cache.json",
        geocode_fn=lambda address: GeocodeResult(latitude=44.43, longitude=26.10),
    )
    store = FakeStore()

    report = run_ingestion(
        tmp_path,
        geocoder=geocoder,
        stations=[],
        embedder=FakeEmbedder(),
        store=store,
    )

    assert report.succeeded == 1
    assert len(report.enrich_failures) == 1
    assert report.enrich_failures[0][0] == "999999999"
    assert store.upserted[0][0] == "304473136"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingestion_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'realestate.ingestion_pipeline'`

- [ ] **Step 3: Write the implementation**

```python
# src/realestate/ingestion_pipeline.py
from dataclasses import dataclass, field
from pathlib import Path

from realestate.embed.base import Embedder
from realestate.enrich.geocoder import CachedGeocoder
from realestate.enrich.pipeline import enrich_listing
from realestate.ingest.olx_loader import load_olx_html_directory
from realestate.store.base import VectorStore


@dataclass
class IngestionReport:
    succeeded: int = 0
    parse_failures: list[tuple[Path, str]] = field(default_factory=list)
    enrich_failures: list[tuple[str, str]] = field(default_factory=list)


def run_ingestion(
    html_dir: Path,
    *,
    geocoder: CachedGeocoder,
    stations: list[dict],
    embedder: Embedder,
    store: VectorStore,
) -> IngestionReport:
    raw_listings, parse_failures = load_olx_html_directory(html_dir)

    succeeded = 0
    enrich_failures: list[tuple[str, str]] = []
    for raw in raw_listings:
        try:
            enriched = enrich_listing(raw, geocoder=geocoder, stations=stations)
            vector = embedder.embed_passage(enriched.description)
            store.upsert(
                enriched.external_id,
                vector,
                payload=enriched.model_dump(),
            )
            succeeded += 1
        except Exception as exc:  # noqa: BLE001 - quarantine, don't crash the batch
            enrich_failures.append((raw.external_id, str(exc)))

    return IngestionReport(
        succeeded=succeeded, parse_failures=parse_failures, enrich_failures=enrich_failures
    )
```

Note: a per-listing failure here (e.g. a price string like "La cerere"/"on request" that
`normalize_price_eur` can't parse) is quarantined into `enrich_failures` rather than aborting the
whole batch — mirroring `load_olx_html_directory`'s quarantine pattern for parse failures. This was
added after review flagged that the original brief's bare loop would crash an entire ~100-listing
ingestion run on one malformed real listing.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ingestion_pipeline.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Write the real CLI entry point**

This scrapes fresh listings (skipping any already downloaded, per Task 4's idempotency) and then
runs them through the full pipeline in one command.

```python
# scripts/run_ingestion.py
"""Scrape OLX.ro listings (if needed) and run the full ingestion pipeline.

Usage: python scripts/run_ingestion.py 100
"""

import sys
from pathlib import Path

from realestate.embed.sentence_embedder import MultilingualE5Embedder
from realestate.enrich.geocoder import CachedGeocoder
from realestate.enrich.poi import fetch_bucharest_subway_stations
from realestate.ingest.olx_scraper import download_listings
from realestate.ingestion_pipeline import run_ingestion
from realestate.store.qdrant_store import QdrantListingStore

if __name__ == "__main__":
    target_count = int(sys.argv[1])
    html_dir = Path("data/raw/olx_html")

    newly_downloaded = download_listings(target_count, html_dir)
    print(f"Downloaded {len(newly_downloaded)} new listings (target: {target_count} total on disk).")

    geocoder = CachedGeocoder(cache_path=Path("data/cache/geocode_cache.json"))
    stations = fetch_bucharest_subway_stations()
    embedder = MultilingualE5Embedder(device="mps")
    store = QdrantListingStore()

    report = run_ingestion(
        html_dir, geocoder=geocoder, stations=stations, embedder=embedder, store=store
    )

    print(f"Ingested {report.succeeded} listings.")
    if report.parse_failures:
        print(f"{len(report.parse_failures)} files failed to parse:")
        for path, error in report.parse_failures:
            print(f"  {path}: {error}")
    if report.enrich_failures:
        print(f"{len(report.enrich_failures)} listings failed to enrich/embed/store:")
        for external_id, error in report.enrich_failures:
            print(f"  {external_id}: {error}")
```

- [ ] **Step 6: Commit**

```bash
mkdir -p scripts
git add src/realestate/ingestion_pipeline.py scripts/run_ingestion.py tests/test_ingestion_pipeline.py
git commit -m "feat: wire ingestion pipeline end-to-end with a CLI entry point"
```

---

### Task 12: Local LLM query parser (Ollama)

**Files:**
- Create: `src/realestate/query/__init__.py`
- Create: `src/realestate/query/parser.py`
- Test: `tests/query/test_parser.py`

**Interfaces:**
- Produces: `ParsedQuery(filters: dict, semantic_text: str)` (pydantic model; `filters` may contain `price_max`, `rooms`, `max_subway_minutes`, any subset)
- Produces: `OllamaQueryParser(model="qwen2.5:14b", host="http://localhost:11434")` with `.parse(query: str) -> ParsedQuery`. On any parse failure (unreachable Ollama, invalid JSON, schema mismatch), falls back to `ParsedQuery(filters={}, semantic_text=query)` per the spec's error-handling rule — never raises.

- [ ] **Step 1: Write the failing test**

The LLM call itself isn't unit tested (that's the live-Ollama smoke test in Step 6) — the parsing
and fallback *logic* is tested by injecting a fake HTTP call.

```python
# tests/query/test_parser.py
from realestate.query.parser import OllamaQueryParser, ParsedQuery


class FakeResponse:
    def __init__(self, json_body):
        self._json_body = json_body

    def raise_for_status(self):
        pass

    def json(self):
        return self._json_body


def test_parse_extracts_filters_and_semantic_text():
    def fake_post(url, json, timeout):
        return FakeResponse(
            {
                "response": (
                    '{"filters": {"price_max": 150000, "max_subway_minutes": 15}, '
                    '"semantic_text": "bright pet-friendly hardwood floors"}'
                )
            }
        )

    parser = OllamaQueryParser(post_fn=fake_post)
    result = parser.parse(
        "bright pet-friendly apartment with hardwood floors, "
        "max 150000 EUR, max 15 min walk to subway"
    )

    assert result.filters["price_max"] == 150000
    assert result.filters["max_subway_minutes"] == 15
    assert result.semantic_text == "bright pet-friendly hardwood floors"


def test_parse_falls_back_to_pure_semantic_text_on_invalid_json():
    def fake_post(url, json, timeout):
        return FakeResponse({"response": "not valid json"})

    parser = OllamaQueryParser(post_fn=fake_post)
    result = parser.parse("some query")

    assert result.filters == {}
    assert result.semantic_text == "some query"


def test_parse_falls_back_when_request_raises():
    def fake_post(url, json, timeout):
        raise ConnectionError("Ollama not running")

    parser = OllamaQueryParser(post_fn=fake_post)
    result = parser.parse("some query")

    assert result.filters == {}
    assert result.semantic_text == "some query"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/query/test_parser.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'realestate.query.parser'`

- [ ] **Step 3: Write the implementation**

```python
# src/realestate/query/parser.py
import json
from typing import Callable

import requests
from pydantic import BaseModel, ValidationError


class ParsedQuery(BaseModel):
    filters: dict
    semantic_text: str


_PARSE_PROMPT = """Extract structured filters and the remaining semantic intent from this \
real estate search query. Respond with ONLY a JSON object of the form:
{{"filters": {{"price_max": <number or omit>, "rooms": <number or omit>, \
"max_subway_minutes": <number or omit>}}, "semantic_text": "<remaining descriptive text>"}}

Query: {query}
"""


def _default_post(url: str, json: dict, timeout: int):
    return requests.post(url, json=json, timeout=timeout)


class OllamaQueryParser:
    def __init__(
        self,
        model: str = "qwen2.5:14b",
        host: str = "http://localhost:11434",
        post_fn: Callable = _default_post,
    ):
        self._model = model
        self._host = host
        self._post_fn = post_fn

    def parse(self, query: str) -> ParsedQuery:
        try:
            response = self._post_fn(
                f"{self._host}/api/generate",
                json={
                    "model": self._model,
                    "prompt": _PARSE_PROMPT.format(query=query),
                    "format": "json",
                    "stream": False,
                },
                timeout=30,
            )
            response.raise_for_status()
            raw = json.loads(response.json()["response"])
            return ParsedQuery(**raw)
        except (requests.RequestException, ConnectionError, json.JSONDecodeError, ValidationError, KeyError):
            return ParsedQuery(filters={}, semantic_text=query)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/query/test_parser.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
mkdir -p tests/query
touch tests/query/__init__.py
git add src/realestate/query/ tests/query/test_parser.py
git commit -m "feat: add local LLM query parser with safe fallback"
```

- [ ] **Step 6: Manual smoke test against real Ollama**

Run: `ollama pull qwen2.5:14b` then, in a Python shell:
```python
from realestate.query.parser import OllamaQueryParser
print(OllamaQueryParser().parse("bright pet-friendly apartment, max 150000 EUR"))
```
Expected: a `ParsedQuery` with a populated `price_max` filter — if the model doesn't reliably
produce valid JSON, adjust `_PARSE_PROMPT` wording (this is expected prompt-iteration, not a bug).

---

### Task 13: Retrieval orchestrator

**Files:**
- Create: `src/realestate/query/retrieval.py`
- Test: `tests/query/test_retrieval.py`

**Interfaces:**
- Consumes: `OllamaQueryParser`/`ParsedQuery` (Task 12), `Embedder` (Task 9), `VectorStore` (Task 10)
- Produces: `search(query: str, *, parser, embedder: Embedder, store: VectorStore, limit: int = 10) -> list`

- [ ] **Step 1: Write the failing test**

```python
# tests/query/test_retrieval.py
from realestate.query.parser import ParsedQuery
from realestate.query.retrieval import search


class FakeParser:
    def __init__(self, parsed: ParsedQuery):
        self._parsed = parsed

    def parse(self, query: str) -> ParsedQuery:
        return self._parsed


class FakeEmbedder:
    def embed_query(self, text: str) -> list[float]:
        assert text == "bright hardwood floors"
        return [0.2] * 768


class FakeStore:
    def __init__(self):
        self.last_call_kwargs = None

    def query(self, vector, **kwargs):
        self.last_call_kwargs = kwargs
        return ["fake-result"]


def test_search_passes_parsed_filters_and_embedded_semantic_text_to_store():
    parsed = ParsedQuery(
        filters={"price_max": 150000, "max_subway_minutes": 15},
        semantic_text="bright hardwood floors",
    )
    store = FakeStore()

    results = search(
        "bright apartment with hardwood floors, max 150000 EUR, 15 min to subway",
        parser=FakeParser(parsed),
        embedder=FakeEmbedder(),
        store=store,
        limit=5,
    )

    assert results == ["fake-result"]
    assert store.last_call_kwargs == {
        "price_max": 150000,
        "rooms": None,
        "max_subway_minutes": 15,
        "limit": 5,
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/query/test_retrieval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'realestate.query.retrieval'`

- [ ] **Step 3: Write the implementation**

```python
# src/realestate/query/retrieval.py
from realestate.embed.base import Embedder
from realestate.store.base import VectorStore


def search(
    query: str,
    *,
    parser,
    embedder: Embedder,
    store: VectorStore,
    limit: int = 10,
) -> list:
    parsed = parser.parse(query)
    vector = embedder.embed_query(parsed.semantic_text)
    return store.query(
        vector,
        price_max=parsed.filters.get("price_max"),
        rooms=parsed.filters.get("rooms"),
        max_subway_minutes=parsed.filters.get("max_subway_minutes"),
        limit=limit,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/query/test_retrieval.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/realestate/query/retrieval.py tests/query/test_retrieval.py
git commit -m "feat: orchestrate hybrid retrieval (parsed filters + semantic vector search)"
```

---

### Task 14: Eval — reverse query generation

**Files:**
- Create: `src/realestate/eval/__init__.py`
- Create: `src/realestate/eval/generate_queries.py`
- Test: `tests/eval/test_generate_queries.py`

**Interfaces:**
- Consumes: `EnrichedListing` (Task 2)
- Produces: `EvalPair(query: str, relevant_listing_id: str)`
- Produces: `generate_eval_query(listing: EnrichedListing, *, generate_fn: Callable[[str], str]) -> EvalPair` — `generate_fn` is the LLM call, injected so tests don't need a live model

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_generate_queries.py
from realestate.eval.generate_queries import generate_eval_query
from realestate.models import EnrichedListing

LISTING = EnrichedListing(
    external_id="304473136",
    title="Vand apartament 2 camere TITAN",
    description="Apartament luminos, decomandat, in Titan.",
    price_eur=100000.0,
    rooms=2,
    built_area_sqm=48.0,
    floor_number=3,
    construction_year_range="1977 – 1990",
    layout_type="Decomandat",
    address_text="Bucuresti - Ilfov, Bucuresti, Sectorul 3",
    latitude=44.43,
    longitude=26.10,
    location_confidence="ok",
    nearest_subway_station="Titan",
    subway_walking_minutes=6.5,
    image_urls=[],
)


def test_generate_eval_query_uses_listing_details_in_the_prompt():
    captured_prompts = []

    def fake_generate(prompt: str) -> str:
        captured_prompts.append(prompt)
        return "Bright 2-room apartment near Titan subway station"

    pair = generate_eval_query(LISTING, generate_fn=fake_generate)

    assert pair.relevant_listing_id == "304473136"
    assert pair.query == "Bright 2-room apartment near Titan subway station"
    assert "Sectorul 3" in captured_prompts[0]
    assert "Titan" in captured_prompts[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_generate_queries.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'realestate.eval.generate_queries'`

- [ ] **Step 3: Write the implementation**

```python
# src/realestate/eval/generate_queries.py
from dataclasses import dataclass
from typing import Callable

from realestate.models import EnrichedListing

_GENERATE_PROMPT = """You are simulating a real estate search user. Given this listing, \
write ONE natural-language search query that a real user might type to find it. Do not \
mention the price or the exact address, but you may reference the neighborhood or nearby \
transit if relevant. Respond with only the query text.

Title: {title}
Description: {description}
Neighborhood: {address_text}
Nearest subway: {nearest_subway_station} ({subway_walking_minutes} min walk)
"""


@dataclass
class EvalPair:
    query: str
    relevant_listing_id: str


def generate_eval_query(
    listing: EnrichedListing, *, generate_fn: Callable[[str], str]
) -> EvalPair:
    prompt = _GENERATE_PROMPT.format(
        title=listing.title,
        description=listing.description,
        address_text=listing.address_text,
        nearest_subway_station=listing.nearest_subway_station,
        subway_walking_minutes=listing.subway_walking_minutes,
    )
    query = generate_fn(prompt)
    return EvalPair(query=query, relevant_listing_id=listing.external_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_generate_queries.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
mkdir -p tests/eval
touch tests/eval/__init__.py
git add src/realestate/eval/generate_queries.py tests/eval/test_generate_queries.py
git commit -m "feat: generate eval queries from listings via reverse generation"
```

---

### Task 15: Eval — IR metrics and run harness

**Files:**
- Create: `src/realestate/eval/metrics.py`
- Create: `src/realestate/eval/run_eval.py`
- Test: `tests/eval/test_metrics.py`

**Interfaces:**
- Consumes: `EvalPair` (Task 14)
- Produces: `recall_at_k(retrieved_ids: list[str], relevant_id: str, k: int) -> float`
- Produces: `ndcg_at_k(retrieved_ids: list[str], relevant_id: str, k: int) -> float`
- Produces: `mean_reciprocal_rank(retrieved_ids: list[str], relevant_id: str) -> float`
- Produces: `evaluate(pairs: list[EvalPair], search_fn: Callable[[str], list[str]], k: int = 10) -> dict[str, float]` (keys: `recall_at_k`, `ndcg_at_k`, `mrr`, each averaged across `pairs`)

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_metrics.py
import math
from realestate.eval.generate_queries import EvalPair
from realestate.eval.metrics import evaluate, mean_reciprocal_rank, ndcg_at_k, recall_at_k


def test_recall_at_k_hit():
    assert recall_at_k(["a", "b", "c"], relevant_id="b", k=3) == 1.0


def test_recall_at_k_miss_outside_k():
    assert recall_at_k(["a", "b", "c"], relevant_id="d", k=3) == 0.0


def test_ndcg_at_k_rewards_higher_rank():
    top_rank = ndcg_at_k(["target", "b", "c"], relevant_id="target", k=3)
    bottom_rank = ndcg_at_k(["a", "b", "target"], relevant_id="target", k=3)
    assert top_rank == 1.0
    assert 0 < bottom_rank < top_rank


def test_mean_reciprocal_rank_first_position():
    assert mean_reciprocal_rank(["target", "b"], relevant_id="target") == 1.0


def test_mean_reciprocal_rank_second_position():
    assert mean_reciprocal_rank(["a", "target"], relevant_id="target") == 0.5


def test_mean_reciprocal_rank_not_found():
    assert mean_reciprocal_rank(["a", "b"], relevant_id="target") == 0.0


def test_evaluate_averages_metrics_across_pairs():
    pairs = [
        EvalPair(query="q1", relevant_listing_id="1"),
        EvalPair(query="q2", relevant_listing_id="2"),
    ]

    def fake_search(query: str) -> list[str]:
        return {"q1": ["1", "x", "y"], "q2": ["x", "2", "y"]}[query]

    scores = evaluate(pairs, search_fn=fake_search, k=3)

    assert scores["recall_at_k"] == 1.0
    assert 0.0 < scores["mrr"] < 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'realestate.eval.metrics'`

- [ ] **Step 3: Write the implementation**

```python
# src/realestate/eval/metrics.py
import math
from typing import Callable

from realestate.eval.generate_queries import EvalPair


def recall_at_k(retrieved_ids: list[str], relevant_id: str, k: int) -> float:
    return 1.0 if relevant_id in retrieved_ids[:k] else 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_id: str, k: int) -> float:
    top_k = retrieved_ids[:k]
    if relevant_id not in top_k:
        return 0.0
    rank = top_k.index(relevant_id) + 1  # 1-indexed
    return 1.0 / math.log2(rank + 1)


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_id: str) -> float:
    if relevant_id not in retrieved_ids:
        return 0.0
    return 1.0 / (retrieved_ids.index(relevant_id) + 1)


def evaluate(
    pairs: list[EvalPair], search_fn: Callable[[str], list[str]], k: int = 10
) -> dict[str, float]:
    recalls, ndcgs, rrs = [], [], []
    for pair in pairs:
        retrieved = search_fn(pair.query)
        recalls.append(recall_at_k(retrieved, pair.relevant_listing_id, k))
        ndcgs.append(ndcg_at_k(retrieved, pair.relevant_listing_id, k))
        rrs.append(mean_reciprocal_rank(retrieved, pair.relevant_listing_id))

    n = len(pairs)
    return {
        "recall_at_k": sum(recalls) / n,
        "ndcg_at_k": sum(ndcgs) / n,
        "mrr": sum(rrs) / n,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_metrics.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Write the real run harness**

```python
# src/realestate/eval/run_eval.py
"""Build an eval set from currently ingested listings and score the live retrieval pipeline.

Usage: python -m realestate.eval.run_eval
"""

from realestate.embed.sentence_embedder import MultilingualE5Embedder
from realestate.eval.generate_queries import generate_eval_query
from realestate.eval.metrics import evaluate
from realestate.query.parser import OllamaQueryParser
from realestate.query.retrieval import search
from realestate.store.qdrant_store import QdrantListingStore

if __name__ == "__main__":
    store = QdrantListingStore()
    embedder = MultilingualE5Embedder(device="mps")
    parser = OllamaQueryParser()

    # Eval pairs are built from whatever is currently ingested, by scrolling the raw
    # points back out of Qdrant (payload was stored as the full EnrichedListing on upsert).
    all_points = store._client.scroll(collection_name="listings", limit=1000)[0]

    def ollama_generate(prompt: str) -> str:
        import requests

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "qwen2.5:14b", "prompt": prompt, "stream": False},
            timeout=30,
        )
        return response.json()["response"].strip()

    pairs = []
    for point in all_points:
        payload = point.payload
        from realestate.models import EnrichedListing

        listing = EnrichedListing(**{k: v for k, v in payload.items() if k != "external_id"} | {"external_id": payload["external_id"]})
        pairs.append(generate_eval_query(listing, generate_fn=ollama_generate))

    def search_fn(query: str) -> list[str]:
        results = search(query, parser=parser, embedder=embedder, store=store, limit=10)
        return [r.payload["external_id"] for r in results]

    scores = evaluate(pairs, search_fn=search_fn, k=10)
    print(scores)
```

- [ ] **Step 6: Commit**

```bash
git add src/realestate/eval/metrics.py src/realestate/eval/run_eval.py tests/eval/test_metrics.py
git commit -m "feat: add IR evaluation metrics and end-to-end eval harness"
```

---

## Self-Review Notes

- **Spec coverage:** ingestion (Tasks 3-4), enrichment/normalization/geocoding/POI (Tasks 5-8),
  embedding (Task 9), storage (Task 10), end-to-end pipeline (Task 11), hybrid query parsing +
  retrieval (Tasks 12-13), evaluation harness (Tasks 14-15) — all spec sections have a
  corresponding task. UI, fraud scoring, and RL recommendations are explicitly out of scope per
  the spec's Future Phases Backlog, and are not tasked here.
- **Data source revision:** the spec originally scoped imobiliare.ro/storia.ro scraping; both were
  found to run an active Cloudflare managed challenge on real requests (verified empirically, not
  assumed), so Tasks 3-4 and the CLI in Task 11 were rewritten around OLX.ro instead — confirmed
  scrapable (robots.txt allows it, plain 200 responses, clean embedded JSON per listing). All
  downstream field names (`EnrichedListing`, `SPEC_LABEL_MAP`, eval fixtures) were updated to match
  OLX's actual schema rather than imobiliare's, and cross-checked for staleness across Tasks 2, 5,
  8, 11, and 14.
- **Type consistency:** `Embedder`/`VectorStore` protocol method signatures (Tasks 9, 10) match
  their usage in `ingestion_pipeline.py` (Task 11) and `retrieval.py` (Task 13). `EnrichedListing`
  fields (Task 2: `floor_number`, `construction_year_range`, `layout_type`) match what
  `enrich_listing` (Task 8) constructs and what `generate_eval_query` (Task 14) reads.
- **Known follow-up baked into Task 4:** OLX's `params` schema was verified against one apartment
  listing; other property types (houses, land) may use different param keys, flagged inline as a
  spot-check to do once real listings are downloaded, rather than assumed to generalize.

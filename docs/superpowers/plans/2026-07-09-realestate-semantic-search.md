# Real Estate Semantic Search (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 1 foundation from the design spec — ingest real Bucharest listings (from manually collected imobiliare.ro pages), enrich them with geocoding and subway-walking-distance, embed and store them in a vector DB, and answer natural-language queries via hybrid (structured filter + geospatial + dense vector) retrieval, with an offline evaluation harness.

**Architecture:** A modular-monolith Python package (`realestate`) with five interface-bound layers — `ingest`, `enrich`, `embed`, `store`, `query` — plus an `eval` layer for offline IR metrics. Each layer depends only on the abstractions of its neighbors (see Dependency Inversion in the spec), so components are swappable and independently testable.

**Tech Stack:** Python 3.11, pydantic (schemas), BeautifulSoup4 + lxml (HTML parsing), requests (Nominatim/Overpass/OSRM/Ollama HTTP calls), sentence-transformers + PyTorch/MPS (embeddings, `intfloat/multilingual-e5-base`), Qdrant (self-hosted via Docker) + qdrant-client, Ollama (local LLM, query parsing), pytest.

## Global Constraints

- No automated network scraping of imobiliare.ro/storia.ro — listing HTML is manually collected by the user and read from local files only (see spec: both sites actively block automated fetches; circumventing that is out of scope).
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
pythonpath = ["src"]

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
- Produces: `RawListing(external_id: str, title: str, description: str, price_raw: str, address_raw: str, specs: dict[str, str], image_urls: list[str], source_file: str)`
- Produces: `EnrichedListing(external_id: str, title: str, description: str, price_eur: float, rooms: int | None, built_area_sqm: float | None, bathrooms: int | None, construction_material: str | None, building_type: str | None, floor_regime: str | None, address_text: str, latitude: float | None, longitude: float | None, location_confidence: str, nearest_subway_station: str | None, subway_walking_minutes: float | None, image_urls: list[str])`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
from realestate.models import RawListing, EnrichedListing


def test_raw_listing_construction():
    listing = RawListing(
        external_id="275727717",
        title="Garsoniera 40mp in Centrul Istoric",
        description="Vanzare garsoniera...",
        price_raw="135.000 €",
        address_raw="Centrul Istoric, Bucuresti",
        specs={"Suprafata construita:": "50 mp"},
        image_urls=["https://i.roamcdn.net/prop/imo/example.jpg"],
        source_file="275727717.html",
    )
    assert listing.external_id == "275727717"
    assert listing.specs["Suprafata construita:"] == "50 mp"


def test_enriched_listing_defaults_for_missing_location():
    listing = EnrichedListing(
        external_id="275727717",
        title="Garsoniera 40mp",
        description="Vanzare garsoniera...",
        price_eur=135000.0,
        rooms=1,
        built_area_sqm=50.0,
        bathrooms=1,
        construction_material="Caramida",
        building_type="Bloc de apartamente",
        floor_regime="P+2E",
        address_text="Centrul Istoric, Bucuresti",
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


class EnrichedListing(BaseModel):
    external_id: str
    title: str
    description: str
    price_eur: float
    rooms: int | None
    built_area_sqm: float | None
    bathrooms: int | None
    construction_material: str | None
    building_type: str | None
    floor_regime: str | None
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

### Task 3: Imobiliare.ro HTML parser

This is grounded in real HTML fragments captured directly from a live imobiliare.ro listing page
(`https://www.imobiliare.ro/oferta/garsoniera-de-vanzare-ultracentral-centrul-istoric-40mp-275727717`),
not guessed selectors.

**Files:**
- Create: `src/realestate/ingest/__init__.py`
- Create: `src/realestate/ingest/imobiliare_parser.py`
- Test: `tests/ingest/test_imobiliare_parser.py`

**Interfaces:**
- Consumes: `RawListing` from `realestate.models`
- Produces: `parse_imobiliare_html(html: str, external_id: str) -> RawListing`

- [ ] **Step 1: Write the failing test using a real captured fixture**

```python
# tests/ingest/test_imobiliare_parser.py
import pytest
from realestate.ingest.imobiliare_parser import parse_imobiliare_html

# This fixture assembles real HTML fragments captured from a live imobiliare.ro
# listing page — not synthesized/guessed markup.
SAMPLE_HTML = """
<html><body>
<img class="relative h-full w-full object-contain"
     src="https://i.roamcdn.net/prop/imo/gallery-main-900w-watermark/example1.jpg"
     title="Garsoniera 40mp in Centrul Istoric - Str. Selari nr. 2, imobil fara risc seismic"
     alt="Garsoniera 40mp in Centrul Istoric" width="375" height="270">
<img class="relative h-full w-full object-contain"
     src="https://i.roamcdn.net/prop/imo/gallery-main-900w-watermark/example2.jpg"
     title="Garsoniera 40mp in Centrul Istoric - Str. Selari nr. 2, imobil fara risc seismic"
     alt="Garsoniera 40mp in Centrul Istoric" width="375" height="270">

<div class="mb-1 flex flex-nowrap items-end space-x-1 md:mb-0" aria-label="price">
  <span class="text-xl font-semibold leading-none md:text-xxl md:font-extrabold">135.000 &euro;</span>
</div>

<p class="mb-2 flex items-center text-sm text-gray-500 md:hidden" data-cy="listing-address">
    Centrul Istoric, Bucuresti
</p>

<div class="readable clamped text-content mb-0 whitespace-pre-line text-justify md:mb-2"
     id="truncatedDescription">
    Va propunem spre vanzare o garsoniera situata in Centrul Istoric.

    Caracteristici:
    Suprafata utila: 40 mp
</div>

<section class="listing-specifications-component px-3 py-3" data-cy="basic-info-section">
  <div class="w-full">
    <div class="grid w-full grid-cols-1 gap-x-2 md:grid-cols-2">
      <div class="flex w-full justify-between gap-x-2 border-b border-gray-200 py-3">
        <span class="flex shrink-0 text-sm text-grey-550">Suprafata construita:</span>
        <span class="flex text-sm font-bold text-grey-550">50 mp</span>
      </div>
      <div class="flex w-full justify-between gap-x-2 border-b border-gray-200 py-3">
        <span class="flex shrink-0 text-sm text-grey-550">Nr. bai:</span>
        <span class="flex text-sm font-bold text-grey-550">1</span>
      </div>
      <div class="flex w-full justify-between gap-x-2 border-b border-gray-200 py-3">
        <span class="flex shrink-0 text-sm text-grey-550">Structura rezistenta:</span>
        <span class="flex text-sm font-bold text-grey-550">Caramida</span>
      </div>
    </div>
  </div>
</section>
</body></html>
"""


def test_parses_price():
    listing = parse_imobiliare_html(SAMPLE_HTML, external_id="275727717")
    assert listing.price_raw == "135.000 €"


def test_parses_address():
    listing = parse_imobiliare_html(SAMPLE_HTML, external_id="275727717")
    assert listing.address_raw == "Centrul Istoric, Bucuresti"


def test_parses_title_from_image_attribute():
    listing = parse_imobiliare_html(SAMPLE_HTML, external_id="275727717")
    assert listing.title == "Garsoniera 40mp in Centrul Istoric - Str. Selari nr. 2, imobil fara risc seismic"


def test_parses_description_full_text_not_truncated():
    listing = parse_imobiliare_html(SAMPLE_HTML, external_id="275727717")
    assert "Suprafata utila: 40 mp" in listing.description


def test_parses_spec_rows():
    listing = parse_imobiliare_html(SAMPLE_HTML, external_id="275727717")
    assert listing.specs["Suprafata construita:"] == "50 mp"
    assert listing.specs["Nr. bai:"] == "1"
    assert listing.specs["Structura rezistenta:"] == "Caramida"


def test_deduplicates_gallery_image_urls():
    listing = parse_imobiliare_html(SAMPLE_HTML, external_id="275727717")
    assert listing.image_urls == [
        "https://i.roamcdn.net/prop/imo/gallery-main-900w-watermark/example1.jpg",
        "https://i.roamcdn.net/prop/imo/gallery-main-900w-watermark/example2.jpg",
    ]


def test_raises_when_price_element_missing():
    with pytest.raises(ValueError, match="price"):
        parse_imobiliare_html("<html><body></body></html>", external_id="000")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ingest/test_imobiliare_parser.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'realestate.ingest.imobiliare_parser'`

- [ ] **Step 3: Write the implementation**

```python
# src/realestate/ingest/imobiliare_parser.py
from bs4 import BeautifulSoup
from realestate.models import RawListing


def parse_imobiliare_html(html: str, external_id: str) -> RawListing:
    soup = BeautifulSoup(html, "lxml")

    price_container = soup.find(attrs={"aria-label": "price"})
    price_el = price_container.find("span") if price_container else None
    if price_el is None:
        raise ValueError(f"listing {external_id}: price element not found")
    price_raw = price_el.get_text(strip=True)

    address_el = soup.find(attrs={"data-cy": "listing-address"})
    if address_el is None:
        raise ValueError(f"listing {external_id}: address element not found")
    address_raw = address_el.get_text(strip=True)

    description_el = soup.find(id="truncatedDescription")
    if description_el is None:
        raise ValueError(f"listing {external_id}: description element not found")
    description = description_el.get_text(separator="\n", strip=True)

    title_img = soup.find("img", attrs={"title": True})
    if title_img is None:
        raise ValueError(f"listing {external_id}: title not found (no img[title] element)")
    title = title_img["title"].strip()

    image_urls: list[str] = []
    for img in soup.find_all("img", class_="object-contain", src=True):
        if img["src"] not in image_urls:
            image_urls.append(img["src"])

    specs: dict[str, str] = {}
    specs_section = soup.find(attrs={"data-cy": "basic-info-section"})
    if specs_section is not None:
        for row in specs_section.find_all("div", class_="justify-between"):
            spans = row.find_all("span", recursive=False)
            if len(spans) != 2:
                continue
            specs[spans[0].get_text(strip=True)] = spans[1].get_text(strip=True)

    return RawListing(
        external_id=external_id,
        title=title,
        description=description,
        price_raw=price_raw,
        address_raw=address_raw,
        specs=specs,
        image_urls=image_urls,
        source_file=f"{external_id}.html",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ingest/test_imobiliare_parser.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
mkdir -p tests/ingest
touch tests/ingest/__init__.py
git add src/realestate/ingest/ tests/ingest/
git commit -m "feat: parse imobiliare.ro listing HTML into RawListing"
```

---

### Task 4: Manual HTML collection loader

**Files:**
- Create: `src/realestate/ingest/manual_loader.py`
- Test: `tests/ingest/test_manual_loader.py`

**Interfaces:**
- Consumes: `parse_imobiliare_html` from Task 3
- Produces: `load_manual_html_directory(directory: Path) -> tuple[list[RawListing], list[tuple[Path, str]]]` — returns `(listings, failures)`, where `failures` is `(file_path, error_message)` for any file that failed to parse. Failures never raise — that's the pipeline's quarantine mechanism (see spec's Error Handling section).

- [ ] **Step 1: Write the failing test**

```python
# tests/ingest/test_manual_loader.py
from pathlib import Path
from realestate.ingest.manual_loader import load_manual_html_directory

VALID_HTML = """
<html><body>
<img class="relative h-full w-full object-contain" src="https://example.com/a.jpg"
     title="Test listing title">
<div aria-label="price"><span>100.000 &euro;</span></div>
<p data-cy="listing-address">Some Area, Bucuresti</p>
<div id="truncatedDescription">Some description text.</div>
</body></html>
"""

MALFORMED_HTML = "<html><body><p>no price here</p></body></html>"


def test_loads_valid_html_files(tmp_path: Path):
    (tmp_path / "111.html").write_text(VALID_HTML, encoding="utf-8")

    listings, failures = load_manual_html_directory(tmp_path)

    assert len(listings) == 1
    assert listings[0].external_id == "111"
    assert failures == []


def test_quarantines_malformed_files_instead_of_raising(tmp_path: Path):
    (tmp_path / "111.html").write_text(VALID_HTML, encoding="utf-8")
    (tmp_path / "222.html").write_text(MALFORMED_HTML, encoding="utf-8")

    listings, failures = load_manual_html_directory(tmp_path)

    assert len(listings) == 1
    assert len(failures) == 1
    assert failures[0][0].name == "222.html"
    assert "price" in failures[0][1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ingest/test_manual_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'realestate.ingest.manual_loader'`

- [ ] **Step 3: Write the implementation**

```python
# src/realestate/ingest/manual_loader.py
from pathlib import Path
from realestate.ingest.imobiliare_parser import parse_imobiliare_html
from realestate.models import RawListing


def load_manual_html_directory(
    directory: Path,
) -> tuple[list[RawListing], list[tuple[Path, str]]]:
    listings: list[RawListing] = []
    failures: list[tuple[Path, str]] = []

    for html_file in sorted(directory.glob("*.html")):
        external_id = html_file.stem
        try:
            html = html_file.read_text(encoding="utf-8")
            listings.append(parse_imobiliare_html(html, external_id))
        except Exception as exc:  # noqa: BLE001 - quarantine, don't crash the batch
            failures.append((html_file, str(exc)))

    return listings, failures
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ingest/test_manual_loader.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/realestate/ingest/manual_loader.py tests/ingest/test_manual_loader.py
git commit -m "feat: load a directory of manually collected listing HTML files"
```

---

### Task 5: Field normalizer

**Files:**
- Create: `src/realestate/enrich/__init__.py`
- Create: `src/realestate/enrich/normalizer.py`
- Test: `tests/enrich/test_normalizer.py`

**Interfaces:**
- Produces: `normalize_price_eur(price_raw: str) -> float`
- Produces: `normalize_rooms(title: str) -> int | None`
- Produces: `normalize_specs(specs: dict[str, str]) -> dict[str, object]` (keys: `built_area_sqm: float | None`, `bathrooms: int | None`, `kitchens: int | None`, `construction_material: str | None`, `building_type: str | None`, `transaction_type: str | None`, `floor_regime: str | None`, `payment_method: str | None`)

- [ ] **Step 1: Write the failing test**

```python
# tests/enrich/test_normalizer.py
from realestate.enrich.normalizer import (
    normalize_price_eur,
    normalize_rooms,
    normalize_specs,
)


def test_normalize_price_eur_parses_dotted_thousands():
    assert normalize_price_eur("135.000 €") == 135000.0


def test_normalize_rooms_detects_studio():
    assert normalize_rooms("Garsoniera 40mp in Centrul Istoric") == 1


def test_normalize_rooms_detects_explicit_count():
    assert normalize_rooms("Apartament 3 camere de vanzare") == 3


def test_normalize_rooms_returns_none_when_unknown():
    assert normalize_rooms("Teren de vanzare") is None


def test_normalize_specs_maps_known_labels():
    raw_specs = {
        "Suprafata construita:": "50 mp",
        "Nr. bai:": "1",
        "Structura rezistenta:": "Caramida",
    }
    normalized = normalize_specs(raw_specs)
    assert normalized["built_area_sqm"] == 50.0
    assert normalized["bathrooms"] == 1
    assert normalized["construction_material"] == "Caramida"


def test_normalize_specs_ignores_unknown_labels():
    normalized = normalize_specs({"Un label necunoscut:": "valoare"})
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
    "Suprafata construita:": "built_area_sqm",
    "Nr. bucatarii:": "kitchens",
    "Nr. bai:": "bathrooms",
    "Structura rezistenta:": "construction_material",
    "Tip imobil:": "building_type",
    "Tip tranzactie:": "transaction_type",
    "Regim inaltime:": "floor_regime",
    "Modalitate de plata:": "payment_method",
}

_NUMERIC_FIELDS = {"built_area_sqm", "bathrooms", "kitchens"}


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
        elif field in ("bathrooms", "kitchens"):
            match = re.search(r"(\d+)", raw_value)
            normalized[field] = int(match.group(1)) if match else None
        else:
            normalized[field] = raw_value
    return normalized
```

Note: the test fixtures above use plain ASCII labels (e.g. `"Nr. bai:"` not `"Nr. băi:"`) to keep this
listing readable — when wiring this against real saved HTML in Task 8, verify the exact diacritics
imobiliare.ro renders (`ă`, `ă`, `ț`) and adjust `SPEC_LABEL_MAP` keys to match exactly, since dict
lookup is exact-match. Add a step there to print `raw_listing.specs.keys()` against one real saved
file and confirm the map's keys line up before trusting this silently drops unmapped labels.

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

- [ ] **Step 1: Write the failing test**

```python
# tests/enrich/test_pipeline.py
from realestate.enrich.geocoder import CachedGeocoder, GeocodeResult
from realestate.enrich.pipeline import enrich_listing
from realestate.models import RawListing

RAW = RawListing(
    external_id="275727717",
    title="Garsoniera 40mp in Centrul Istoric",
    description="Vanzare garsoniera in Centrul Istoric.",
    price_raw="135.000 €",
    address_raw="Centrul Istoric, Bucuresti",
    specs={"Suprafata construita:": "50 mp", "Nr. bai:": "1"},
    image_urls=["https://example.com/a.jpg"],
    source_file="275727717.html",
)

STATIONS = [{"name": "Universitate", "lat": 44.4356, "lon": 26.1023}]


def test_enrich_listing_with_successful_geocode(tmp_path):
    geocoder = CachedGeocoder(
        cache_path=tmp_path / "cache.json",
        geocode_fn=lambda address: GeocodeResult(latitude=44.4325, longitude=26.1013),
    )

    def fake_nearest_station(lat, lon, stations, **kwargs):
        return ("Universitate", 6.5)

    enriched = enrich_listing(
        RAW, geocoder=geocoder, stations=STATIONS, nearest_station_fn=fake_nearest_station
    )

    assert enriched.price_eur == 135000.0
    assert enriched.rooms == 1
    assert enriched.built_area_sqm == 50.0
    assert enriched.bathrooms == 1
    assert enriched.latitude == 44.4325
    assert enriched.location_confidence == "ok"
    assert enriched.nearest_subway_station == "Universitate"
    assert enriched.subway_walking_minutes == 6.5


def test_enrich_listing_flags_low_confidence_location_when_geocode_fails(tmp_path):
    geocoder = CachedGeocoder(
        cache_path=tmp_path / "cache.json",
        geocode_fn=lambda address: None,
    )

    enriched = enrich_listing(
        RAW, geocoder=geocoder, stations=STATIONS, nearest_station_fn=lambda *a, **k: None
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
    geocode_result = geocoder.geocode(raw.address_raw)

    latitude = geocode_result.latitude if geocode_result else None
    longitude = geocode_result.longitude if geocode_result else None
    location_confidence = "ok" if geocode_result else "low_confidence_location"

    station_name: str | None = None
    walking_minutes: float | None = None
    if geocode_result is not None:
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
        bathrooms=specs.get("bathrooms"),
        construction_material=specs.get("construction_material"),
        building_type=specs.get("building_type"),
        floor_regime=specs.get("floor_regime"),
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
Expected: PASS (2 passed)

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
- Consumes: `load_manual_html_directory` (Task 4), `enrich_listing` (Task 8), `Embedder` (Task 9), `VectorStore` (Task 10)
- Produces: `run_ingestion(html_dir: Path, *, geocoder, stations, embedder: Embedder, store: VectorStore) -> IngestionReport` where `IngestionReport` has `.succeeded: int`, `.parse_failures: list[tuple[Path, str]]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingestion_pipeline.py
from pathlib import Path

from realestate.enrich.geocoder import CachedGeocoder, GeocodeResult
from realestate.ingestion_pipeline import run_ingestion

VALID_HTML = """
<html><body>
<img class="relative h-full w-full object-contain" src="https://example.com/a.jpg"
     title="Garsoniera 40mp in Centrul Istoric">
<div aria-label="price"><span>135.000 &euro;</span></div>
<p data-cy="listing-address">Centrul Istoric, Bucuresti</p>
<div id="truncatedDescription">Vanzare garsoniera in Centrul Istoric.</div>
</body></html>
"""


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
    (tmp_path / "275727717.html").write_text(VALID_HTML, encoding="utf-8")
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
    assert store.upserted[0][0] == "275727717"
    assert store.upserted[0][2]["price_eur"] == 135000.0
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
from realestate.ingest.manual_loader import load_manual_html_directory
from realestate.store.base import VectorStore


@dataclass
class IngestionReport:
    succeeded: int = 0
    parse_failures: list[tuple[Path, str]] = field(default_factory=list)


def run_ingestion(
    html_dir: Path,
    *,
    geocoder: CachedGeocoder,
    stations: list[dict],
    embedder: Embedder,
    store: VectorStore,
) -> IngestionReport:
    raw_listings, parse_failures = load_manual_html_directory(html_dir)

    succeeded = 0
    for raw in raw_listings:
        enriched = enrich_listing(raw, geocoder=geocoder, stations=stations)
        vector = embedder.embed_passage(enriched.description)
        store.upsert(
            enriched.external_id,
            vector,
            payload=enriched.model_dump(),
        )
        succeeded += 1

    return IngestionReport(succeeded=succeeded, parse_failures=parse_failures)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ingestion_pipeline.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Write the real CLI entry point**

```python
# scripts/run_ingestion.py
"""Run the full ingestion pipeline against manually collected imobiliare.ro HTML.

Usage: python scripts/run_ingestion.py data/raw/imobiliare_html
"""

import sys
from pathlib import Path

from realestate.embed.sentence_embedder import MultilingualE5Embedder
from realestate.enrich.geocoder import CachedGeocoder
from realestate.enrich.poi import fetch_bucharest_subway_stations
from realestate.ingestion_pipeline import run_ingestion
from realestate.store.qdrant_store import QdrantListingStore

if __name__ == "__main__":
    html_dir = Path(sys.argv[1])
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
    external_id="275727717",
    title="Garsoniera 40mp in Centrul Istoric",
    description="Garsoniera luminoasa, parchet, in Centrul Istoric.",
    price_eur=135000.0,
    rooms=1,
    built_area_sqm=50.0,
    bathrooms=1,
    construction_material="Caramida",
    building_type="Bloc de apartamente",
    floor_regime="P+2E",
    address_text="Centrul Istoric, Bucuresti",
    latitude=44.43,
    longitude=26.10,
    location_confidence="ok",
    nearest_subway_station="Universitate",
    subway_walking_minutes=6.5,
    image_urls=[],
)


def test_generate_eval_query_uses_listing_details_in_the_prompt():
    captured_prompts = []

    def fake_generate(prompt: str) -> str:
        captured_prompts.append(prompt)
        return "Bright studio near Universitate subway station"

    pair = generate_eval_query(LISTING, generate_fn=fake_generate)

    assert pair.relevant_listing_id == "275727717"
    assert pair.query == "Bright studio near Universitate subway station"
    assert "Centrul Istoric" in captured_prompts[0]
    assert "Universitate" in captured_prompts[0]
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
- **Type consistency:** `Embedder`/`VectorStore` protocol method signatures (Tasks 9, 10) match
  their usage in `ingestion_pipeline.py` (Task 11) and `retrieval.py` (Task 13). `EnrichedListing`
  fields (Task 2) match what `enrich_listing` (Task 8) constructs and what `generate_eval_query`
  (Task 14) reads.
- **Known follow-up baked into Task 5:** the `SPEC_LABEL_MAP` diacritics must be verified against
  real saved HTML before Task 8 is trusted end-to-end — flagged inline in that task rather than
  glossed over, since silently dropping unmapped spec labels is exactly the kind of thing that
  looks like it works until you check real data.

# Bucharest real-estate semantic search

Natural-language search over Bucharest apartment listings ("2 camere sub 90.000 €, sector 3,
aproape de metrou, liniștit"). Listings are scraped from OLX, enriched (price, rooms, sector,
location, subway access), embedded, and stored in Qdrant. Queries are parsed by a local LLM
into hard filters + semantic text.

## Quick start

```bash
docker compose up -d qdrant                 # vector DB on :6333 (OSRM service needs data, see below)
uv sync --extra dev                         # Python 3.14 venv in .venv
ollama pull qwen3.6:27b                     # local LLM for query parsing + location clues
uv run python scripts/run_ingestion.py 100  # scrape (if needed) + enrich + embed + store
uv run uvicorn realestate.web.app:app --reload   # web demo on :8000
uv run python -m realestate.eval.run_eval [--strict] [--price-margin 1.15] [--queries q.json]
```

Ingestion reads `data/raw/olx_html/*.html`; OLX currently answers 403 to the scraper (not
bypassed), so it runs on the 100 saved pages (copy from
`.worktrees/phase1-data-pipeline/data/raw/olx_html/` if missing). A full run takes ~30-40 min,
almost all of it LLM clue extraction, and overwrites the `listings` collection.

## Quality gate (before every commit)

```bash
uv run ruff check src tests && uv run ruff format --check src tests
uv run pyright
uv run pytest -m "not integration"
```

TDD: write the failing test first. External calls (HTTP, LLM, geocoder, routing) are injected
(`post_fn`, `search_fn`, `walking_fn`, ...) so unit tests are offline and deterministic.
Commits: one short subject line, no trailers; stage files explicitly; never stage `docs/`,
`.tokensave/`, `.memsearch/`. Push via the personal SSH remote (`git@personal:izabelamaria24/Test.git`).

## Architecture

```
ingest/   olx_scraper (download) -> olx_parser (HTML -> RawListing) -> olx_loader (directory)
enrich/   normalizer (price, rooms, specs, sector)
          location_clues (LLM -> street, number, neighbourhood, metro, walk minutes, landmarks)
          geocoder (Nominatim search -> candidate places with shapes, cached)
          location (clues -> Evidence) -> fusion (probability grid -> FusedLocation)
          poi (subway stations from Overpass, cached; nearest station + walking time via OSRM)
          pipeline (RawListing -> EnrichedListing)
embed/    MultilingualE5Embedder (768-d)
store/    QdrantListingStore (payload filters: price, rooms, sector, subway minutes)
query/    parser (LLM -> filters + semantic text), retrieval (exact matches first, then
          labelled near-misses: price x1.15, neighbouring sectors)
eval/     generate_queries, metrics, run_eval
web/      FastAPI app + static index.html (badges for exact/near matches, strict toggle)
ingestion_pipeline.py  orchestrates a batch; flags OLX map points shared by >=3 listings
```

Protocols (`Embedder`, `VectorStore`, `PlaceSearcher`, `LocationClueExtractor`) keep layers
swappable. `EnrichedListing` fields added later have defaults so old Qdrant payloads still load.

## Location: how it works and why

The hard part of the project. Facts about the data (100 OLX listings):
- `address_text` is only "Bucuresti - Ilfov, Bucuresti, Sectorul N".
- OLX map points are privacy-blurred (radius 2-5 km); 67/100 are one of 6 per-sector default
  points and carry no information.
- ~44% of descriptions name a street; many name a neighbourhood, metro station or landmarks.

Algorithm (`enrich/location.py`, `enrich/fusion.py`):
1. The LLM extracts clues; output is validated (lists -> first item, "Sector 3" is not a
   neighbourhood, implausible walk minutes dropped, chain stores need a branch name).
2. Each clue becomes `Evidence`: candidate geometries (up to 3 Nominatim matches, with street
   lines / area outlines), a `reach_m` (how far the flat may be from it) and a `reliability`.
   Sector outline, OLX circle (unless a default point), building, street (only if the building
   isn't found; retried without ", București" for blocks just over the city limit), metro
   (station list; reach = stated minutes x 80 m), landmarks, neighbourhood.
3. Grid fusion over Bucharest (100 m cells): each clue's likelihood is 1 within reach and fades
   over 0.5 x reach; clues combine as a robust mixture (`reliability x clue + (1 - reliability)
   x uniform`), so a wrong clue is outvoted, and an ambiguous name resolves to the candidate the
   other clues agree with.
4. Estimate = centre of the most probable connected region (not the plain mean, which can fall
   between modes); uncertainty = radius holding 90% of the probability; clues within reach of
   the estimate are `location_evidence`, the rest `location_conflicts`.
5. Subway walking time is computed only when uncertainty <= 1 km.

Result of the last ingestion (no subway stations, Overpass was down): all 100 located; 27
within 300 m, 19 within 1 km, 24 within 3 km, 30 beyond; 9 listings with a conflicting landmark.

Reach/reliability values are hand-set judgement calls; there is no ground truth to calibrate
them yet (see "Open problems").

## External services and caches

| Service | Use | Notes |
|---|---|---|
| Ollama `qwen3.6:27b` (local) | query parsing, location clues | needs `"think": false`, `"format": "json"` |
| Nominatim (public) | place search | 1 req/s, identifying User-Agent, bounded to Bucharest viewbox; cache `data/cache/places_cache.json` |
| Overpass (public) | subway stations | 406 without User-Agent; often 502/timeouts; cached once in `data/cache/subway_stations.json` |
| OSRM (docker) | walking minutes | **not running**: needs `data/osrm/romania-latest.osrm` (download Geofabrik PBF, run `osrm-extract -p /opt/foot.lua`, `osrm-partition`, `osrm-customize`) |
| Qdrant 1.11 (docker) | vectors + payload | client 1.19: pass `check_compatibility=False` |

`data/raw/`, `data/cache/`, `data/osrm/`, `.worktrees/` are gitignored.

## Status (2026-10-31)

Stacked branches, must land in order:
1. `feat/sector-filter-strict-toggle` — sector filter, strict/flexible widening, web toggle,
   configurable eval. PR #29 open.
2. `feat/exact-first-near-misses` — exact matches first, then labelled near-misses (18af8f1).
3. `feat/location-from-text` — LLM clues + grid fusion, cached stations (2e0c235, 8219515).

Known gaps:
- No subway walking minutes for any listing (OSRM not set up; stations missing in last run), so
  queries with a subway limit return nothing.
- LLM clue extraction is not cached; every re-ingest repeats it.
- No ground-truth locations, so accuracy and calibration are unmeasured.

## Roadmap

Near term:
1. Walking minutes: straight-line estimate (`distance x 1.3 / 80 m/min`) now, OSRM later; with
   both, the ratio flags barriers (railways, river, fenced areas).
2. Cache LLM clues keyed by text hash + prompt version.
3. Ground-truth set: label ~100 listings with their true building (text + street view), then
   measure median error and calibration (does the 90% radius contain the truth 90% of the time?).

Ideas worth exploring (see the discussion in the 2026-10-31 session):
- Search over uncertainty: keep each listing's location distribution and rank by the
  probability that it satisfies the query (e.g. P(walk <= 10 min)), using city-wide fields
  precomputed once on the same grid (minutes to metro, distance to parks, seismic risk).
- Same-building / cross-platform matching (identical text, perceptual photo hashes, agent
  phone) as graph clustering; merge evidence across matched listings.
- Match listing attributes (floors, year, block type) against OSM buildings in the likely area.
- Learned priors: price per m², building era and photo style -> likely neighbourhoods, trained on
  listings that are located precisely, used as one more evidence layer (corpus-level EM).
- Local OSM stack from one Geofabrik extract (Nominatim + Overpass + OSRM) for reproducibility
  and no rate limits.
- Product ideas from `notes.txt`: fraud/trust score, seismic risk score, nearby amenities,
  fair-price and inflation-adjusted price estimates.

# AI-Powered Real Estate Aggregator — Phase 1: Data Pipeline & Semantic Search

## Vision & Project Structure

Long-term vision: an AI-driven real estate aggregator that replaces filter-heavy search with
semantic understanding, multi-modal fraud/trust scoring, and learned recommendations. This is a
2-year effort meant to double as a hobby project, a master's thesis (FMI UNIBUC, Applied AI), and
potentially a real product — no fixed deadline pressure, complexity is welcome.

The full vision spans three largely independent subsystems. Building all three at once isn't
tractable as a single spec, so the project is split into sequential phases, each with its own
design → plan → implementation cycle:

1. **Phase 1 (this spec): Data pipeline + semantic search.** Foundation everything else depends
   on — ingesting real listings and enabling natural-language search over them.
2. **Phase 2: Multi-modal fraud/trust scoring.** CLIP-style image-text consistency, a
   synthetically labeled fraud dataset, later GNN-based fraud-ring detection. Built on top of
   Phase 1's listing store.
3. **Phase 3: Learned recommendations.** Starts as content-based recommendation using Phase 1's
   embeddings; only becomes a genuine RL formulation once the product has real user interaction
   data — RL over recommendations cannot be meaningfully trained without real traffic, so this
   phase is deliberately deferred until the product exists and is used.

This document specifies **Phase 1 only**.

## Phase 1 Goal

Given a natural-language query (e.g. "a bright, pet-friendly apartment with hardwood floors, max
15 minutes walking distance from a subway station"), return real, ranked Bucharest-area listings
that satisfy it — with a rigorous, reusable evaluation harness, not just a working demo. UI is
explicitly out of scope; a bare CLI/script interface is sufficient for now.

## Market & Language Scope

Primary market: Romania (Bucharest first, given subway-distance and future seismic-risk data
availability). Embeddings use a multilingual pretrained model as the baseline so the door stays
open to non-Romanian markets/languages later, with fine-tuning as a stretch goal once
query-listing pairs exist to train on.

## Data Strategy

- **Prototyping data:** open Kaggle datasets (e.g. Bucharest House Price Dataset, generic
  apartment/housing datasets) to build and validate the pipeline before dealing with scraping
  constraints.
- **Real data:** small-scale, rate-limited, respectful scraping of Romanian listing platforms
  (e.g. imobiliare.ro, storia.ro) for research/development use, not redistribution. A paid
  aggregator API (e.g. PropAPIS, ~$99–299/mo) is a possible later convenience for bulk pulls, not
  needed for Phase 1.
- **Fraud-labeled data** (Phase 2 concern, noted here for continuity): real confirmed-fraud
  examples are essentially unobtainable from scraped data since platforms remove them. The plan is
  to synthetically construct fraud by taking real legit listings and injecting known patterns
  (reused images across different-city listings, diffusion-generated fake photos, price/amenity
  mismatches, duplicated/paraphrased descriptions), optionally seeded with real scam phrasing from
  consumer-protection sources (r/Scams, FTC/EU alerts) for authentic language patterns.

## Architecture Overview

Five independently testable layers:

1. **Ingestion** — scrapers + Kaggle loaders, normalized into one common `RawListing` schema.
2. **Enrichment** — geocoding, nearest-subway walking-distance computation, structured field
   normalization (price, rooms, area).
3. **Embedding** — pretrained multilingual sentence embedding model (baseline), swappable for a
   fine-tuned model later.
4. **Storage** — Qdrant (self-hosted via Docker), storing both the vector and structured metadata
   (payload) per listing.
5. **Query** — parses a natural-language query into structured constraints + a semantic remainder,
   filters+searches Qdrant, returns ranked results.

### Retrieval design: hybrid (structured filters + geospatial + dense vector)

Three approaches were considered:

- **Pure dense retrieval** — flatten listing into text, embed, cosine-similarity search. Simplest,
  but fails on hard/checkable constraints (e.g. "max 15 min walking distance" needs an actual
  distance computation, not a fuzzy text match).
- **Hybrid retrieval (chosen)** — parse the query into structured constraints (rooms, price range,
  pet-friendly, POI-distance) plus a free-text semantic remainder (style/emotional intent like
  "bright", "hardwood floors"). Structured constraints are enforced via metadata filtering +
  precomputed geospatial data; the semantic remainder is handled via dense vector similarity. This
  mirrors production retrieval systems (cf. DPR) and is naturally set up for ablation studies
  (contribution of geo layer vs. semantic layer vs. fine-tuned vs. baseline embeddings) — valuable
  for the thesis, not just the product.
- **LLM re-ranking** — retrieve a broad candidate set cheaply, then have an LLM holistically
  re-rank against the query. Deferred: usable as an optional layer on top of the hybrid approach's
  shortlist later, not part of the Phase 1 core.

### Geospatial data source

OpenStreetMap Overpass API (subway station locations) + a self-hosted OSRM instance (walking-route
distance calculation), not Google Maps. Google's Distance Matrix/Routing APIs meter per request,
which is a poor fit for a long-running, iteration-heavy research project; OSM/OSRM is free,
unlimited at this scale, self-hosted, and its methodology is fully inspectable — useful for
explaining the approach in a thesis. Google Maps may be used manually to spot-check a handful of
computed distances, not as the production engine.

### Local-first inference

Target hardware: Apple M5 Max, 48GB unified memory. This comfortably runs an 8–14B (or ~30B
quantized) LLM locally via MLX-LM or Ollama, so the query parser runs entirely on-device instead
of calling an external API per query — no per-query cost, no rate limits, works offline. Embedding
fine-tuning (sentence-transformer scale, <1B params) also trains quickly on-device via MPS/MLX.
External APIs remain a swappable fallback (see Dependency Inversion below) if a stronger model is
ever needed than what fits locally.

## Components

- `ingest/` — one scraper module per source + a Kaggle loader, each emitting the same
  `RawListing` schema. Scrapers are rate-limited and cache raw HTML so re-runs don't re-hit the
  source.
- `enrich/` — geocoder (address → lat/lon), POI-distance calculator (lat/lon → nearest subway +
  walking minutes via OSRM), field normalizer (freeform price/area text → numeric).
- `embed/` — wraps the sentence embedding model; takes normalized text, returns a vector.
  Swappable so baseline vs. fine-tuned model is a config change.
- `store/` — thin Qdrant client wrapper: upsert listing (vector + payload), query (filter + vector
  search).
- `query/` — NL query parser (local LLM extracting `{filters: {...}, semantic_text: "..."}`) + the
  retrieval orchestrator combining filtered search with vector ranking.
- `eval/` — offline IR evaluation harness: Recall@k, NDCG@k, MRR against a labeled
  query→relevant-listing set.

## Engineering Principles

- **Single Responsibility** — each layer does exactly one job; a scraper doesn't know about
  embeddings, the query orchestrator doesn't know about Qdrant internals.
- **Open/Closed** — `ingest/` and `embed/` are built against interfaces (`Scraper`, `Embedder`),
  not concrete classes. Adding a source or swapping the embedding model means writing a new
  implementation, not touching existing code.
- **Liskov Substitution** — every scraper implements the same `Scraper` interface and returns the
  same `RawListing` shape, so the ingestion orchestrator can run any of them interchangeably.
- **Interface Segregation** — narrow interfaces per role (`Embedder.embed(text) -> vector`,
  `VectorStore.upsert(...)`/`.query(...)`) rather than one broad data-layer interface.
- **Dependency Inversion** — the query orchestrator depends on `VectorStore`/`Embedder`
  abstractions, injected at startup, not on Qdrant or a specific model directly. This is what
  makes "baseline vs. fine-tuned embedding" and "Qdrant vs. another vector DB" config-level swaps
  instead of rewrites.

## Scalability & Performance

- Ingestion is idempotent (dedupe by a stable listing ID) and rate-limited/batched, so re-scraping
  never creates duplicates and can be horizontally parallelized later.
- Embedding generation runs as batch inference, isolated as its own worker, scalable (or GPU-bound)
  independently of the query-serving path.
- Qdrant performs filter-then-ANN-search natively — structured filters (price, rooms, geo) are
  applied before the vector similarity step, so query latency doesn't degrade as filters get more
  selective. This is the main performance lever for the hybrid retrieval approach.
- The layers are module boundaries within a modular monolith for now; the interface-based design
  means they can be split into separate services (ingestion pipeline, embedding worker, query API)
  later without a redesign.

## Data Flow

**Ingestion time:**
`Scraper/KaggleLoader → RawListing → Enricher (geocode + POI distance + field normalization) → Embedder (text → vector) → VectorStore.upsert(vector, payload)`

Each stage is a pure transformation with a typed input/output — testable in isolation, and
replayable independently (e.g. re-embed everything after fine-tuning, without re-scraping or
re-geocoding).

**Query time:**
`NL query → QueryParser (local LLM) → {structured filters, semantic text} → Embedder.embed(semantic text) → VectorStore.query(filters, vector) → ranked results`

Both the parser and embedder run locally, so query latency is dominated by local inference +
Qdrant's filtered ANN search — no network round-trip to an external API in the hot path.

## Error Handling

- **Scraping** — respect robots.txt/rate limits, retry with backoff on transient failures, cache
  raw HTML so a crash mid-run doesn't lose progress or force re-hitting the source.
- **Geocoding/POI** — if an address fails to geocode, the listing is still stored but flagged
  `low_confidence_location` rather than dropped; it's just unusable for geo-constrained queries
  until fixed.
- **Malformed listings** — schema validation at the `RawListing` boundary; failures go to a
  quarantine table for later inspection instead of crashing the pipeline.
- **Query parsing** — if the local LLM produces an unparseable/invalid filter structure, fall back
  to treating the entire query as semantic text (pure vector search) rather than failing the
  request.

## Evaluation (also serves as thesis experiment infrastructure)

No real user click data exists yet, so the labeled eval set is built via "reverse generation":
take real listings, use an LLM to generate natural-language queries that *should* match them, and
use those (query, relevant-listing) pairs as ground truth. Measure Recall@k, NDCG@k, MRR. This
harness is what enables real ablations later — baseline vs. fine-tuned embeddings, pure-vector vs.
hybrid retrieval — ready from day one rather than bolted on afterward.

## Out of Scope for Phase 1 (Future Phases Backlog)

Captured here so they aren't lost, but explicitly not designed or built in Phase 1:

- **Fraud/trust scoring** (Phase 2) — CLIP-style image-text consistency, synthetic fraud dataset,
  GNN fraud-ring detection. More relevant to short-term/Airbnb-style rentals.
- **Structural/seismic risk scoring** — Bucharest-specific danger score based on zone, year of
  construction, floor count, and Romania's public "bulină roșie" (red-dot) seismically at-risk
  building registry. More relevant to long-term rental/purchase decisions than fraud is.
- **Fair-value pricing** — an estimated "true value" for a listing, including inflation-adjusted
  pricing comparisons.
- **POI enrichment display** — surfacing nearby points of interest (parks, etc.) relevant to a
  user's stated preferences on a listing's detail page.
- **Demographic/audience best-fit** — categorizing which type of person/household a listing suits
  best.
- **Learned recommendations** (Phase 3) — content-based initially, formulated as RL only once real
  user interaction data exists.

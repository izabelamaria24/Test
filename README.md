# realestate

Semantic search over Bucharest real-estate listings — Phase 1 of an AI-powered
real-estate aggregator (master's thesis + hobby project).

Pipeline: OLX.ro HTML → parse → enrich (geocode, subway distance, normalize) →
embed (`multilingual-e5`, 768-dim) → store in Qdrant → natural-language query →
LLM parse (Ollama) → hybrid retrieval (structured filters + geospatial + vector).

## Setup

```bash
uv sync --extra dev          # create .venv with runtime + dev deps
docker compose up -d         # Qdrant (+ OSRM) for integration tests / live demo
pre-commit install           # run lint/format on every commit
```

## Quality gates

All changes must pass these before commit (also enforced in CI):

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run pyright
uv run pytest -m "not integration"      # unit tests (no live services)
```

Integration tests need live Qdrant/Ollama and are marked `@pytest.mark.integration`:

```bash
uv run pytest -m integration
```

## Workflow

`main` is protected. All work goes through a feature branch → PR → AI review +
green CI → merge. See the `quality-gate` skill in `.claude/skills/` for the full
loop and repo conventions.

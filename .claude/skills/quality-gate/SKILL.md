---
name: quality-gate
description: >-
  Enforce this repo's quality bar before every commit on the realestate project.
  Use for ANY code change here: run the TDD loop, run ruff/pyright/pytest, keep
  staging clean, and follow the commit + personal-identity conventions. Invoke
  before staging or committing anything in this repository.
---

# quality-gate

Project-local quality gate for `realestate`. Every code change must pass these
gates **before commit**. Do not commit red.

## Identity (CRITICAL)

- Use the PERSONAL GitHub account `izabelamaria24` for everything in this repo.
- NEVER use a work / corporate account, key, or token here — no commits, pushes,
  `gh` calls, or auth. Do not name or embed such accounts in committed files.
- `git` push/pull already use the `personal` SSH alias — no `gh` needed for git.
- For `gh` API calls, scope per-command; never change global `gh` state:
  `GH_TOKEN=$(gh auth token --user izabelamaria24) gh <cmd>`

## Workflow

1. **Branch** off `main`: `git switch -c <type>/<slug>` (feat/fix/chore/docs/...).
2. **TDD**: write a failing test, then the minimal code to pass it. Red -> green
   -> refactor. Keep each branch/PR to one logical change.
3. **Run the gates** (venv via uv):
   ```bash
   uv run ruff check src tests
   uv run ruff format src tests        # or --check in CI
   uv run pyright
   uv run pytest -m "not integration"
   ```
   Integration tests (live Qdrant/Ollama) are marked `@pytest.mark.integration`
   and run locally only: `uv run pytest -m integration`.
4. **Fix** everything until all gates are green. Never commit on red.
5. **Stage narrowly**: only the code/test/config files this change touches.
   Never stage `docs/`, `.memsearch/`, `.tokensave/`, `.superpowers/`, `.venv/`,
   or caches.
6. **Commit**: a single short imperative subject line. No `Co-Authored-By`.
7. **PR**: push the branch and open a PR into `main`. CI must pass and the AI
   review must be addressed before merge (branch protection enforces this).

## Code quality standards

- **Architecture**: respect the layered design (`ingest`/`enrich`/`embed`/
  `store`/`query`/`eval`). Each layer depends only on its neighbor's abstraction
  (Protocols in `store/base.py` etc.), never on a concrete implementation.
- **Functions**: small and single-purpose; prefer pure functions for logic that
  can be pure (e.g. normalizers). Push side effects (I/O, HTTP, DB) to the edges.
- **Typing**: full type hints on public functions. Keep `pyright` basic-clean —
  no `object` leaking into typed fields; model structured data with `TypedDict`,
  `pydantic`, or dataclasses instead of loose dicts.
- **Naming**: descriptive, no abbreviations that aren't domain terms. Booleans
  read as predicates (`is_`, `has_`).
- **Errors**: validate only at system boundaries (parsing, external responses).
  Chain exceptions with `raise ... from err`. Don't add defensive handling for
  states that can't occur.
- **No dead code**: no commented-out blocks, unused imports/vars, or speculative
  helpers for a single caller. Ruff `F`/`B` catch most of this — keep it clean.
- **Comments**: only for non-obvious *why*, one line. Don't restate the code.
- **Dependencies**: add a dep only when it earns its place; pin via `uv` and
  commit `uv.lock`. Prefer the stdlib.

## Testing standards

- Every behavior change ships with a test. Bug fix = failing test first, then fix.
- Tests are deterministic and isolated: inject fakes for HTTP/LLM/geocoder
  (`post_fn`, fake stations, etc.); no network in unit tests.
- Anything needing live Qdrant/Ollama/OSRM is `@pytest.mark.integration`.
- Test behavior and edge cases (None/empty/boundary), not implementation detail.

## Security

- No secrets, tokens, or credentials in code, tests, or fixtures — ever.
- Treat scraped HTML / LLM output / user queries as untrusted; validate and
  constrain before use (the query parser already drops implausible filters).
- Respect scraping etiquette already encoded (rate limits, allowed sources).
- Review new dependencies for provenance before adding.

## Definition of done

- ruff (lint + format), pyright, and non-integration pytest all green.
- New/changed behavior is covered by a deterministic test.
- Change is narrowly scoped and self-contained (one logical change per PR).
- Committed on the personal identity with a clean, single-line subject.
- CI green and AI review addressed before merge.

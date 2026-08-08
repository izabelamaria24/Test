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

- Use the PERSONAL GitHub account `izabelamaria24` only. Never the work account
  `ijilavu_adobe`.
- `git` push/pull already use the `personal` SSH alias — no `gh` needed for git.
- For `gh` API calls, scope per-command; never change global gh state:
  `GH_TOKEN=$(gh auth token --user izabelamaria24) gh <cmd>`

## Workflow

1. **Branch** off `main`: `git switch -c <type>/<slug>` (feat/fix/chore/...).
2. **TDD**: write a failing test, then the minimal code to pass it. Red -> green.
3. **Run the gates** (venv via uv):
   ```bash
   uv run ruff check src tests
   uv run ruff format src tests        # or --check in CI
   uv run pyright
   uv run pytest -m "not integration"
   ```
   Integration tests (live Qdrant/Ollama) are marked `@pytest.mark.integration`
   and run locally only: `uv run pytest -m integration`.
4. **Fix** everything until all four gates are green. Never commit on red.
5. **Stage narrowly**: only the code/test/config files this change touches.
   Never stage `docs/`, `.memsearch/`, `.tokensave/`, `.venv/`, or caches.
6. **Commit**: a single short subject line. No `Co-Authored-By` trailer.
7. **PR**: push the branch and open a PR into `main`. CI must pass and an AI
   review must be addressed before merge (branch protection enforces this).

## Definition of done

- ruff (lint + format), pyright, and non-integration pytest all green.
- New/changed behavior is covered by a test.
- Commit staged narrowly with a clean subject line, on the personal identity.

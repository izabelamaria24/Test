---
name: python-ai-review
description: >-
  Review Python AI/ML projects (embeddings, retrieval/RAG, model-serving, eval
  harnesses) for the finer defects that generic linters and general AI review
  bots miss: model lifecycle, device/config portability, resource hygiene,
  evaluation leakage, reproducibility, embedding correctness, and inference
  cost. Use when reviewing a PR, diff, or module in an ML/LLM Python codebase.
---

# python-ai-review

A review rubric for Python **AI/ML** projects — model serving, embeddings,
vector search / RAG, and offline evaluation. It targets the class of issues that
`ruff`/`pyright`/`bandit` and general AI reviewers routinely miss because they
require understanding ML runtime behaviour.

Apply it to a diff or module. For each finding, report: file:line, a severity
(`blocker` / `major` / `minor` / `nit`), the concrete risk, and a fix. Prefer
few high-signal findings over many nits.

## 1. Model & resource lifecycle

- **No heavy work at import.** Models, tokenizers, DB/HTTP clients must NOT be
  constructed at module import time (breaks tests, tooling, CI, and startup).
  Flag module-level `Model(...)`, `SentenceTransformer(...)`, `QdrantClient(...)`.
- **Lazy init + startup warmup.** Prefer lazy construction (DI / cached getter)
  *plus* a server startup hook that warms the model, so the first request isn't
  slow. Flag lazy-only serving paths where a cold first request is user-facing.
- **Thundering herd.** Under concurrency, an un-synchronised lazy loader lets
  multiple requests build the same model at once. Flag `lru_cache`-only lazy
  loads on hot paths without a startup warmup or lock.
- **Close what you open.** `requests.Session`, DB clients, file handles, and
  torch resources must be closed (context manager or shutdown hook). Flag
  self-created clients that are never released.
- **Inference mode (torch).** Serving/eval paths should call `model.eval()`
  (switches Dropout/BatchNorm to inference behaviour) *and* wrap inference in
  `torch.inference_mode()` / `torch.no_grad()` when autograd outputs aren't
  needed — these are separate concerns; `eval()` does not disable gradients.

## 2. Config & portability

- **No hard-coded device.** `device="mps"`/`"cuda"` in library/serving code ties
  it to one machine. Require env/settings with a safe default (`cpu`) and a CPU
  fallback. Flag hard-coded devices outside a config layer.
- **No hard-coded model names / endpoints.** Model IDs, Ollama/OpenAI URLs, and
  vector-DB hosts belong in config, not scattered literals. Flag duplicated
  literals across web/eval/ingest.
- **Timeouts on every external call.** Every `requests`/httpx/model-API call
  needs an explicit `timeout`. Flag any network call without one (bandit misses
  `session.get`).

## 3. Evaluation correctness (highest-value, bots miss these)

- **Train/eval leakage.** A gold document also being in the indexed corpus is
  normal for retrieval metrics (standard IR separates corpus, queries, and
  relevance judgments, then scores returned corpus items) — that alone is not
  leakage. Flag real contamination instead: train/test overlap, and query- or
  answer-generation that leaks the gold answer/relevance into the query.
- **Metric honesty.** Retrieval metrics (recall@k, NDCG, MRR) measure whether a
  *known* relevant item is returned — not whether a relaxed/"compromise" result
  is desirable. Flag claims that IR metrics validate subjective quality.
- **Determinism.** Set/seed RNG (`random`, `numpy`, `torch`) where results feed
  assertions or reported numbers. Flag seed-free stochastic eval/tests.
- **Pinned model versions.** Reproducible results require pinned model + library
  versions (committed lockfile). Flag floating model tags in eval.

## 4. Embeddings & retrieval

- **Prefix/instruction correctness (model- and task-specific).** E5 requires
  `query:` / `passage:`; BGE uses an optional *query* instruction and no passage
  prefix; base GTE needs none while instruction-tuned variants (e.g.
  `gte-Qwen2-instruct`) need a query instruction. Resolve the model revision
  before flagging — a silent quality killer when wrong.
- **Normalization matches the metric.** Some stores normalize internally (e.g.
  Qdrant normalizes on upload for cosine, making cosine == dot). Require
  client-side normalization of *both* indexed and query vectors only when
  `dot`/`ip` is used as a cosine proxy and the store doesn't normalize
  equivalently. Flag genuine mismatches.
- **Dimensionality + distance config.** Vector dim and distance metric in the
  store must match the model. Flag hard-coded dims that can drift from the model.
- **Filter-then-ANN.** Structured filters should be native DB filters (pre-ANN),
  not Python post-filtering that silently drops recall. Flag post-filtering.

## 5. Inference cost & performance

- **Batch, don't loop.** Per-item `model.encode` in a loop over many items is
  slow; batch it. Flag N-call loops where a batch call exists.
- **Don't recompute.** Re-embedding unchanged data on every run wastes compute;
  ingestion should be idempotent (deterministic IDs). Flag full re-embeds.
- **Blocking work in async paths.** CPU/GPU-bound inference inside `async def`
  handlers blocks the event loop. Flag heavy sync work in async endpoints.

## 6. Data pipeline & safety

- Treat scraped HTML, LLM output, and user queries as **untrusted**; validate at
  boundaries. Flag `eval`/`exec`/unbounded parsing of external text.
- Respect source etiquette already encoded (rate limits, robots, allowed hosts).
- No secrets, tokens, or dataset paths hard-coded in code or fixtures.
- Graceful degradation: serving endpoints should return a clear 5xx (not a raw
  stack trace) when a model/vector-DB/LLM dependency is down; expose `/health`.

## Output format

```text
[severity] path:line — <one-line risk>
  fix: <concrete change>
```

End with a short verdict (ship / ship-with-fixes / needs-work) and up to 3
must-fix items (or `none`).

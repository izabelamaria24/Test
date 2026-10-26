"""Build an eval set from currently ingested listings and score the live retrieval pipeline.

Usage:
    python -m realestate.eval.run_eval [--strict] [--price-margin 1.15] [--queries PATH]

To compare settings, pass the same --queries file to every run: it is generated with the LLM
on first use and reused afterwards, so each setting is scored on identical queries.
"""

import argparse
import dataclasses
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path

import requests

from realestate.embed.sentence_embedder import MultilingualE5Embedder
from realestate.eval.generate_queries import EvalPair, generate_eval_query
from realestate.eval.metrics import evaluate
from realestate.models import EnrichedListing
from realestate.query.parser import OllamaQueryParser
from realestate.query.retrieval import DEFAULT_FLEXIBLE_PRICE_MARGIN, search
from realestate.store.qdrant_store import QdrantListingStore

_K = 10


def _price_margin(value: str) -> float:
    try:
        margin = float(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"not a number: {value!r}") from err
    if not (math.isfinite(margin) and margin >= 1.0):
        raise argparse.ArgumentTypeError("must be a number >= 1.0 (1.0 = no price widening)")
    return margin


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score the live retrieval pipeline on eval queries built from ingested listings."
    )
    parser.add_argument(
        "--strict", action="store_true", help="exact price/rooms/sector filters (default: flexible)"
    )
    parser.add_argument(
        "--price-margin",
        type=_price_margin,
        default=DEFAULT_FLEXIBLE_PRICE_MARGIN,
        help="flexible mode's price_max multiplier (ignored with --strict)",
    )
    parser.add_argument(
        "--queries", type=Path, default=None, help="JSON file of eval queries to reuse/create"
    )
    return parser.parse_args(argv)


def load_or_generate_pairs(
    path: Path | None, *, generate: Callable[[], list[EvalPair]]
) -> list[EvalPair]:
    if path is not None and path.exists():
        return [EvalPair(**item) for item in json.loads(path.read_text(encoding="utf-8"))]
    pairs = generate()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [dataclasses.asdict(pair) for pair in pairs]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return pairs


def _ollama_generate(prompt: str) -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        # think: False is required for reasoning-capable models (e.g. qwen3.6) - without
        # it, output goes into a separate "thinking" field and "response" comes back empty.
        json={"model": "qwen3.6:27b", "prompt": prompt, "stream": False, "think": False},
        timeout=30,
    )
    return response.json()["response"].strip()


def _generate_pairs(store: QdrantListingStore) -> list[EvalPair]:
    # Eval pairs are built from whatever is currently ingested, by scrolling the raw
    # points back out of Qdrant (payload was stored as the full EnrichedListing on upsert).
    pairs = []
    query_failures = []
    for point in store.list_all(limit=1000):
        payload = point.payload
        listing = EnrichedListing(
            **{k: v for k, v in payload.items() if k != "external_id"}
            | {"external_id": payload["external_id"]}
        )
        try:
            pairs.append(generate_eval_query(listing, generate_fn=_ollama_generate))
        except Exception as exc:  # noqa: BLE001 - quarantine, don't crash the eval run
            query_failures.append((listing.external_id, str(exc)))

    print(f"Generated {len(pairs)} eval queries.")
    if query_failures:
        print(f"{len(query_failures)} listings failed to generate an eval query:")
        for external_id, error in query_failures:
            print(f"  {external_id}: {error}")
    return pairs


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    store = QdrantListingStore()
    embedder = MultilingualE5Embedder(device="mps")
    parser = OllamaQueryParser()

    pairs = load_or_generate_pairs(args.queries, generate=lambda: _generate_pairs(store))

    def search_fn(query: str) -> list[str]:
        results = search(
            query,
            parser=parser,
            embedder=embedder,
            store=store,
            limit=_K,
            strict=args.strict,
            flexible_price_margin=args.price_margin,
        )
        return [r.payload["external_id"] for r in results]

    scores = evaluate(pairs, search_fn=search_fn, k=_K)
    mode = "strict" if args.strict else f"flexible (price margin x{args.price_margin})"
    print(f"mode={mode} k={_K} queries={len(pairs)}")
    print(scores)


if __name__ == "__main__":
    main()

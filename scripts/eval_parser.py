"""Score the LLM query parser against the hand-labelled queries.

Usage:
  python scripts/eval_parser.py [dev|test|all]     # default: dev

Every prediction is saved to data/eval/results/ so errors can be inspected later.
"""

import json
import math
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from realestate.eval.parser_eval import FIELDS, GoldQuery, ParserScore, load_gold, score
from realestate.query.parser import OllamaQueryParser

LABELS = Path("data/eval/labels.csv")
RESULTS_DIR = Path("data/eval/results")


def _pct(value: float) -> str:
    return "  -  " if math.isnan(value) else f"{value:5.0%}"


def _print_score(title: str, result: ParserScore) -> None:
    print(f"\n{title}: {result.exact_queries}/{result.queries} queries fully correct")
    print(f"  {'field':20} {'precision':>9} {'recall':>7}  correct wrong missed invented")
    for name in FIELDS:
        s = result.fields[name]
        print(
            f"  {name:20} {_pct(s.precision):>9} {_pct(s.recall):>7}"
            f"  {s.correct:7} {s.wrong:5} {s.missed:6} {s.invented:8}"
        )


def _print_coverage(golds: list[GoldQuery]) -> None:
    with_gaps = [g for g in golds if g.unsupported]
    print(
        f"\nQueries with a hard requirement the parser cannot express: "
        f"{len(with_gaps)}/{len(golds)}"
    )
    for name, count in Counter(n for g in golds for n in g.unsupported).most_common():
        print(f"  {name:16} {count}")


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    golds = [g for g in load_gold(LABELS) if split == "all" or g.split == split]
    parser = OllamaQueryParser()

    predictions: dict[str, dict] = {}
    records = []
    for gold in golds:
        started = time.perf_counter()
        parsed = parser.parse(gold.query)
        seconds = time.perf_counter() - started
        predictions[gold.id] = parsed.filters
        records.append(
            {
                "id": gold.id,
                "source": gold.source,
                "query": gold.query,
                "expected": gold.expected,
                "predicted": parsed.filters,
                "semantic_text": parsed.semantic_text,
                "seconds": round(seconds, 2),
            }
        )
        print(f"{gold.id} {seconds:5.1f}s {parsed.filters}", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"parser_baseline_{split}_{stamp}.json"
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")

    for source in ("user", "generated"):
        subset = [g for g in golds if g.source == source]
        if subset:
            _print_score(f"{source} queries", score(subset, predictions))
    _print_score("all queries", score(golds, predictions))
    _print_coverage(golds)
    times = sorted(r["seconds"] for r in records)
    print(f"\nSeconds per query: median {times[len(times) // 2]}, max {times[-1]}")
    print(f"Saved predictions to {out_path}")

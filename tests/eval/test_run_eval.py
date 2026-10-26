from pathlib import Path

import pytest

from realestate.eval.generate_queries import EvalPair
from realestate.eval.run_eval import load_or_generate_pairs, parse_args
from realestate.query.retrieval import DEFAULT_FLEXIBLE_PRICE_MARGIN

PAIRS = [
    EvalPair(query="garsoniera luminoasa sector 2", relevant_listing_id="a1"),
    EvalPair(query="apartament 3 camere lângă metrou", relevant_listing_id="b2"),
]


def test_parse_args_defaults_to_flexible_with_the_default_margin():
    args = parse_args([])

    assert args.strict is False
    assert args.price_margin == DEFAULT_FLEXIBLE_PRICE_MARGIN
    assert args.queries is None


def test_parse_args_reads_strict_margin_and_queries_path():
    args = parse_args(["--strict", "--price-margin", "1.3", "--queries", "q.json"])

    assert args.strict is True
    assert args.price_margin == 1.3
    assert args.queries == Path("q.json")


@pytest.mark.parametrize("bad_margin", ["0.9", "-1", "abc"])
def test_parse_args_rejects_price_margin_below_one_or_non_numeric(bad_margin):
    with pytest.raises(SystemExit):
        parse_args(["--price-margin", bad_margin])


def test_load_or_generate_pairs_generates_and_saves_when_file_is_missing(tmp_path: Path):
    path = tmp_path / "nested" / "queries.json"

    pairs = load_or_generate_pairs(path, generate=lambda: PAIRS)

    assert pairs == PAIRS
    assert path.exists()


def test_load_or_generate_pairs_reuses_saved_queries_without_regenerating(tmp_path: Path):
    path = tmp_path / "queries.json"
    load_or_generate_pairs(path, generate=lambda: PAIRS)

    def fail_if_called() -> list[EvalPair]:
        raise AssertionError("saved queries should be reused, not regenerated")

    assert load_or_generate_pairs(path, generate=fail_if_called) == PAIRS


def test_load_or_generate_pairs_without_a_path_just_generates():
    assert load_or_generate_pairs(None, generate=lambda: PAIRS) == PAIRS

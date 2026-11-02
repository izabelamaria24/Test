import csv
from pathlib import Path

import pytest

from realestate.eval.parser_eval import GoldQuery, load_gold, score

_COLUMNS = [
    "id",
    "source",
    "split",
    "query",
    "listing_type",
    "price_min",
    "price_max",
    "currency",
    "rooms_min",
    "rooms_max",
    "area_min_m2",
    "sectors",
    "neighbourhoods",
    "near_place",
    "max_walk_minutes",
    "floor_rule",
    "furnished",
    "built_after",
    "parking",
    "pets",
    "other_hard",
    "soft_wishes",
    "no_valid_answer",
    "notes",
    "labelled_by",
]


def _write(tmp_path: Path, *rows: dict) -> Path:
    path = tmp_path / "labels.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({"source": "user", "split": "dev", "query": "q", **row})
    return path


def test_loads_the_fields_the_parser_can_express(tmp_path):
    path = _write(
        tmp_path,
        {
            "id": "u1",
            "listing_type": "Buy",
            "price_max": "120000",
            "currency": "eur",
            "rooms_min": "2",
            "rooms_max": "2",
            "sectors": "6",
            "near_place": "metrou Lujerului",
            "max_walk_minutes": "10 minute",
        },
    )

    (gold,) = load_gold(path)

    assert gold.expected == {
        "listing_type": "sale",
        "price_max": 120000.0,
        "rooms": 2,
        "sector": 6,
        "max_subway_minutes": 10.0,
    }


def test_lei_budgets_are_expected_in_euros(tmp_path):
    (gold,) = load_gold(_write(tmp_path, {"id": "u1", "price_max": "2500", "currency": "RON"}))

    assert gold.expected["price_max"] == 500.0
    assert gold.price_tolerance > 0


def test_records_hard_constraints_the_parser_cannot_express(tmp_path):
    (gold,) = load_gold(
        _write(
            tmp_path,
            {
                "id": "u1",
                "rooms_min": "4",
                "sectors": "1;4",
                "neighbourhoods": "Titan, Dristor",
                "near_place": "Parcul IOR",
                "max_walk_minutes": "7",
                "pets": "yes",
                "soft_wishes": "bright",
            },
        )
    )

    assert "rooms" not in gold.expected
    assert "sector" not in gold.expected
    assert "max_subway_minutes" not in gold.expected
    assert set(gold.unsupported) == {
        "room range",
        "several sectors",
        "neighbourhoods",
        "near_place",
        "pets",
    }


def _gold(id: str = "q", **expected) -> GoldQuery:
    return GoldQuery(
        id=id, source="user", split="dev", query="q", expected=expected, unsupported=()
    )


def test_scores_correct_wrong_missed_and_invented_values():
    golds = [_gold("q1", rooms=2, price_max=500.0, sector=3), _gold("q2", rooms=1)]
    predictions = {"q1": {"rooms": 2, "price_max": "450", "listing_type": "rent"}, "q2": {}}

    result = score(golds, predictions)

    assert result.fields["rooms"].correct == 1
    assert result.fields["rooms"].missed == 1
    assert result.fields["price_max"].wrong == 1
    assert result.fields["sector"].missed == 1
    assert result.fields["listing_type"].invented == 1
    assert result.exact_queries == 0


def test_a_query_is_exact_when_every_field_matches_and_nothing_is_invented():
    result = score(
        [_gold(rooms=2, listing_type="rent")], {"q": {"rooms": "2", "listing_type": "rent"}}
    )

    assert result.exact_queries == 1
    assert result.fields["rooms"].precision == 1.0
    assert result.fields["rooms"].recall == 1.0


def test_lei_conversion_is_judged_with_a_tolerance():
    gold = GoldQuery(
        id="q",
        source="user",
        split="dev",
        query="q",
        expected={"price_max": 500.0},
        unsupported=(),
        price_tolerance=0.1,
    )

    assert score([gold], {"q": {"price_max": 520}}).fields["price_max"].correct == 1
    assert score([gold], {"q": {"price_max": 2500}}).fields["price_max"].wrong == 1


@pytest.mark.parametrize("bad", ["many", [2], True])
def test_unreadable_predicted_values_count_as_wrong(bad):
    result = score([_gold(rooms=2)], {"q": {"rooms": bad}})

    assert result.fields["rooms"].wrong == 1


def test_a_null_prediction_means_no_filter():
    result = score([_gold(rooms=2)], {"q": {"rooms": None}})

    assert result.fields["rooms"].missed == 1

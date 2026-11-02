"""Score a query parser's filters against hand-labelled queries (data/eval/labels.csv)."""

import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

FIELDS = ("listing_type", "price_max", "rooms", "sector", "max_subway_minutes")
# Approximate rate; lei budgets are scored with a tolerance rather than an exact amount.
_RON_PER_EUR = 5.0
_RON_TOLERANCE = 0.1
_LISTING_TYPES = {"buy": "sale", "sale": "sale", "sell": "sale", "rent": "rent"}
_ANY_METRO = "metro (any)"
_PARSER_CANNOT_EXPRESS = {
    "price_min": "price_min",
    "area_min_m2": "area_min_m2",
    "neighbourhoods": "neighbourhoods",
    "floor_rule": "floor_rule",
    "furnished": "furnished",
    "built_after": "built_after",
    "parking": "parking",
    "pets": "pets",
    "other_hard": "other_hard",
}


@dataclass(frozen=True)
class GoldQuery:
    id: str
    source: str
    split: str
    query: str
    expected: dict[str, object]
    # Hard constraints in the query that the parser's filter schema has no way to express.
    unsupported: tuple[str, ...]
    price_tolerance: float = 0.0


def _number(text: str) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", text.replace(" ", ""))
    return float(match.group()) if match else None


def _items(text: str) -> list[str]:
    return [item.strip() for item in re.split(r"[;,]", text) if item.strip()]


def _gold_from_row(row: dict[str, str]) -> GoldQuery:
    expected: dict[str, object] = {}
    unsupported = [name for column, name in _PARSER_CANNOT_EXPRESS.items() if row[column].strip()]

    listing_type = _LISTING_TYPES.get(row["listing_type"].strip().lower())
    if listing_type:
        expected["listing_type"] = listing_type

    tolerance = 0.0
    price_max = _number(row["price_max"])
    if price_max is not None:
        if row["currency"].strip().upper() == "RON":
            price_max /= _RON_PER_EUR
            tolerance = _RON_TOLERANCE
        expected["price_max"] = price_max

    rooms_min, rooms_max = _number(row["rooms_min"]), _number(row["rooms_max"])
    if rooms_min is not None and rooms_min == rooms_max:
        expected["rooms"] = int(rooms_min)
    elif rooms_min is not None or rooms_max is not None:
        unsupported.append("room range")

    sectors = _items(row["sectors"])
    if len(sectors) == 1 and (sector := _number(sectors[0])) is not None:
        expected["sector"] = int(sector)
    elif len(sectors) > 1:
        unsupported.append("several sectors")

    places = _items(row["near_place"])
    minutes = _number(row["max_walk_minutes"])
    if minutes is not None and places and all("metro" in place.lower() for place in places):
        expected["max_subway_minutes"] = minutes
    if any(place.lower() != _ANY_METRO for place in places):
        unsupported.append("near_place")

    return GoldQuery(
        id=row["id"],
        source=row["source"],
        split=row["split"],
        query=row["query"],
        expected=expected,
        unsupported=tuple(unsupported),
        price_tolerance=tolerance,
    )


def load_gold(path: Path) -> list[GoldQuery]:
    with path.open(encoding="utf-8", newline="") as f:
        return [_gold_from_row(row) for row in csv.DictReader(f)]


@dataclass
class FieldScore:
    correct: int = 0
    wrong: int = 0
    missed: int = 0
    invented: int = 0

    @property
    def precision(self) -> float:
        predicted = self.correct + self.wrong + self.invented
        return self.correct / predicted if predicted else math.nan

    @property
    def recall(self) -> float:
        expected = self.correct + self.wrong + self.missed
        return self.correct / expected if expected else math.nan


@dataclass
class ParserScore:
    fields: dict[str, FieldScore] = field(default_factory=lambda: {f: FieldScore() for f in FIELDS})
    exact_queries: int = 0
    queries: int = 0


def _as_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _matches(name: str, expected: object, predicted: object, tolerance: float) -> bool:
    if name == "listing_type":
        return isinstance(predicted, str) and predicted.strip().lower() == expected
    number = _as_number(predicted)
    if number is None or not isinstance(expected, int | float):
        return False
    return abs(number - expected) <= tolerance * expected


def score(golds: list[GoldQuery], predictions: dict[str, dict]) -> ParserScore:
    result = ParserScore(queries=len(golds))
    for gold in golds:
        predicted = {k: v for k, v in predictions.get(gold.id, {}).items() if v is not None}
        is_exact = True
        for name in FIELDS:
            counts = result.fields[name]
            has_expected, has_predicted = name in gold.expected, name in predicted
            if has_expected and has_predicted:
                tolerance = gold.price_tolerance if name == "price_max" else 0.0
                if _matches(name, gold.expected[name], predicted[name], tolerance):
                    counts.correct += 1
                else:
                    counts.wrong += 1
                    is_exact = False
            elif has_expected:
                counts.missed += 1
                is_exact = False
            elif has_predicted:
                counts.invented += 1
                is_exact = False
        result.exact_queries += is_exact
    return result

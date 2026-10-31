from dataclasses import dataclass

import pytest

from realestate.query.parser import ParsedQuery
from realestate.query.retrieval import SearchResult, search


class FakeParser:
    def __init__(self, parsed: ParsedQuery):
        self._parsed = parsed

    def parse(self, query: str) -> ParsedQuery:
        return self._parsed


class FakeEmbedder:
    def __init__(self):
        self.calls = 0

    def embed_passage(self, text: str) -> list[float]:
        raise AssertionError("search must only embed the query")

    def embed_query(self, text: str) -> list[float]:
        assert text == "bright hardwood floors"
        self.calls += 1
        return [0.2] * 768


@dataclass
class FakePoint:
    id: str
    score: float
    payload: dict


class FakeStore:
    def __init__(self, responses: list[list[FakePoint]] | None = None):
        self.last_call_kwargs = None
        self.calls: list[dict] = []
        self.responses = responses or [[]]

    def query(self, vector, **kwargs):
        self.last_call_kwargs = kwargs
        self.calls.append({"vector": vector, **kwargs})
        return self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]

    def upsert(self, external_id: str, vector: list[float], payload: dict) -> None:
        raise AssertionError("search must not modify listings")

    def list_all(self, limit: int = 1000) -> list:
        raise AssertionError("search must not scan all listings")


def _search(filters: dict, *, strict: bool) -> dict:
    store = FakeStore()
    search(
        "query",
        parser=FakeParser(ParsedQuery(filters=filters, semantic_text="bright hardwood floors")),
        embedder=FakeEmbedder(),
        store=store,
        limit=5,
        strict=strict,
    )
    assert store.last_call_kwargs is not None
    return store.last_call_kwargs


def test_search_passes_parsed_filters_and_embedded_semantic_text_to_store():
    parsed = ParsedQuery(
        filters={"price_max": 150000, "max_subway_minutes": 15},
        semantic_text="bright hardwood floors",
    )
    store = FakeStore()

    results = search(
        "bright apartment with hardwood floors, max 150000 EUR, 15 min to subway",
        parser=FakeParser(parsed),
        embedder=FakeEmbedder(),
        store=store,
        limit=5,
        strict=True,
    )

    assert results == []
    assert store.last_call_kwargs == {
        "price_max": 150000,
        "rooms": None,
        "sectors": None,
        "max_subway_minutes": 15,
        "listing_type": None,
        "limit": 5,
    }


def test_search_filters_by_the_listing_type_in_the_query():
    assert _search({"listing_type": "rent"}, strict=True)["listing_type"] == "rent"


def test_explicit_listing_type_overrides_the_query_and_applies_to_the_top_up():
    store = FakeStore()

    search(
        "apartament de vanzare",
        parser=FakeParser(
            ParsedQuery(
                filters={"listing_type": "sale", "sector": 2},
                semantic_text="bright hardwood floors",
            )
        ),
        embedder=FakeEmbedder(),
        store=store,
        listing_type="rent",
    )

    assert [call["listing_type"] for call in store.calls] == ["rent", "rent"]


def test_strict_search_uses_exact_sector_and_price():
    kwargs = _search({"price_max": 100000, "sector": 2}, strict=True)

    assert kwargs["sectors"] == [2]
    assert kwargs["price_max"] == 100000


def test_flexible_search_widens_price_and_adds_neighbouring_sectors():
    kwargs = _search({"price_max": 100000, "sector": 2}, strict=False)

    assert kwargs["sectors"] == [1, 2, 3]
    assert kwargs["price_max"] == pytest.approx(115000)


def test_flexible_search_wraps_around_the_sector_ring():
    assert _search({"sector": 1}, strict=False)["sectors"] == [1, 2, 6]
    assert _search({"sector": 6}, strict=False)["sectors"] == [1, 5, 6]


def test_flexible_search_keeps_rooms_exact():
    assert _search({"rooms": 2}, strict=False)["rooms"] == 2


def _search_with_margin(*, strict: bool, margin: float) -> dict:
    store = FakeStore()
    search(
        "query",
        parser=FakeParser(
            ParsedQuery(filters={"price_max": 100000}, semantic_text="bright hardwood floors")
        ),
        embedder=FakeEmbedder(),
        store=store,
        strict=strict,
        flexible_price_margin=margin,
    )
    assert store.last_call_kwargs is not None
    return store.last_call_kwargs


def test_flexible_search_uses_the_given_price_margin():
    assert _search_with_margin(strict=False, margin=1.3)["price_max"] == pytest.approx(130000)


def test_strict_search_ignores_the_price_margin():
    assert _search_with_margin(strict=True, margin=1.3)["price_max"] == 100000


@pytest.mark.parametrize("strict", [True, False])
def test_search_without_sector_or_price_sends_no_such_filters(strict):
    kwargs = _search({}, strict=strict)

    assert kwargs["sectors"] is None
    assert kwargs["price_max"] is None


def test_search_defaults_to_flexible():
    store = FakeStore()
    search(
        "query",
        parser=FakeParser(
            ParsedQuery(filters={"sector": 4}, semantic_text="bright hardwood floors")
        ),
        embedder=FakeEmbedder(),
        store=store,
    )

    assert store.last_call_kwargs is not None
    assert store.last_call_kwargs["sectors"] == [3, 4, 5]


def _point(
    point_id: str, *, price: float = 100000, sector: int = 2, score: float = 0.9
) -> FakePoint:
    return FakePoint(
        id=point_id,
        score=score,
        payload={"external_id": point_id, "price_eur": price, "sector": sector},
    )


def _search_results(
    store: FakeStore,
    *,
    filters: dict | None = None,
    strict: bool = False,
    limit: int = 3,
    margin: float = 1.15,
    embedder: FakeEmbedder | None = None,
) -> list[SearchResult]:
    return search(
        "query",
        parser=FakeParser(
            ParsedQuery(
                filters=filters if filters is not None else {"price_max": 100000, "sector": 2},
                semantic_text="bright hardwood floors",
            )
        ),
        embedder=embedder or FakeEmbedder(),
        store=store,
        limit=limit,
        strict=strict,
        flexible_price_margin=margin,
    )


def test_flexible_search_puts_exact_matches_before_higher_similarity_near_misses():
    exact = _point("exact", score=0.2)
    over_budget = _point("over-budget", price=106000, score=0.99)
    neighbouring = _point("neighbouring", sector=3, score=0.95)
    store = FakeStore([[exact], [over_budget, neighbouring, exact]])
    embedder = FakeEmbedder()

    results = _search_results(store, embedder=embedder)

    assert [r.id for r in results] == ["exact", "over-budget", "neighbouring"]
    assert [r.score for r in results] == [0.2, 0.99, 0.95]
    assert len(store.calls) == 2
    assert store.calls[0] == {
        "vector": [0.2] * 768,
        "price_max": 100000,
        "rooms": None,
        "sectors": [2],
        "max_subway_minutes": None,
        "listing_type": None,
        "limit": 3,
    }
    assert store.calls[1]["price_max"] == pytest.approx(115000)
    assert store.calls[1]["sectors"] == [1, 2, 3]
    assert embedder.calls == 1


def test_search_labels_exact_and_flexible_results_without_changing_payloads():
    exact = _point("exact")
    over_budget = _point("over-budget", price=106000)
    neighbouring = _point("neighbouring", sector=3)
    both = _point("both", price=110000, sector=1)
    store = FakeStore([[exact], [over_budget, neighbouring, both, exact]])

    results = _search_results(store, limit=4)

    assert [r.is_exact_match for r in results] == [True, False, False, False]
    assert [r.match_labels for r in results] == [
        [],
        ["+6% over budget"],
        ["sector 3 (neighbouring)"],
        ["+10% over budget", "sector 1 (neighbouring)"],
    ]
    assert results[0].payload == exact.payload
    assert "match_labels" not in exact.payload


def test_flexible_search_skips_top_up_when_exact_results_fill_the_limit():
    store = FakeStore([[_point("one"), _point("two")]])

    results = _search_results(store, limit=2)

    assert [r.id for r in results] == ["one", "two"]
    assert len(store.calls) == 1
    assert store.calls[0]["price_max"] == 100000
    assert store.calls[0]["sectors"] == [2]
    assert all(r.is_exact_match for r in results)


def test_strict_search_never_tops_up_an_underfilled_result_set():
    store = FakeStore([[_point("exact")], [_point("near", price=106000)]])

    results = _search_results(store, strict=True)

    assert [r.id for r in results] == ["exact"]
    assert results[0].is_exact_match is True
    assert results[0].match_labels == []
    assert len(store.calls) == 1


def test_top_up_requests_the_full_limit_to_allow_for_overlapping_exact_results():
    exact = _point("exact")
    near_one = _point("one", price=101000)
    near_two = _point("two", sector=3)
    store = FakeStore([[exact], [exact, near_one, near_two]])

    results = _search_results(store)

    assert [r.id for r in results] == ["exact", "one", "two"]
    assert store.calls[1]["limit"] == 3


def test_top_up_deduplicates_by_point_id_even_for_different_result_objects():
    store = FakeStore(
        [[_point("exact")], [_point("exact", score=0.8), _point("near", price=101000)]]
    )

    results = _search_results(store)

    assert [r.id for r in results] == ["exact", "near"]


def test_top_up_handles_no_exact_matches_and_fewer_candidates_than_requested():
    store = FakeStore([[], [_point("near", sector=3)]])

    results = _search_results(store)

    assert [r.id for r in results] == ["near"]
    assert results[0].match_labels == ["sector 3 (neighbouring)"]


def test_new_exact_hits_from_the_wider_search_still_appear_before_near_misses():
    store = FakeStore([[_point("exact")], [_point("near", price=101000), _point("other-exact")]])

    results = _search_results(store)

    assert [r.id for r in results] == ["exact", "other-exact", "near"]
    assert [r.is_exact_match for r in results] == [True, True, False]


def test_top_up_preserves_hard_room_and_subway_filters():
    store = FakeStore()

    _search_results(
        store,
        filters={"price_max": 100000, "sector": 2, "rooms": 2, "max_subway_minutes": 15},
    )

    assert len(store.calls) == 2
    assert all(call["rooms"] == 2 for call in store.calls)
    assert all(call["max_subway_minutes"] == 15 for call in store.calls)


@pytest.mark.parametrize("filters", [{}, {"rooms": 2}, {"max_subway_minutes": 15}])
def test_search_without_relaxable_filters_does_not_repeat_the_same_search(filters):
    store = FakeStore()

    assert _search_results(store, filters=filters) == []
    assert len(store.calls) == 1


def test_search_without_price_widening_or_sector_does_not_repeat_the_same_search():
    store = FakeStore()

    assert _search_results(store, filters={"price_max": 100000}, margin=1.0) == []
    assert len(store.calls) == 1


def test_exact_budget_and_sector_boundaries_have_no_near_miss_labels():
    store = FakeStore([[_point("boundary", price=100000, sector=2)]])

    result = _search_results(store, limit=1)[0]

    assert result.is_exact_match is True
    assert result.match_labels == []


def test_small_budget_overrun_does_not_get_rounded_to_zero_percent():
    store = FakeStore([[], [_point("near", price=100010)]])

    assert _search_results(store)[0].match_labels == ["+0.01% over budget"]


@pytest.mark.parametrize("limit", [0, -1])
def test_nonpositive_limit_returns_no_results_without_searching(limit):
    store = FakeStore()
    embedder = FakeEmbedder()

    assert _search_results(store, limit=limit, embedder=embedder) == []
    assert store.calls == []
    assert embedder.calls == 0

import pytest

from realestate.query.parser import ParsedQuery
from realestate.query.retrieval import search


class FakeParser:
    def __init__(self, parsed: ParsedQuery):
        self._parsed = parsed

    def parse(self, query: str) -> ParsedQuery:
        return self._parsed


class FakeEmbedder:
    def embed_query(self, text: str) -> list[float]:
        assert text == "bright hardwood floors"
        return [0.2] * 768


class FakeStore:
    def __init__(self):
        self.last_call_kwargs = None

    def query(self, vector, **kwargs):
        self.last_call_kwargs = kwargs
        return ["fake-result"]


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

    assert results == ["fake-result"]
    assert store.last_call_kwargs == {
        "price_max": 150000,
        "rooms": None,
        "sectors": None,
        "max_subway_minutes": 15,
        "limit": 5,
    }


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

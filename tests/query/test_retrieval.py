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
    )

    assert results == ["fake-result"]
    assert store.last_call_kwargs == {
        "price_max": 150000,
        "rooms": None,
        "max_subway_minutes": 15,
        "limit": 5,
    }

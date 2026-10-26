from dataclasses import dataclass, field

from fastapi.testclient import TestClient

from realestate.web.app import app, get_embedder, get_parser, get_store


@dataclass
class _ParsedQuery:
    filters: dict
    semantic_text: str


class _FakeParser:
    def __init__(self, filters: dict | None = None):
        self._filters = filters or {}

    def parse(self, query: str) -> _ParsedQuery:
        return _ParsedQuery(filters=self._filters, semantic_text=query)


class _FakeEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [0.0] * 768


@dataclass
class _FakeResult:
    score: float
    payload: dict


@dataclass
class _FakeStore:
    results: list = field(default_factory=list)
    last_sectors: list[int] | None = None

    def query(
        self, vector, *, price_max=None, rooms=None, sectors=None, max_subway_minutes=None, limit=10
    ):
        self.last_sectors = sectors
        return self.results


def _get(path: str, params: dict, *, parser: _FakeParser, store: _FakeStore):
    app.dependency_overrides[get_embedder] = lambda: _FakeEmbedder()
    app.dependency_overrides[get_parser] = lambda: parser
    app.dependency_overrides[get_store] = lambda: store
    try:
        return TestClient(app).get(path, params=params)
    finally:
        app.dependency_overrides.clear()


def test_api_search_maps_payload_fields_through_injected_dependencies():
    result = _FakeResult(
        score=0.9,
        payload={
            "title": "Garsoniera centrala",
            "price_eur": 75000,
            "rooms": 1,
            "built_area_sqm": 35.0,
            "address_text": "Sectorul 2",
            "sector": 2,
            "description": "luminoasa",
        },
    )
    response = _get(
        "/api/search",
        {"q": "garsoniera sector 2"},
        parser=_FakeParser(),
        store=_FakeStore(results=[result]),
    )

    assert response.status_code == 200
    body = response.json()
    assert body == [
        {
            "score": 0.9,
            "title": "Garsoniera centrala",
            "price_eur": 75000,
            "rooms": 1,
            "built_area_sqm": 35.0,
            "address_text": "Sectorul 2",
            "sector": 2,
            "description": "luminoasa",
        }
    ]


def test_api_search_returns_null_sector_for_listings_ingested_before_sector_existed():
    legacy = _FakeResult(
        score=0.5,
        payload={
            "title": "t",
            "price_eur": 1,
            "rooms": 1,
            "built_area_sqm": None,
            "address_text": "a",
            "description": "d",
        },
    )
    response = _get("/api/search", {"q": "x"}, parser=_FakeParser(), store=_FakeStore([legacy]))

    assert response.status_code == 200
    assert response.json()[0]["sector"] is None


def test_api_search_is_flexible_by_default_and_strict_when_asked():
    parser = _FakeParser(filters={"sector": 2})

    flexible_store = _FakeStore()
    _get("/api/search", {"q": "sector 2"}, parser=parser, store=flexible_store)
    strict_store = _FakeStore()
    _get("/api/search", {"q": "sector 2", "strict": "true"}, parser=parser, store=strict_store)

    assert flexible_store.last_sectors == [1, 2, 3]
    assert strict_store.last_sectors == [2]

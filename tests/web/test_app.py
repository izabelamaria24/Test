from dataclasses import dataclass, field

from fastapi.testclient import TestClient

from realestate.web.app import app, get_embedder, get_parser, get_store


@dataclass
class _ParsedQuery:
    filters: dict
    semantic_text: str


class _FakeParser:
    def parse(self, query: str) -> _ParsedQuery:
        return _ParsedQuery(filters={}, semantic_text=query)


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

    def query(self, vector, *, price_max=None, rooms=None, max_subway_minutes=None, limit=10):
        return self.results


def test_api_search_maps_payload_fields_through_injected_dependencies():
    result = _FakeResult(
        score=0.9,
        payload={
            "title": "Garsoniera centrala",
            "price_eur": 75000,
            "rooms": 1,
            "built_area_sqm": 35.0,
            "address_text": "Sectorul 2",
            "description": "luminoasa",
        },
    )
    app.dependency_overrides[get_embedder] = lambda: _FakeEmbedder()
    app.dependency_overrides[get_parser] = lambda: _FakeParser()
    app.dependency_overrides[get_store] = lambda: _FakeStore(results=[result])
    try:
        response = TestClient(app).get("/api/search", params={"q": "garsoniera sector 2"})
    finally:
        app.dependency_overrides.clear()

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
            "description": "luminoasa",
        }
    ]

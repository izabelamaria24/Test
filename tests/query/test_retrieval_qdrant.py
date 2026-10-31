"""Exercise exact-first retrieval against Qdrant's real filters without a server."""

from collections.abc import Iterator

import pytest
from qdrant_client import QdrantClient

from realestate.query.parser import ParsedQuery
from realestate.query.retrieval import search
from realestate.store.qdrant_store import QdrantListingStore


class _Parser:
    def parse(self, query: str) -> ParsedQuery:
        return ParsedQuery(
            filters={"price_max": 100000, "sector": 2, "rooms": 2, "max_subway_minutes": 15},
            semantic_text=query,
        )


class _Embedder:
    def embed_passage(self, text: str) -> list[float]:
        raise AssertionError("search must only embed the query")

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> Iterator[QdrantListingStore]:
    client = QdrantClient(":memory:")
    monkeypatch.setattr("realestate.store.qdrant_store.QdrantClient", lambda **kwargs: client)
    try:
        yield QdrantListingStore(vector_size=2)
    finally:
        client.close()


def _upsert(
    store: QdrantListingStore,
    external_id: str,
    *,
    vector: list[float] | None = None,
    price: float = 100000,
    sector: int = 2,
    rooms: int = 2,
    subway_minutes: float = 10,
    listing_type: str = "sale",
) -> None:
    store.upsert(
        external_id,
        vector or [1.0, 0.0],
        {
            "price_eur": price,
            "sector": sector,
            "rooms": rooms,
            "subway_walking_minutes": subway_minutes,
            "listing_type": listing_type,
        },
    )


def test_exact_first_retrieval_applies_real_qdrant_filters_and_similarity_order(store):
    _upsert(store, "exact", vector=[0.1, 0.9])
    _upsert(store, "over-budget", price=106000)
    _upsert(store, "neighbouring", vector=[0.9, 0.1], sector=3)
    _upsert(store, "too-expensive", price=116000)
    _upsert(store, "wrong-sector", sector=5)
    _upsert(store, "wrong-rooms", rooms=3)
    _upsert(store, "too-far", subway_minutes=20)

    results = search(
        "bright apartment", parser=_Parser(), embedder=_Embedder(), store=store, limit=5
    )

    assert [r.payload["external_id"] for r in results] == ["exact", "over-budget", "neighbouring"]
    assert results[0].score < results[1].score
    assert [r.match_labels for r in results] == [
        [],
        ["+6% over budget"],
        ["sector 3 (neighbouring)"],
    ]


def test_flexible_price_margin_includes_the_upper_budget_boundary(store):
    _upsert(store, "boundary", price=115000)

    results = search("bright apartment", parser=_Parser(), embedder=_Embedder(), store=store)

    assert [r.payload["external_id"] for r in results] == ["boundary"]
    assert results[0].match_labels == ["+15% over budget"]


def test_rentals_and_sales_never_mix(store):
    _upsert(store, "for-sale", listing_type="sale")
    _upsert(store, "for-rent", listing_type="rent")

    rentals = search(
        "bright apartment",
        parser=_Parser(),
        embedder=_Embedder(),
        store=store,
        listing_type="rent",
    )

    assert [r.payload["external_id"] for r in rentals] == ["for-rent"]

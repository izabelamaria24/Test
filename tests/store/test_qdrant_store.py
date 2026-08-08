import uuid

import pytest

from realestate.store.qdrant_store import QdrantListingStore

pytestmark = pytest.mark.integration


@pytest.fixture
def store():
    # Unique collection per test run so tests don't interfere with each other.
    return QdrantListingStore(collection=f"test_listings_{uuid.uuid4().hex[:8]}")


def test_upsert_and_query_returns_the_point(store):
    vector = [0.1] * 768
    store.upsert("111", vector, {"price_eur": 100000, "rooms": 1, "subway_walking_minutes": 5.0})

    results = store.query(vector, limit=5)

    assert len(results) == 1
    assert results[0].payload["external_id"] == "111"


def test_query_filters_out_listings_above_price_max(store):
    vector = [0.1] * 768
    store.upsert("cheap", vector, {"price_eur": 50000, "rooms": 1, "subway_walking_minutes": 5.0})
    store.upsert(
        "expensive", vector, {"price_eur": 500000, "rooms": 1, "subway_walking_minutes": 5.0}
    )

    results = store.query(vector, price_max=100000, limit=10)

    ids = [r.payload["external_id"] for r in results]
    assert "cheap" in ids
    assert "expensive" not in ids


def test_query_filters_by_max_subway_minutes(store):
    vector = [0.1] * 768
    store.upsert("close", vector, {"price_eur": 100000, "rooms": 1, "subway_walking_minutes": 5.0})
    store.upsert("far", vector, {"price_eur": 100000, "rooms": 1, "subway_walking_minutes": 40.0})

    results = store.query(vector, max_subway_minutes=15.0, limit=10)

    ids = [r.payload["external_id"] for r in results]
    assert "close" in ids
    assert "far" not in ids


def test_list_all_returns_upserted_points(store):
    vector = [0.1] * 768
    store.upsert("one", vector, {"price_eur": 100000, "rooms": 1, "subway_walking_minutes": 5.0})
    store.upsert("two", vector, {"price_eur": 200000, "rooms": 2, "subway_walking_minutes": 10.0})

    points = store.list_all()

    ids = [p.payload["external_id"] for p in points]
    assert "one" in ids
    assert "two" in ids

from realestate.embed.base import Embedder
from realestate.store.base import VectorStore


def search(
    query: str,
    *,
    parser,
    embedder: Embedder,
    store: VectorStore,
    limit: int = 10,
) -> list:
    parsed = parser.parse(query)
    vector = embedder.embed_query(parsed.semantic_text)
    return store.query(
        vector,
        price_max=parsed.filters.get("price_max"),
        rooms=parsed.filters.get("rooms"),
        max_subway_minutes=parsed.filters.get("max_subway_minutes"),
        limit=limit,
    )

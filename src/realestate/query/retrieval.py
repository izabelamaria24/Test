from realestate.embed.base import Embedder
from realestate.store.base import VectorStore

# Bucharest's sectors 1-6 form a clockwise ring; flexible mode also accepts each one's two neighbours.
_SECTOR_WITH_NEIGHBOURS: dict[int, tuple[int, int, int]] = {
    1: (1, 2, 6),
    2: (1, 2, 3),
    3: (2, 3, 4),
    4: (3, 4, 5),
    5: (4, 5, 6),
    6: (1, 5, 6),
}
_FLEXIBLE_PRICE_MARGIN = 1.15


def search(
    query: str,
    *,
    parser,
    embedder: Embedder,
    store: VectorStore,
    limit: int = 10,
    strict: bool = False,
) -> list:
    parsed = parser.parse(query)
    price_max = parsed.filters.get("price_max")
    sector = parsed.filters.get("sector")

    if strict:
        sectors = [sector] if sector is not None else None
    else:
        if price_max is not None:
            price_max *= _FLEXIBLE_PRICE_MARGIN
        sectors = list(_SECTOR_WITH_NEIGHBOURS[sector]) if sector is not None else None

    vector = embedder.embed_query(parsed.semantic_text)
    return store.query(
        vector,
        price_max=price_max,
        rooms=parsed.filters.get("rooms"),
        sectors=sectors,
        max_subway_minutes=parsed.filters.get("max_subway_minutes"),
        limit=limit,
    )

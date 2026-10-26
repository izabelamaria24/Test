from dataclasses import dataclass

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
DEFAULT_FLEXIBLE_PRICE_MARGIN = 1.15


@dataclass(frozen=True)
class SearchResult:
    id: int | str
    score: float
    payload: dict
    is_exact_match: bool
    match_labels: list[str]


def _match_labels(payload: dict, *, price_max: float | None, sector: int | None) -> list[str]:
    labels = []
    if price_max is not None and payload["price_eur"] > price_max:
        percent_over = (payload["price_eur"] - price_max) / price_max * 100
        labels.append(f"+{percent_over:.2g}% over budget")
    if sector is not None and payload.get("sector") != sector:
        labels.append(f"sector {payload['sector']} (neighbouring)")
    return labels


def search(
    query: str,
    *,
    parser,
    embedder: Embedder,
    store: VectorStore,
    limit: int = 10,
    strict: bool = False,
    flexible_price_margin: float = DEFAULT_FLEXIBLE_PRICE_MARGIN,
) -> list[SearchResult]:
    """Return exact matches first, optionally filling empty slots with labelled near-misses."""
    if limit <= 0:
        return []

    parsed = parser.parse(query)
    price_max = parsed.filters.get("price_max")
    sector = parsed.filters.get("sector")
    sectors = [sector] if sector is not None else None
    vector = embedder.embed_query(parsed.semantic_text)
    results = list(
        store.query(
            vector,
            price_max=price_max,
            rooms=parsed.filters.get("rooms"),
            sectors=sectors,
            max_subway_minutes=parsed.filters.get("max_subway_minutes"),
            limit=limit,
        )
    )

    # Round to cents so float drift doesn't exclude a listing exactly at the widened budget.
    flexible_price_max = (
        round(price_max * flexible_price_margin, 2) if price_max is not None else None
    )
    flexible_sectors = list(_SECTOR_WITH_NEIGHBOURS[sector]) if sector is not None else None
    has_wider_filters = flexible_price_max != price_max or flexible_sectors != sectors
    if not strict and len(results) < limit and has_wider_filters:
        # Request the full limit because the widened search can include every exact match again.
        candidates = store.query(
            vector,
            price_max=flexible_price_max,
            rooms=parsed.filters.get("rooms"),
            sectors=flexible_sectors,
            max_subway_minutes=parsed.filters.get("max_subway_minutes"),
            limit=limit,
        )
        seen_ids = {point.id for point in results}
        for point in candidates:
            if point.id not in seen_ids:
                results.append(point)
                seen_ids.add(point.id)

    annotated = []
    for point in results:
        labels = _match_labels(point.payload, price_max=price_max, sector=sector)
        annotated.append(
            SearchResult(
                id=point.id,
                score=point.score,
                payload=point.payload,
                is_exact_match=not labels,
                match_labels=labels,
            )
        )
    # Keep similarity order within each tier, including any new exact hits from the wider query.
    annotated.sort(key=lambda result: not result.is_exact_match)
    return annotated[:limit]

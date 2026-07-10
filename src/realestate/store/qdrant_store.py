import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)


def _external_id_to_point_id(external_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, external_id))


class QdrantListingStore:
    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection: str = "listings",
        vector_size: int = 768,
    ):
        self._client = QdrantClient(url=url)
        self._collection = collection
        if not self._client.collection_exists(collection):
            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def upsert(self, external_id: str, vector: list[float], payload: dict) -> None:
        self._client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(
                    id=_external_id_to_point_id(external_id),
                    vector=vector,
                    payload={**payload, "external_id": external_id},
                )
            ],
        )

    def query(
        self,
        vector: list[float],
        *,
        price_max: float | None = None,
        rooms: int | None = None,
        max_subway_minutes: float | None = None,
        limit: int = 10,
    ) -> list:
        must = []
        if price_max is not None:
            must.append(FieldCondition(key="price_eur", range=Range(lte=price_max)))
        if rooms is not None:
            must.append(FieldCondition(key="rooms", match=MatchValue(value=rooms)))
        if max_subway_minutes is not None:
            must.append(
                FieldCondition(key="subway_walking_minutes", range=Range(lte=max_subway_minutes))
            )
        query_filter = Filter(must=must) if must else None

        response = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            query_filter=query_filter,
            limit=limit,
        )
        return response.points

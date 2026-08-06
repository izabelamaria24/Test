from typing import Protocol


class VectorStore(Protocol):
    def upsert(self, external_id: str, vector: list[float], payload: dict) -> None: ...

    def query(
        self,
        vector: list[float],
        *,
        price_max: float | None = None,
        rooms: int | None = None,
        max_subway_minutes: float | None = None,
        limit: int = 10,
    ) -> list: ...

    def list_all(self, limit: int = 1000) -> list: ...

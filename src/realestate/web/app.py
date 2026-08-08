"""Minimal demo web server: a search box over the ingested listings.

Usage: uvicorn realestate.web.app:app --reload
"""

import os
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from realestate.embed.sentence_embedder import MultilingualE5Embedder
from realestate.query.parser import OllamaQueryParser
from realestate.query.retrieval import search
from realestate.store.qdrant_store import QdrantListingStore

app = FastAPI()

_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@lru_cache
def get_embedder() -> MultilingualE5Embedder:
    return MultilingualE5Embedder(device=os.getenv("REALESTATE_EMBEDDER_DEVICE", "cpu"))


@lru_cache
def get_parser() -> OllamaQueryParser:
    return OllamaQueryParser()


@lru_cache
def get_store() -> QdrantListingStore:
    return QdrantListingStore()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_static_dir / "index.html")


@app.get("/api/search")
def api_search(
    q: str,
    limit: int = 3,
    embedder: MultilingualE5Embedder = Depends(get_embedder),
    parser: OllamaQueryParser = Depends(get_parser),
    store: QdrantListingStore = Depends(get_store),
) -> list[dict]:
    results = search(q, parser=parser, embedder=embedder, store=store, limit=limit)
    return [
        {
            "score": r.score,
            "title": r.payload["title"],
            "price_eur": r.payload["price_eur"],
            "rooms": r.payload["rooms"],
            "built_area_sqm": r.payload["built_area_sqm"],
            "address_text": r.payload["address_text"],
            "description": r.payload["description"],
        }
        for r in results
    ]

"""Minimal demo web server: a search box over the ingested listings.

Usage: uvicorn realestate.web.app:app --reload
"""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from realestate.embed.sentence_embedder import MultilingualE5Embedder
from realestate.query.parser import OllamaQueryParser
from realestate.query.retrieval import search
from realestate.store.qdrant_store import QdrantListingStore

app = FastAPI()

_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_static_dir), name="static")

_embedder = MultilingualE5Embedder(device=os.getenv("REALESTATE_EMBEDDER_DEVICE", "cpu"))
_parser = OllamaQueryParser()
_store = QdrantListingStore()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_static_dir / "index.html")


@app.get("/api/search")
def api_search(q: str, limit: int = 3) -> list[dict]:
    results = search(q, parser=_parser, embedder=_embedder, store=_store, limit=limit)
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

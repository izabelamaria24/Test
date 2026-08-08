"""Build an eval set from currently ingested listings and score the live retrieval pipeline.

Usage: python -m realestate.eval.run_eval
"""

from realestate.embed.sentence_embedder import MultilingualE5Embedder
from realestate.eval.generate_queries import generate_eval_query
from realestate.eval.metrics import evaluate
from realestate.query.parser import OllamaQueryParser
from realestate.query.retrieval import search
from realestate.store.qdrant_store import QdrantListingStore

if __name__ == "__main__":
    store = QdrantListingStore()
    embedder = MultilingualE5Embedder(device="mps")
    parser = OllamaQueryParser()

    # Eval pairs are built from whatever is currently ingested, by scrolling the raw
    # points back out of Qdrant (payload was stored as the full EnrichedListing on upsert).
    all_points = store.list_all(limit=1000)

    def ollama_generate(prompt: str) -> str:
        import requests

        response = requests.post(
            "http://localhost:11434/api/generate",
            # think: False is required for reasoning-capable models (e.g. qwen3.6) - without
            # it, output goes into a separate "thinking" field and "response" comes back empty.
            json={"model": "qwen3.6:27b", "prompt": prompt, "stream": False, "think": False},
            timeout=30,
        )
        return response.json()["response"].strip()

    pairs = []
    query_failures = []
    for point in all_points:
        payload = point.payload
        from realestate.models import EnrichedListing

        listing = EnrichedListing(
            **{k: v for k, v in payload.items() if k != "external_id"}
            | {"external_id": payload["external_id"]}
        )
        try:
            pairs.append(generate_eval_query(listing, generate_fn=ollama_generate))
        except Exception as exc:  # noqa: BLE001 - quarantine, don't crash the eval run
            query_failures.append((listing.external_id, str(exc)))

    print(f"Generated {len(pairs)} eval queries.")
    if query_failures:
        print(f"{len(query_failures)} listings failed to generate an eval query:")
        for external_id, error in query_failures:
            print(f"  {external_id}: {error}")

    def search_fn(query: str) -> list[str]:
        results = search(query, parser=parser, embedder=embedder, store=store, limit=10)
        return [r.payload["external_id"] for r in results]

    scores = evaluate(pairs, search_fn=search_fn, k=10)
    print(scores)

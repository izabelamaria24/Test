from sentence_transformers import SentenceTransformer


class MultilingualE5Embedder:
    """Wraps intfloat/multilingual-e5-base. Note the model requires 'query: '/'passage: '
    prefixes on input text — this is part of how it was trained, not an arbitrary choice."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base", device: str = "mps"):
        self._model = SentenceTransformer(model_name, device=device)

    def embed_passage(self, text: str) -> list[float]:
        return self._model.encode(f"passage: {text}", normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(f"query: {text}", normalize_embeddings=True).tolist()

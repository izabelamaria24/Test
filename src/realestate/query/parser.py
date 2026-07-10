import json
from typing import Callable

import requests
from pydantic import BaseModel, ValidationError


class ParsedQuery(BaseModel):
    filters: dict
    semantic_text: str


_PARSE_PROMPT = """Extract structured filters and the remaining semantic intent from this \
real estate search query. Respond with ONLY a JSON object of the form:
{{"filters": {{"price_max": <number or omit>, "rooms": <number or omit>, \
"max_subway_minutes": <number or omit>}}, "semantic_text": "<remaining descriptive text>"}}

Query: {query}
"""


def _default_post(url: str, json: dict, timeout: int):
    return requests.post(url, json=json, timeout=timeout)


class OllamaQueryParser:
    def __init__(
        self,
        model: str = "qwen3.6:27b",
        host: str = "http://localhost:11434",
        post_fn: Callable = _default_post,
    ):
        self._model = model
        self._host = host
        self._post_fn = post_fn

    def parse(self, query: str) -> ParsedQuery:
        try:
            response = self._post_fn(
                f"{self._host}/api/generate",
                json={
                    "model": self._model,
                    "prompt": _PARSE_PROMPT.format(query=query),
                    "format": "json",
                    "stream": False,
                },
                timeout=30,
            )
            response.raise_for_status()
            raw = json.loads(response.json()["response"])
            return ParsedQuery(**raw)
        except (requests.RequestException, ConnectionError, json.JSONDecodeError, ValidationError, KeyError, TypeError):
            return ParsedQuery(filters={}, semantic_text=query)

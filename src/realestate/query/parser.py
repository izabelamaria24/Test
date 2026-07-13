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

Only include a filter if the query states an actual number for it. Vague qualitative phrases \
like "close to the subway" or "near downtown" are NOT numbers - do not invent one. Omit the \
filter entirely in that case and leave the phrase in semantic_text instead.

Query: {query}
"""


def _default_post(url: str, json: dict, timeout: int):
    return requests.post(url, json=json, timeout=timeout)


def _drop_implausible_filters(filters: dict) -> dict:
    # A real constraint is never "0 or less" for any of these - price_max/rooms/
    # max_subway_minutes <= 0 means the model invented a threshold from vague language
    # (e.g. "close to the subway" -> max_subway_minutes: 0) rather than an actual stated number.
    return {key: value for key, value in filters.items() if not (isinstance(value, (int, float)) and value <= 0)}


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
                    # Reasoning-capable models (e.g. qwen3.6) default to putting output in a
                    # separate "thinking" field and leaving "response" empty unless told not to.
                    "think": False,
                },
                timeout=30,
            )
            response.raise_for_status()
            raw = json.loads(response.json()["response"])
            if isinstance(raw, dict):
                raw["filters"] = _drop_implausible_filters(raw.get("filters", {}))
            return ParsedQuery(**raw)
        except (requests.RequestException, ConnectionError, json.JSONDecodeError, ValidationError, KeyError, TypeError):
            return ParsedQuery(filters={}, semantic_text=query)

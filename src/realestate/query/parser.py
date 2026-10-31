import json
from collections.abc import Callable

import requests
from pydantic import BaseModel, ValidationError


class ParsedQuery(BaseModel):
    filters: dict
    semantic_text: str


_PARSE_PROMPT = """Extract structured filters and the remaining semantic intent from this \
real estate search query. Respond with ONLY a JSON object of the form:
{{"filters": {{"price_max": <number or omit>, "rooms": <number or omit>, \
"sector": <1-6 or omit>, "max_subway_minutes": <number or omit>, \
"listing_type": <"sale" or "rent" or omit>}}, \
"semantic_text": "<remaining descriptive text>"}}

Only include a filter if the query states an actual number for it. Vague qualitative phrases \
like "close to the subway" or "near downtown" are NOT numbers - do not invent one. Omit the \
filter entirely in that case and leave the phrase in semantic_text instead.

Only set "sector" when the query names a Bucharest sector number (e.g. "sector 2", \
"sectorul 3"); otherwise omit it.

Set "listing_type" to "rent" when the query is about renting (e.g. "chirie", "de închiriat", \
"rent") and to "sale" when it is about buying (e.g. "de vânzare", "cumpăr", "buy"); \
otherwise omit it.

Query: {query}
"""


def _default_post(url: str, json: dict, timeout: int):
    return requests.post(url, json=json, timeout=timeout)


def _is_plausible_filter(key: str, value: object) -> bool:
    if key == "listing_type":
        return value in ("sale", "rent")
    if key == "sector":
        # Retrieval looks sectors up in a fixed 1-6 table, so anything else must not get through.
        return isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 6
    # A real constraint is never "0 or less" for any of these - price_max/rooms/
    # max_subway_minutes <= 0 means the model invented a threshold from vague language
    # (e.g. "close to the subway" -> max_subway_minutes: 0) rather than an actual stated number.
    return not (isinstance(value, (int, float)) and value <= 0)


def _drop_implausible_filters(filters: dict) -> dict:
    return {key: value for key, value in filters.items() if _is_plausible_filter(key, value)}


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
        except (
            requests.RequestException,
            ConnectionError,
            json.JSONDecodeError,
            ValidationError,
            KeyError,
            TypeError,
        ):
            return ParsedQuery(filters={}, semantic_text=query)

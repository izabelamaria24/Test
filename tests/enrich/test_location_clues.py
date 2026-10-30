import json

import pytest
import requests

from realestate.enrich.location_clues import LocationClues, OllamaLocationClueExtractor


class _FakeResponse:
    def __init__(self, body: dict):
        self._body = body

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._body


def _extractor_returning(clues: object, calls: list[dict] | None = None):
    def post_fn(url: str, json: dict, timeout: int):
        if calls is not None:
            calls.append(json)
        return _FakeResponse({"response": _dumps(clues)})

    return OllamaLocationClueExtractor(post_fn=post_fn)


def _dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def test_extracts_location_clues_from_llm_json():
    extractor = _extractor_returning(
        {
            "street": "Strada Sibiu",
            "street_number": "12",
            "neighbourhood": "Drumul Taberei",
            "metro_station": "Râul Doamnei",
            "landmarks": ["Parcul Moghioroș"],
        }
    )

    clues = extractor.extract("Apartament 2 camere", "Strada Sibiu 12, langa parc")

    assert clues == LocationClues(
        street="Strada Sibiu",
        street_number="12",
        neighbourhood="Drumul Taberei",
        metro_station="Râul Doamnei",
        landmarks=["Parcul Moghioroș"],
    )


def test_sends_listing_text_with_json_format_and_thinking_disabled():
    calls: list[dict] = []
    extractor = _extractor_returning({}, calls)

    extractor.extract("Titlu", "x" * 5000)

    request = calls[0]
    assert request["format"] == "json"
    assert request["think"] is False
    assert "Titlu" in request["prompt"]
    assert "x" * 3000 in request["prompt"]
    assert "x" * 3001 not in request["prompt"]


def test_normalizes_loose_llm_output():
    extractor = _extractor_returning(
        {
            "street": "  ",
            "street_number": 65,
            "neighbourhood": "Sector 3",
            "metro_station": ["Brâncoveanu", "Piața Sudului"],
            "landmarks": ["Mall Vitan", "", 7, "Parcul IOR", "Lidl", "Piata"],
        }
    )

    clues = extractor.extract("t", "d")

    assert clues.street is None
    assert clues.street_number == "65"
    assert clues.neighbourhood is None
    assert clues.metro_station == "Brâncoveanu"
    assert clues.landmarks == ["Mall Vitan", "Parcul IOR", "Lidl"]


def test_drops_overlong_values():
    extractor = _extractor_returning({"street": "S" * 101, "landmarks": ["L" * 101]})

    clues = extractor.extract("t", "d")

    assert clues.street is None
    assert clues.landmarks == []


@pytest.mark.parametrize(
    "post_fn",
    [
        lambda url, json, timeout: _FakeResponse({"response": "not json"}),
        lambda url, json, timeout: _FakeResponse({"response": '["a list"]'}),
        lambda url, json, timeout: _FakeResponse({}),
    ],
)
def test_returns_no_clues_for_unusable_llm_output(post_fn):
    assert OllamaLocationClueExtractor(post_fn=post_fn).extract("t", "d") == LocationClues()


def test_returns_no_clues_when_ollama_is_unreachable():
    def post_fn(url, json, timeout):
        raise requests.ConnectionError("down")

    assert OllamaLocationClueExtractor(post_fn=post_fn).extract("t", "d") == LocationClues()

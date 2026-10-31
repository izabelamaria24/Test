import json as json_module

import pytest

from realestate.query.parser import OllamaQueryParser


class FakeResponse:
    def __init__(self, json_body):
        self._json_body = json_body

    def raise_for_status(self):
        pass

    def json(self):
        return self._json_body


def test_parse_extracts_filters_and_semantic_text():
    def fake_post(url, json, timeout):
        return FakeResponse(
            {
                "response": (
                    '{"filters": {"price_max": 150000, "max_subway_minutes": 15}, '
                    '"semantic_text": "bright pet-friendly hardwood floors"}'
                )
            }
        )

    parser = OllamaQueryParser(post_fn=fake_post)
    result = parser.parse(
        "bright pet-friendly apartment with hardwood floors, "
        "max 150000 EUR, max 15 min walk to subway"
    )

    assert result.filters["price_max"] == 150000
    assert result.filters["max_subway_minutes"] == 15
    assert result.semantic_text == "bright pet-friendly hardwood floors"


def test_parse_falls_back_to_pure_semantic_text_on_invalid_json():
    def fake_post(url, json, timeout):
        return FakeResponse({"response": "not valid json"})

    parser = OllamaQueryParser(post_fn=fake_post)
    result = parser.parse("some query")

    assert result.filters == {}
    assert result.semantic_text == "some query"


def test_parse_falls_back_when_request_raises():
    def fake_post(url, json, timeout):
        raise ConnectionError("Ollama not running")

    parser = OllamaQueryParser(post_fn=fake_post)
    result = parser.parse("some query")

    assert result.filters == {}
    assert result.semantic_text == "some query"


def test_parse_falls_back_when_json_is_not_a_dict_number():
    def fake_post(url, json, timeout):
        return FakeResponse({"response": "42"})

    parser = OllamaQueryParser(post_fn=fake_post)
    result = parser.parse("some query")

    assert result.filters == {}
    assert result.semantic_text == "some query"


def test_parse_falls_back_when_json_is_not_a_dict_array():
    def fake_post(url, json, timeout):
        return FakeResponse({"response": "[1,2,3]"})

    parser = OllamaQueryParser(post_fn=fake_post)
    result = parser.parse("some query")

    assert result.filters == {}
    assert result.semantic_text == "some query"


def test_parse_drops_zero_subway_minutes_invented_from_vague_phrasing():
    def fake_post(url, json, timeout):
        return FakeResponse(
            {
                "response": (
                    '{"filters": {"price_max": 250000, "max_subway_minutes": 0}, '
                    '"semantic_text": "apartament luminos"}'
                )
            }
        )

    parser = OllamaQueryParser(post_fn=fake_post)
    result = parser.parse("apartament luminos, aproape de metrou, buget maxim 250000 euro")

    assert result.filters == {"price_max": 250000}
    assert "max_subway_minutes" not in result.filters


def test_parse_drops_negative_filter_values():
    def fake_post(url, json, timeout):
        return FakeResponse(
            {"response": '{"filters": {"rooms": -1, "price_max": 100000}, "semantic_text": "x"}'}
        )

    parser = OllamaQueryParser(post_fn=fake_post)
    result = parser.parse("x")

    assert result.filters == {"price_max": 100000}


def test_parse_keeps_positive_filter_values():
    def fake_post(url, json, timeout):
        return FakeResponse(
            {
                "response": (
                    '{"filters": {"price_max": 150000, "rooms": 2, "max_subway_minutes": 15}, '
                    '"semantic_text": "x"}'
                )
            }
        )

    parser = OllamaQueryParser(post_fn=fake_post)
    result = parser.parse("x")

    assert result.filters == {"price_max": 150000, "rooms": 2, "max_subway_minutes": 15}


def test_parse_extracts_bucharest_sector():
    def fake_post(url, json, timeout):
        return FakeResponse(
            {
                "response": (
                    '{"filters": {"price_max": 75000, "rooms": 1, "sector": 2}, '
                    '"semantic_text": "apartament"}'
                )
            }
        )

    parser = OllamaQueryParser(post_fn=fake_post)
    result = parser.parse("apartament o camera sub 75000 euro sector 2")

    assert result.filters["sector"] == 2


@pytest.mark.parametrize("bad_sector", [7, 0, "2", True, 2.5])
def test_parse_drops_sector_outside_1_to_6(bad_sector):
    def fake_post(url, json, timeout):
        payload = {"filters": {"price_max": 100000, "sector": bad_sector}, "semantic_text": "x"}
        return FakeResponse({"response": json_module.dumps(payload)})

    parser = OllamaQueryParser(post_fn=fake_post)
    result = parser.parse("x")

    assert result.filters == {"price_max": 100000}


@pytest.mark.parametrize(
    ("listing_type", "kept"), [("rent", True), ("sale", True), ("buy", False), (1, False)]
)
def test_parse_keeps_only_sale_or_rent_listing_type(listing_type, kept):
    def fake_post(url, json, timeout):
        payload = {"filters": {"listing_type": listing_type}, "semantic_text": "x"}
        return FakeResponse({"response": json_module.dumps(payload)})

    result = OllamaQueryParser(post_fn=fake_post).parse("apartament de inchiriat")

    assert result.filters == ({"listing_type": listing_type} if kept else {})


def test_parse_prompt_asks_for_listing_type():
    requests_sent = []

    def fake_post(url, json, timeout):
        requests_sent.append(json)
        return FakeResponse({"response": '{"filters": {}, "semantic_text": "x"}'})

    OllamaQueryParser(post_fn=fake_post).parse("x")

    assert '"listing_type"' in requests_sent[0]["prompt"]

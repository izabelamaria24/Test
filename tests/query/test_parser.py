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

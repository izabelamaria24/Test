import copy
import json

import pytest

from realestate.ingest.olx_parser import parse_olx_listing_html

# Fields below mirror the real structure of window.__PRERENDERED_STATE__ verified against
# a live OLX.ro listing page — not a guessed schema. Building the fixture via json.dumps
# (rather than hand-writing escaped JSON) keeps the double-encoding correct.
SAMPLE_AD = {
    "id": 304473136,
    "title": "Vand apartament 2 camere TITAN",
    "description": (
        "Direct proprietar !<br />\nVand apartament 2 camere decomandat<br />\n"
        "* Suprafata utila 48 utili"
    ),
    "price": {"regularPrice": {"value": 100000, "currencyCode": "EUR", "currencySymbol": "€"}},
    "location": {"pathName": "Bucuresti - Ilfov, Bucuresti, Sectorul 3"},
    "params": [
        {"key": "compartimentare", "name": "Compartimentare", "value": "Decomandat"},
        {"key": "m", "name": "Suprafata utila", "value": "48 m²"},
        {"key": "constructie", "name": "An constructie", "value": "1977 – 1990"},
        {"key": "floor", "name": "Etaj", "value": "3"},
    ],
    "photos": [
        "https://frankfurt.apollo.olxcdn.com:443/v1/files/dlbik2gpbb3j1-RO/image;s=750x1000"
    ],
    "map": {"lat": 44.42, "lon": 26.1, "radius": 3, "show_detailed": False, "zoom": 12},
}


def _build_sample_html(ad_data: dict) -> str:
    inner_json_text = json.dumps({"ad": {"ad": ad_data}}, ensure_ascii=False)
    js_string_literal = json.dumps(inner_json_text, ensure_ascii=False)
    return (
        "<html><head><script>window.__PRERENDERED_STATE__= "
        f"{js_string_literal};\n</script></head><body></body></html>"
    )


SAMPLE_HTML = _build_sample_html(SAMPLE_AD)


def test_parses_price():
    listing = parse_olx_listing_html(SAMPLE_HTML, external_id="304473136")
    assert listing.price_raw == "100.000 €"


def test_parses_title_and_strips_html_from_description():
    listing = parse_olx_listing_html(SAMPLE_HTML, external_id="304473136")
    assert listing.title == "Vand apartament 2 camere TITAN"
    assert "Suprafata utila 48 utili" in listing.description
    assert "<br" not in listing.description


def test_parses_address_from_location_pathname():
    listing = parse_olx_listing_html(SAMPLE_HTML, external_id="304473136")
    assert listing.address_raw == "Bucuresti - Ilfov, Bucuresti, Sectorul 3"


def test_parses_specs_by_display_name():
    listing = parse_olx_listing_html(SAMPLE_HTML, external_id="304473136")
    assert listing.specs["Suprafata utila"] == "48 m²"
    assert listing.specs["Etaj"] == "3"
    assert listing.specs["Compartimentare"] == "Decomandat"


def test_parses_image_urls():
    listing = parse_olx_listing_html(SAMPLE_HTML, external_id="304473136")
    assert listing.image_urls == [
        "https://frankfurt.apollo.olxcdn.com:443/v1/files/dlbik2gpbb3j1-RO/image;s=750x1000"
    ]


def test_raises_when_prerendered_state_missing():
    with pytest.raises(ValueError, match="PRERENDERED_STATE"):
        parse_olx_listing_html("<html><body>no state here</body></html>", external_id="000")


def test_parses_map_coordinates():
    listing = parse_olx_listing_html(SAMPLE_HTML, external_id="304473136")
    assert listing.map_lat == 44.42
    assert listing.map_lon == 26.1


def test_map_coordinates_default_to_none_when_absent():
    ad_without_map = {k: v for k, v in SAMPLE_AD.items() if k != "map"}
    html = _build_sample_html(ad_without_map)
    listing = parse_olx_listing_html(html, external_id="304473136")
    assert listing.map_lat is None
    assert listing.map_lon is None


def test_raises_when_state_missing_ad_key():
    # state JSON missing the "ad" key entirely
    inner_json_text = json.dumps({}, ensure_ascii=False)
    js_string_literal = json.dumps(inner_json_text, ensure_ascii=False)
    html = (
        "<html><head><script>window.__PRERENDERED_STATE__= "
        f"{js_string_literal};\n</script></head><body></body></html>"
    )
    with pytest.raises(ValueError, match="invalid state structure"):
        parse_olx_listing_html(html, external_id="304473136")


def test_raises_when_regular_price_missing_currency_symbol():
    # regularPrice present but missing currencySymbol
    ad_data = copy.deepcopy(SAMPLE_AD)
    ad_data["price"]["regularPrice"] = {"value": 100000, "currencyCode": "EUR"}
    html = _build_sample_html(ad_data)
    with pytest.raises(ValueError, match="invalid price structure"):
        parse_olx_listing_html(html, external_id="304473136")

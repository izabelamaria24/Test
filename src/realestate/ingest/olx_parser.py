import json

from bs4 import BeautifulSoup

from realestate.models import RawListing

_STATE_PREFIX = "window.__PRERENDERED_STATE__= "


def _extract_ad_json(html: str) -> dict:
    start = html.find(_STATE_PREFIX)
    if start == -1:
        raise ValueError("PRERENDERED_STATE script assignment not found")
    start += len(_STATE_PREFIX)
    end = html.find(";\n", start)
    if end == -1:
        raise ValueError("could not find end of PRERENDERED_STATE assignment")
    js_string_literal = html[start:end]
    inner_json_text = json.loads(js_string_literal)  # un-escape the JS string
    return json.loads(inner_json_text)  # parse the actual JSON payload


def _clean_description(raw_html_description: str) -> str:
    return BeautifulSoup(raw_html_description, "lxml").get_text(separator="\n").strip()


def parse_olx_listing_html(html: str, external_id: str) -> RawListing:
    state = _extract_ad_json(html)
    try:
        ad = state["ad"]["ad"]
    except KeyError as err:
        raise ValueError(f"listing {external_id}: invalid state structure - ad not found") from err

    price_info = ad.get("price", {}).get("regularPrice")
    if not price_info:
        raise ValueError(f"listing {external_id}: no regular price found")
    try:
        price_raw = f"{price_info['value']:,.0f} {price_info['currencySymbol']}".replace(",", ".")
    except KeyError as err:
        raise ValueError(
            f"listing {external_id}: invalid price structure - missing value or currencySymbol"
        ) from err

    specs = {param["name"]: param["value"] for param in ad.get("params", [])}
    map_data = ad.get("map") or {}

    return RawListing(
        external_id=external_id,
        title=ad.get("title", ""),
        description=_clean_description(ad.get("description", "")),
        price_raw=price_raw,
        address_raw=ad.get("location", {}).get("pathName", ""),
        specs=specs,
        image_urls=ad.get("photos", []),
        source_file=f"{external_id}.html",
        map_lat=map_data.get("lat"),
        map_lon=map_data.get("lon"),
    )

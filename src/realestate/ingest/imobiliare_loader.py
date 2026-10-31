"""Turn imobiliare.ro records (as returned by the Apify actor) into RawListings.

The actor's `latitude`/`longitude` were empty for every record checked (2026-10-31), so the
location comes from the structured `streetAddress` plus the usual text clues.
"""

import html
import json
import re
from pathlib import Path

from realestate.models import ListingType, RawListing

_LISTING_TYPES: dict[str, ListingType] = {"sell": "sale", "sale": "sale", "rent": "rent"}
# The actor reports a few floors in English; the spec normalizer understands the Romanian terms.
_FLOOR_TERMS = {"ground-floor": "parter", "basement": "demisol", "semi-basement": "demisol"}
_NUMBER = r"\d+[a-z]?(?:-\d+[a-z]?)?"
_LEADING_NUMBER = re.compile(rf"^(?:nr\.?\s*)?({_NUMBER}(?:\s[a-z](?=\s))?)\s+(.+)$", re.I)
_TRAILING_NUMBER = re.compile(rf"^(.+?)\s*(?:nr\.?\s*|\s)({_NUMBER})$", re.I)
_NUMBER_PART = re.compile(rf"^nr\.?\s*({_NUMBER})$", re.I)


def _text(value: object) -> str:
    return html.unescape(value).strip() if isinstance(value, str) else ""


def _neighbourhood_from_address(address: str) -> str:
    # "<street>, <neighbourhood>, Sector N": matched the `neighborhood` field on all 40 pilot ads.
    parts = [part.strip() for part in address.split(",") if part.strip()]
    if len(parts) >= 2 and parts[-1].casefold().startswith("sector"):
        return parts[-2]
    return ""


def split_street_address(street_address: str, neighbourhood: str) -> tuple[str | None, str | None]:
    """Street and building number from e.g. "2a valea argesului, Drumul Taberei, Sector 6"."""
    parts = [part.strip() for part in street_address.split(",") if part.strip()]
    if not parts:
        return None, None
    first = parts[0]
    if first.casefold() == neighbourhood.casefold() or first.casefold().startswith("sector"):
        return None, None
    if match := _LEADING_NUMBER.match(first):
        return match.group(2), match.group(1)
    if match := _TRAILING_NUMBER.match(first):
        return match.group(1), match.group(2)
    # "Strada Sergent Ilie Petre,nr 84": the number ended up in its own comma-separated part.
    if len(parts) > 1 and (match := _NUMBER_PART.match(parts[1])):
        return first, match.group(1)
    return first, None


def parse_imobiliare_record(record: dict, *, source_file: str) -> RawListing:
    listing_type = _LISTING_TYPES.get(record.get("operationType") or "")
    if listing_type is None:
        raise ValueError(f"unknown operationType {record.get('operationType')!r}")
    if record.get("priceCurrency") != "EUR" or not record.get("priceText"):
        raise ValueError(f"no EUR price: {record.get('priceText')!r} {record.get('priceCurrency')}")

    address = _text(record.get("streetAddress"))
    neighbourhood = _text(record.get("neighborhood")) or _neighbourhood_from_address(address)
    city = _text(record.get("city"))
    if city and city.casefold() not in address.casefold():
        address = f"{address}, {city}" if address else city
    street, street_number = split_street_address(address, neighbourhood)

    specs: dict[str, str] = {}
    if record.get("surfaceTotal") is not None:
        specs["Suprafata utila"] = f"{record['surfaceTotal']:g} m²"
    if record.get("floor") is not None:
        floor = str(record["floor"])
        specs["Etaj"] = _FLOOR_TERMS.get(floor, floor)
    if record.get("yearBuilt") is not None:
        specs["An constructie"] = str(record["yearBuilt"])

    return RawListing(
        external_id=f"imobiliare-{record['id']}",
        title=_text(record.get("title")),
        description=_text(record.get("descriptionText") or record.get("descriptionPreview")),
        price_raw=_text(record["priceText"]),
        address_raw=address,
        specs=specs,
        image_urls=list(record.get("images") or []),
        source_file=source_file,
        map_lat=None,
        map_lon=None,
        listing_type=listing_type,
        street=street,
        street_number=street_number,
        neighbourhood=neighbourhood or None,
    )


def load_imobiliare_directory(
    json_dir: Path,
) -> tuple[list[RawListing], list[tuple[Path, str]]]:
    # Later files win, so a listing fetched again replaces its older copy.
    listings: dict[str, RawListing] = {}
    failures: list[tuple[Path, str]] = []
    for path in sorted(json_dir.glob("*.json")):
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            failures.append((path, str(exc)))
            continue
        for record in records:
            try:
                listing = parse_imobiliare_record(record, source_file=path.name)
            except (ValueError, KeyError, TypeError) as exc:
                failures.append((path, f"{record.get('id')}: {exc}"))
                continue
            listings[listing.external_id] = listing
    return list(listings.values()), failures

import re
import unicodedata

SPEC_LABEL_MAP: dict[str, str] = {
    "Suprafata utila": "built_area_sqm",
    "Etaj": "floor_number",
    "An constructie": "construction_year_range",
    "Compartimentare": "layout_type",
}

_NUMERIC_FIELDS = {"built_area_sqm", "floor_number"}

_NON_NUMERIC_FLOOR_TERMS: dict[str, int] = {
    "parter": 0,
    "demisol": -1,
}


def normalize_price_eur(price_raw: str) -> float:
    match = re.search(r"([\d.]+)\s*€", price_raw)
    if not match:
        raise ValueError(f"unrecognized price format: {price_raw!r}")
    digits = match.group(1).replace(".", "")
    return float(digits)


def normalize_rooms(title: str) -> int | None:
    lowered = title.lower()
    match = re.search(r"(\d+)\s*camere", lowered)
    if match:
        return int(match.group(1))
    if "garsonier" in lowered:
        return 1
    return None


def normalize_specs(specs: dict[str, str]) -> dict[str, object]:
    normalized: dict[str, object] = {
        "built_area_sqm": None,
        "floor_number": None,
        "construction_year_range": None,
        "layout_type": None,
    }
    for raw_label, raw_value in specs.items():
        field = SPEC_LABEL_MAP.get(raw_label)
        if field is None:
            continue
        if field == "built_area_sqm":
            match = re.search(r"([\d.,]+)", raw_value)
            normalized[field] = float(match.group(1).replace(",", ".")) if match else None
        elif field == "floor_number":
            # Check for known non-numeric floor terms first
            normalized_raw = unicodedata.normalize("NFD", raw_value.lower()).encode("ascii", "ignore").decode()
            for term, floor_value in _NON_NUMERIC_FLOOR_TERMS.items():
                if term in normalized_raw:
                    normalized[field] = floor_value
                    break
            else:
                # Fall back to digit extraction
                match = re.search(r"(\d+)", raw_value)
                normalized[field] = int(match.group(1)) if match else None
        else:
            normalized[field] = raw_value
    return normalized

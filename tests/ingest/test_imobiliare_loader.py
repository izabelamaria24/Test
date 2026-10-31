import json
from pathlib import Path

import pytest

from realestate.ingest.imobiliare_loader import (
    load_imobiliare_directory,
    parse_imobiliare_record,
    split_street_address,
)

RECORD = {
    "id": "276113430",
    "url": "https://www.imobiliare.ro/oferta/apartament-de-inchiriat-sector-6-drumul-taberei",
    "operationType": "rent",
    "title": "Apartament cu 3 camere decomandat, mobilat în Drumul Taberei",
    "priceText": "630 EUR",
    "priceAmount": 630,
    "priceCurrency": "EUR",
    "city": "Sector 6",
    "neighborhood": "Drumul Taberei",
    "streetAddress": "2a valea argesului, Drumul Taberei, Sector 6",
    "floor": "4",
    "yearBuilt": 1975,
    "surfaceTotal": 75,
    "images": ["https://example.com/1.jpg"],
    "descriptionText": "Apartament luminos, aproape de metrou Romancierilor.",
    "latitude": None,
    "longitude": None,
}


def _record(**overrides) -> dict:
    return {**RECORD, **overrides}


def test_parses_a_rental_into_a_raw_listing():
    listing = parse_imobiliare_record(RECORD, source_file="rent.json")

    assert listing.external_id == "imobiliare-276113430"
    assert listing.listing_type == "rent"
    assert listing.title == RECORD["title"]
    assert listing.description == RECORD["descriptionText"]
    assert listing.price_raw == "630 EUR"
    assert listing.address_raw == "2a valea argesului, Drumul Taberei, Sector 6"
    assert listing.street == "valea argesului"
    assert listing.street_number == "2a"
    assert listing.neighbourhood == "Drumul Taberei"
    assert listing.specs == {"Suprafata utila": "75 m²", "Etaj": "4", "An constructie": "1975"}
    assert listing.image_urls == ["https://example.com/1.jpg"]
    assert listing.source_file == "rent.json"
    assert listing.map_lat is None


def test_sale_is_labelled_sale():
    assert parse_imobiliare_record(_record(operationType="sell"), source_file="f").listing_type == (
        "sale"
    )


def test_html_entities_are_decoded():
    listing = parse_imobiliare_record(
        _record(
            streetAddress="68 Bd. Bucureștii Noi, Bucure&#537;tii Noi, Sector 1",
            neighborhood="Bucure&#537;tii Noi",
            city="Sector 1",
        ),
        source_file="f",
    )

    assert listing.address_raw == "68 Bd. Bucureștii Noi, Bucureștii Noi, Sector 1"
    assert listing.neighbourhood == "Bucureștii Noi"


def test_sector_is_added_to_the_address_when_missing():
    listing = parse_imobiliare_record(
        _record(streetAddress="nr 36 Bulevardul Nicolae Bălcescu, Universitate, Ultracentral"),
        source_file="f",
    )

    assert listing.address_raw.endswith(", Sector 6")


# Runs started from ad URLs leave `neighborhood` and `city` empty.
@pytest.mark.parametrize(
    ("street_address", "neighbourhood", "street", "number"),
    [
        ("Floreasca, Sector 1", "Floreasca", None, None),
        (
            "30 Strada Doamna Ghica, Doamna Ghica, Sector 2",
            "Doamna Ghica",
            "Strada Doamna Ghica",
            "30",
        ),
        ("Avia&#539;iei, Sector 1", "Aviației", None, None),
        (
            "nr 36 Bulevardul Nicolae Bălcescu, Universitate, Ultracentral",
            None,
            "Bulevardul Nicolae Bălcescu",
            "36",
        ),
    ],
)
def test_neighbourhood_comes_from_the_address_when_the_field_is_empty(
    street_address, neighbourhood, street, number
):
    listing = parse_imobiliare_record(
        _record(streetAddress=street_address, neighborhood=None, city=None), source_file="f"
    )

    assert (listing.neighbourhood, listing.street, listing.street_number) == (
        neighbourhood,
        street,
        number,
    )


def test_ground_floor_uses_the_romanian_term_the_normalizer_knows():
    listing = parse_imobiliare_record(_record(floor="ground-floor"), source_file="f")

    assert listing.specs["Etaj"] == "parter"


def test_missing_optional_specs_are_left_out():
    listing = parse_imobiliare_record(
        _record(floor=None, yearBuilt=None, surfaceTotal=None), source_file="f"
    )

    assert listing.specs == {}


@pytest.mark.parametrize(
    "change",
    [{"priceCurrency": "RON"}, {"priceText": None}, {"operationType": "auction"}],
)
def test_rejects_records_it_cannot_price_or_classify(change):
    with pytest.raises(ValueError):
        parse_imobiliare_record(_record(**change), source_file="f")


@pytest.mark.parametrize(
    ("street_address", "neighbourhood", "expected"),
    [
        (
            "2a valea argesului, Drumul Taberei, Sector 6",
            "Drumul Taberei",
            ("valea argesului", "2a"),
        ),
        (
            "132 A Bulevardul Metalurgiei, Metalurgiei, Sector 4",
            "Metalurgiei",
            ("Bulevardul Metalurgiei", "132 A"),
        ),
        (
            "128 Calea 13 Septembrie, 13 Septembrie, Sector 5",
            "13 Septembrie",
            ("Calea 13 Septembrie", "128"),
        ),
        (
            "nr 36 Bulevardul Nicolae Bălcescu, Universitate",
            "Universitate",
            ("Bulevardul Nicolae Bălcescu", "36"),
        ),
        ("3-5 Strada Emil Racoviță, Berceni, Sector 4", "Berceni", ("Strada Emil Racoviță", "3-5")),
        ("Bd. Pipera nr.1D-11, Pipera, Sector 1", "Pipera", ("Bd. Pipera", "1D-11")),
        (
            "Strada Sergent Ilie Petre,nr 84, Militari, Sector 6",
            "Militari",
            ("Strada Sergent Ilie Petre", "84"),
        ),
        ("Bulevardul Pipera, Pipera, Sector 1", "Pipera", ("Bulevardul Pipera", None)),
        ("Strada Teleajen 40, Vatra Luminoasă", "Vatra Luminoasă", ("Strada Teleajen", "40")),
        ("Pipera, Sector 1", "Pipera", (None, None)),
        ("Sector 1", "Pipera", (None, None)),
    ],
)
def test_splits_street_and_number(street_address, neighbourhood, expected):
    assert split_street_address(street_address, neighbourhood) == expected


def test_loads_every_json_file_and_reports_bad_records(tmp_path: Path):
    (tmp_path / "rent_1.json").write_text(json.dumps([RECORD, _record(priceCurrency="RON")]))
    (tmp_path / "sell_1.json").write_text(json.dumps([_record(id="2", operationType="sell")]))
    (tmp_path / "broken.json").write_text("{not json")

    listings, failures = load_imobiliare_directory(tmp_path)

    assert sorted(listing.external_id for listing in listings) == [
        "imobiliare-2",
        "imobiliare-276113430",
    ]
    assert sorted(path.name for path, _ in failures) == ["broken.json", "rent_1.json"]

from realestate.enrich.normalizer import (
    normalize_price_eur,
    normalize_rooms,
    normalize_specs,
)


def test_normalize_price_eur_parses_dotted_thousands():
    assert normalize_price_eur("100.000 €") == 100000.0


def test_normalize_rooms_detects_studio():
    assert normalize_rooms("Garsoniera 40mp in Centrul Istoric") == 1


def test_normalize_rooms_detects_explicit_count():
    assert normalize_rooms("Apartament 3 camere de vanzare") == 3


def test_normalize_rooms_returns_none_when_unknown():
    assert normalize_rooms("Teren de vanzare") is None


def test_normalize_specs_maps_known_labels():
    raw_specs = {
        "Suprafata utila": "48 m²",
        "Etaj": "3",
        "An constructie": "1977 – 1990",
        "Compartimentare": "Decomandat",
    }
    normalized = normalize_specs(raw_specs)
    assert normalized["built_area_sqm"] == 48.0
    assert normalized["floor_number"] == 3
    assert normalized["construction_year_range"] == "1977 – 1990"
    assert normalized["layout_type"] == "Decomandat"
    assert len(normalized) == 4  # All 4 keys always present


def test_normalize_specs_ignores_unknown_labels():
    normalized = normalize_specs({"Un label necunoscut": "valoare"})
    assert normalized == {
        "built_area_sqm": None,
        "floor_number": None,
        "construction_year_range": None,
        "layout_type": None,
    }


def test_normalize_specs_handles_parter_floor():
    normalized = normalize_specs({"Etaj": "Parter"})
    assert normalized["floor_number"] == 0
    assert normalized["built_area_sqm"] is None
    assert normalized["construction_year_range"] is None
    assert normalized["layout_type"] is None


def test_normalize_specs_handles_demisol_floor():
    normalized = normalize_specs({"Etaj": "Demisol"})
    assert normalized["floor_number"] == -1
    assert normalized["built_area_sqm"] is None
    assert normalized["construction_year_range"] is None
    assert normalized["layout_type"] is None

from realestate.models import RawListing, EnrichedListing


def test_raw_listing_construction():
    listing = RawListing(
        external_id="304473136",
        title="Vand apartament 2 camere TITAN",
        description="Direct proprietar, vand apartament 2 camere...",
        price_raw="100.000 €",
        address_raw="Bucuresti - Ilfov, Bucuresti, Sectorul 3",
        specs={"Suprafata utila": "48 m²"},
        image_urls=["https://frankfurt.apollo.olxcdn.com:443/v1/files/example/image"],
        source_file="304473136.html",
    )
    assert listing.external_id == "304473136"
    assert listing.specs["Suprafata utila"] == "48 m²"


def test_enriched_listing_defaults_for_missing_location():
    listing = EnrichedListing(
        external_id="304473136",
        title="Vand apartament 2 camere TITAN",
        description="Direct proprietar, vand apartament 2 camere...",
        price_eur=100000.0,
        rooms=2,
        built_area_sqm=48.0,
        floor_number=3,
        construction_year_range="1977 – 1990",
        layout_type="Decomandat",
        address_text="Bucuresti - Ilfov, Bucuresti, Sectorul 3",
        latitude=None,
        longitude=None,
        location_confidence="low_confidence_location",
        nearest_subway_station=None,
        subway_walking_minutes=None,
        image_urls=[],
    )
    assert listing.location_confidence == "low_confidence_location"
    assert listing.latitude is None

from realestate.eval.generate_queries import generate_eval_query
from realestate.models import EnrichedListing

LISTING = EnrichedListing(
    external_id="304473136",
    title="Vand apartament 2 camere TITAN",
    description="Apartament luminos, decomandat, in Titan.",
    price_eur=100000.0,
    rooms=2,
    built_area_sqm=48.0,
    floor_number=3,
    construction_year_range="1977 – 1990",
    layout_type="Decomandat",
    address_text="Bucuresti - Ilfov, Bucuresti, Sectorul 3",
    latitude=44.43,
    longitude=26.10,
    location_confidence="ok",
    nearest_subway_station="Titan",
    subway_walking_minutes=6.5,
    image_urls=[],
)


def test_generate_eval_query_uses_listing_details_in_the_prompt():
    captured_prompts = []

    def fake_generate(prompt: str) -> str:
        captured_prompts.append(prompt)
        return "Bright 2-room apartment near Titan subway station"

    pair = generate_eval_query(LISTING, generate_fn=fake_generate)

    assert pair.relevant_listing_id == "304473136"
    assert pair.query == "Bright 2-room apartment near Titan subway station"
    assert "Sectorul 3" in captured_prompts[0]
    assert "Titan" in captured_prompts[0]

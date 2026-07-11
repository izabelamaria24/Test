from dataclasses import dataclass
from typing import Callable

from realestate.models import EnrichedListing

_GENERATE_PROMPT = """You are simulating a real estate search user. Given this listing, \
write ONE natural-language search query that a real user might type to find it. Do not \
mention the price or the exact address, but you may reference the neighborhood or nearby \
transit if relevant. Respond with only the query text.

Title: {title}
Description: {description}
Neighborhood: {address_text}
Nearest subway: {nearest_subway_station} ({subway_walking_minutes} min walk)
"""


@dataclass
class EvalPair:
    query: str
    relevant_listing_id: str


def generate_eval_query(
    listing: EnrichedListing, *, generate_fn: Callable[[str], str]
) -> EvalPair:
    prompt = _GENERATE_PROMPT.format(
        title=listing.title,
        description=listing.description,
        address_text=listing.address_text,
        nearest_subway_station=listing.nearest_subway_station,
        subway_walking_minutes=listing.subway_walking_minutes,
    )
    query = generate_fn(prompt)
    return EvalPair(query=query, relevant_listing_id=listing.external_id)

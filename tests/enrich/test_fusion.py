import math

import shapely
from shapely.geometry import LineString, Point, box

from realestate.enrich.fusion import Evidence, fuse

# A spot in Titan, roughly in the middle of the grid.
LON, LAT = 26.15, 44.42
M_PER_DEG_LAT = 111_320
M_PER_DEG_LON = 111_320 * math.cos(math.radians(LAT))


def _offset(east_m: float, north_m: float) -> Point:
    return Point(LON + east_m / M_PER_DEG_LON, LAT + north_m / M_PER_DEG_LAT)


def _distance_m(lon: float, lat: float, point: Point) -> float:
    return math.hypot((lon - point.x) * M_PER_DEG_LON, (lat - point.y) * M_PER_DEG_LAT)


def test_no_evidence_gives_no_location():
    assert fuse([]) is None


def test_evidence_outside_bucharest_is_ignored():
    assert fuse([Evidence("street", (Point(23.6, 46.77),), reach_m=75)]) is None


def test_single_precise_clue_is_located_precisely():
    target = _offset(0, 0)

    location = fuse([Evidence("building", (target,), reach_m=50)])

    assert location is not None
    assert _distance_m(location.longitude, location.latitude, target) < 100
    assert location.uncertainty_m < 300
    assert location.agreeing == ("building",)


def test_overlapping_clues_narrow_the_location():
    wide_a = Evidence("olx_point", (_offset(-1000, 0),), reach_m=1500)
    wide_b = Evidence("metro", (_offset(1000, 0),), reach_m=1500)

    alone = fuse([wide_a])
    combined = fuse([wide_a, wide_b])

    assert alone is not None and combined is not None
    assert _distance_m(combined.longitude, combined.latitude, _offset(0, 0)) < 200
    assert combined.uncertainty_m < alone.uncertainty_m
    assert set(combined.agreeing) == {"olx_point", "metro"}


def test_a_contradicting_clue_is_outvoted_and_reported():
    street = Evidence("street", (LineString([_offset(-500, 0), _offset(500, 0)]),), reach_m=75)
    metro = Evidence("metro", (_offset(0, 300),), reach_m=600)
    wrong = Evidence("landmark", (_offset(0, -8000),), reach_m=800, reliability=0.7)

    location = fuse([street, metro, wrong])

    assert location is not None
    assert _distance_m(location.longitude, location.latitude, _offset(0, 0)) < 400
    assert set(location.agreeing) == {"street", "metro"}
    assert location.conflicting == ("landmark",)


def test_ambiguous_place_name_resolves_to_the_candidate_other_clues_agree_with():
    near, far = _offset(0, 0), _offset(6000, 6000)
    street = Evidence("street", (far, near), reach_m=75)
    neighbourhood = Evidence(
        "neighbourhood",
        (box(*_offset(-800, -800).coords[0], *_offset(800, 800).coords[0]),),
        reach_m=0,
    )

    location = fuse([street, neighbourhood])

    assert location is not None
    assert _distance_m(location.longitude, location.latitude, near) < 150
    assert location.uncertainty_m < 300


def test_a_precise_clue_beats_several_loosely_used_nearby_names():
    # Ads call places 1-3 km away "nearby" and stretch neighbourhood names, so a building
    # address should not be outvoted by a couple of such clues agreeing with each other.
    building = Evidence("building", (_offset(0, 0),), reach_m=75, reliability=0.9)
    neighbourhood = Evidence("neighbourhood", (_offset(2300, 0),), reach_m=1000, reliability=0.7)
    landmark = Evidence("landmark", (_offset(2500, 300),), reach_m=1500, reliability=0.6)

    location = fuse([building, neighbourhood, landmark])

    assert location is not None
    assert _distance_m(location.longitude, location.latitude, _offset(0, 0)) < 150
    assert "building" in location.agreeing


def test_equally_likely_places_give_one_of_them_not_the_midpoint():
    west, east = _offset(-2000, 0), _offset(2000, 0)

    location = fuse([Evidence("landmark", (west, east), reach_m=100)])

    assert location is not None
    nearest_m = min(
        _distance_m(location.longitude, location.latitude, west),
        _distance_m(location.longitude, location.latitude, east),
    )
    assert nearest_m < 200
    assert location.uncertainty_m > 2000


def test_polygon_clue_places_the_estimate_inside_it():
    area = shapely.box(*_offset(-1500, -1000).coords[0], *_offset(1500, 1000).coords[0])

    location = fuse([Evidence("sector", (area,), reach_m=0)])

    assert location is not None
    assert area.contains(Point(location.longitude, location.latitude))
    assert 1000 < location.uncertainty_m < 2500

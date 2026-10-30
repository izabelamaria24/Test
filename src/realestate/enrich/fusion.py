"""Combine uncertain location clues into one position by probabilistic grid fusion.

Each clue gives every ~100 m grid cell over Bucharest a likelihood: 1 within `reach_m` of
any of its candidate geometries, fading outside over a distance that grows with the reach. Clues are combined as a robust mixture
(a clue is right with probability `reliability`, otherwise uninformative), so a single
wrong clue is outvoted instead of vetoing the rest, and an ambiguous name with several
candidate places resolves to whichever candidate the other clues agree with.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import shapely
from scipy import ndimage
from shapely.geometry.base import BaseGeometry

# lon_min, lat_min, lon_max, lat_max: Bucharest plus its immediate Ilfov edge.
GRID_BOUNDS = (25.96, 44.33, 26.23, 44.54)
_CELL_M = 100
_MIN_SOFT_EDGE_M = 50
_SOFT_EDGE_FRACTION_OF_REACH = 0.5
_MASS_FOR_UNCERTAINTY = 0.9
_PLAUSIBLE_FRACTION_OF_BEST = 1e-3
_REGION_FRACTION_OF_BEST = 0.1
_LAT0 = (GRID_BOUNDS[1] + GRID_BOUNDS[3]) / 2
_M_PER_DEG_LAT = 111_320.0
_M_PER_DEG_LON = 111_320.0 * math.cos(math.radians(_LAT0))


@dataclass(frozen=True)
class Evidence:
    kind: str
    # Possible places the clue refers to, in lon/lat; several when a name is ambiguous.
    candidates: tuple[BaseGeometry, ...]
    reach_m: float
    reliability: float = 0.9


@dataclass(frozen=True)
class FusedLocation:
    latitude: float
    longitude: float
    # Radius around the estimate holding 90% of the probability.
    uncertainty_m: float
    agreeing: tuple[str, ...]
    conflicting: tuple[str, ...]


def _to_metres(coords: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (
            (coords[:, 0] - GRID_BOUNDS[0]) * _M_PER_DEG_LON,
            (coords[:, 1] - GRID_BOUNDS[1]) * _M_PER_DEG_LAT,
        )
    )


def _grid() -> tuple[np.ndarray, np.ndarray]:
    width = (GRID_BOUNDS[2] - GRID_BOUNDS[0]) * _M_PER_DEG_LON
    height = (GRID_BOUNDS[3] - GRID_BOUNDS[1]) * _M_PER_DEG_LAT
    xs, ys = np.meshgrid(
        np.arange(_CELL_M / 2, width, _CELL_M), np.arange(_CELL_M / 2, height, _CELL_M)
    )
    return xs, ys


_GRID_X_2D, _GRID_Y_2D = _grid()
_GRID_SHAPE = _GRID_X_2D.shape
_GRID_X, _GRID_Y = _GRID_X_2D.ravel(), _GRID_Y_2D.ravel()
_GRID_POINTS = shapely.points(_GRID_X, _GRID_Y)


def _distance_m(evidence: Evidence, points) -> np.ndarray:
    projected = [shapely.transform(geometry, _to_metres) for geometry in evidence.candidates]
    return np.min([shapely.distance(points, geometry) for geometry in projected], axis=0)


def _soft_edge_m(evidence: Evidence) -> float:
    # Vague clues have vague edges: a neighbourhood name stretches further than a house number.
    return max(_MIN_SOFT_EDGE_M, _SOFT_EDGE_FRACTION_OF_REACH * evidence.reach_m)


def _likelihood(evidence: Evidence) -> np.ndarray:
    excess = np.maximum(_distance_m(evidence, _GRID_POINTS) - evidence.reach_m, 0.0)
    return np.exp(-0.5 * (excess / _soft_edge_m(evidence)) ** 2)


def _centre_of_most_probable_region(posterior: np.ndarray) -> tuple[float, float]:
    # The mean of a map with several separate peaks can fall between them, in a place no clue
    # supports; take the centre of the connected high-probability region with the most mass.
    high = (posterior >= _REGION_FRACTION_OF_BEST * posterior.max()).reshape(_GRID_SHAPE)
    labels, region_count = cast("tuple[np.ndarray, int]", ndimage.label(high))
    labels = labels.ravel()
    masses = ndimage.sum(posterior, labels, index=np.arange(1, region_count + 1))
    in_region = labels == int(np.argmax(masses)) + 1
    weights = posterior * in_region
    weights /= weights.sum()
    return float(weights @ _GRID_X), float(weights @ _GRID_Y)


def fuse(evidence: Sequence[Evidence]) -> FusedLocation | None:
    cell_count = _GRID_X.size
    log_posterior = np.zeros(cell_count)
    used: list[Evidence] = []
    for item in evidence:
        likelihood = _likelihood(item)
        total = likelihood.sum()
        if total == 0:
            continue
        log_posterior += np.log(
            item.reliability * likelihood / total + (1 - item.reliability) / cell_count
        )
        used.append(item)
    if not used:
        return None

    posterior = np.exp(log_posterior - log_posterior.max())
    # Drop the city-wide floor left by "every clue was wrong"; keep any real competing mode.
    posterior[posterior < _PLAUSIBLE_FRACTION_OF_BEST] = 0.0
    posterior /= posterior.sum()
    x, y = _centre_of_most_probable_region(posterior)

    radii = np.hypot(_GRID_X - x, _GRID_Y - y)
    order = np.argsort(radii)
    within = np.searchsorted(np.cumsum(posterior[order]), _MASS_FOR_UNCERTAINTY)
    uncertainty_m = max(float(radii[order][min(within, cell_count - 1)]), _CELL_M / 2)

    estimate = shapely.points([x], [y])
    agrees = [
        (item.kind, bool(_distance_m(item, estimate)[0] <= item.reach_m + 2 * _soft_edge_m(item)))
        for item in used
    ]
    return FusedLocation(
        latitude=GRID_BOUNDS[1] + y / _M_PER_DEG_LAT,
        longitude=GRID_BOUNDS[0] + x / _M_PER_DEG_LON,
        uncertainty_m=uncertainty_m,
        agreeing=tuple(dict.fromkeys(kind for kind, ok in agrees if ok)),
        conflicting=tuple(dict.fromkeys(kind for kind, ok in agrees if not ok)),
    )

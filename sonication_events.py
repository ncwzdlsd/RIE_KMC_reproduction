from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from constants import Species
from lattice_build import Lattice
from paper_parameters import (
    PAPER_DISSOLUTION_PROBABILITY,
    PAPER_SONICATION_RADIUS_NM,
)


class SonicationEventType(str, Enum):
    CORROSION = "sonication_corrosion"


@dataclass(frozen=True)
class SonicationParameters:
    """Paper-inspired stochastic corrosion at the particle-solution interface.

    The paper specifies a 1 nm sphere and a 10% dissolution probability, but
    does not report the event frequency.  ``event_rate`` is the KMC propensity
    per eligible nanoparticle-solution interface center (s^-1 per site), and
    must be fitted before interpreting KMC time quantitatively.
    """

    event_rate: float
    radius_nm: float = PAPER_SONICATION_RADIUS_NM
    dissolution_probability: float = PAPER_DISSOLUTION_PROBABILITY

    def __post_init__(self):
        if self.event_rate < 0.0 or not math.isfinite(self.event_rate):
            raise ValueError("event_rate must be finite and non-negative")
        if self.radius_nm <= 0.0 or not math.isfinite(self.radius_nm):
            raise ValueError("radius_nm must be finite and positive")
        if not 0.0 <= self.dissolution_probability <= 1.0:
            raise ValueError("dissolution_probability must be between 0 and 1")


@dataclass(frozen=True)
class SonicationEvent:
    event_type: SonicationEventType
    rate: float
    surface_support_site_ids: np.ndarray


def build_sonication_event(
    lattice: Lattice,
    parameters: SonicationParameters,
    external_surface: np.ndarray,
) -> SonicationEvent | None:
    if parameters.event_rate == 0.0:
        return None

    support_mask = (
        external_surface
        & (
            (lattice.occupation == Species.CE)
            | (lattice.occupation == Species.O)
        )
    )
    surface_support_site_ids = np.flatnonzero(support_mask).astype(
        np.int32, copy=False
    )
    if len(surface_support_site_ids) == 0:
        return None

    return SonicationEvent(
        event_type=SonicationEventType.CORROSION,
        # This is an n-fold-way aggregation of one identical KMC event for
        # every eligible interface center.  Selecting the aggregate event and
        # then choosing a center uniformly is exactly equivalent to storing
        # every center separately in the event catalog.
        rate=parameters.event_rate * len(surface_support_site_ids),
        surface_support_site_ids=surface_support_site_ids,
    )


def apply_sonication_event(
    lattice: Lattice,
    event: SonicationEvent,
    parameters: SonicationParameters,
    rng: np.random.Generator,
) -> np.ndarray:
    candidate_ids = event.surface_support_site_ids
    center_id = int(candidate_ids[rng.integers(len(candidate_ids))])
    center = lattice.positions_nm[center_id]
    offsets = lattice.positions_nm[candidate_ids] - center
    within_sphere = np.einsum("ij,ij->i", offsets, offsets) <= (
        parameters.radius_nm**2
    )
    local_surface_ids = candidate_ids[within_sphere]
    dissolve = rng.random(len(local_surface_ids)) < (
        parameters.dissolution_probability
    )
    removed_site_ids = local_surface_ids[dissolve]
    lattice.occupation[removed_site_ids] = Species.EMPTY
    return removed_site_ids

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from constants import SiteType, Species
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
    does not report the event frequency.  ``event_rate`` is therefore an input
    that must be fitted before interpreting KMC time quantitatively.
    """

    event_rate: float
    radius_nm: float = PAPER_SONICATION_RADIUS_NM
    dissolution_probability: float = PAPER_DISSOLUTION_PROBABILITY
    maximum_chemical_potential_boost_ev: float = 0.0
    events_for_maximum_boost: int = 1
    mean_field_growth_atoms_per_event: int = 0
    growth_capture_radius_nm: float = 1.0

    def __post_init__(self):
        if self.event_rate < 0.0 or not math.isfinite(self.event_rate):
            raise ValueError("event_rate must be finite and non-negative")
        if self.radius_nm <= 0.0 or not math.isfinite(self.radius_nm):
            raise ValueError("radius_nm must be finite and positive")
        if not 0.0 <= self.dissolution_probability <= 1.0:
            raise ValueError("dissolution_probability must be between 0 and 1")
        if (
            self.maximum_chemical_potential_boost_ev < 0.0
            or not math.isfinite(self.maximum_chemical_potential_boost_ev)
        ):
            raise ValueError(
                "maximum_chemical_potential_boost_ev must be finite and non-negative"
            )
        if self.events_for_maximum_boost <= 0:
            raise ValueError("events_for_maximum_boost must be positive")
        if self.mean_field_growth_atoms_per_event < 0:
            raise ValueError("mean_field_growth_atoms_per_event must be non-negative")
        if self.growth_capture_radius_nm <= 0.0:
            raise ValueError("growth_capture_radius_nm must be positive")


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
        rate=parameters.event_rate,
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


def apply_mean_field_ripening_growth(
    lattice: Lattice,
    parameters: SonicationParameters,
    rng: np.random.Generator,
) -> np.ndarray:
    """Coarse-grain unresolved donor particles into a local CeOx growth burst.

    The atomistic paper model represents a much larger solution population than
    the compact OVITO demonstration.  This optional visualization accelerator
    deposits Ce/O at support growth-front sites close to pre-nucleated Ir.
    """
    target_count = parameters.mean_field_growth_atoms_per_event
    ir_ids = np.flatnonzero(lattice.occupation == Species.IR)
    if target_count == 0 or len(ir_ids) == 0:
        return np.empty(0, dtype=np.int32)

    empty_ids = np.flatnonzero(lattice.occupation == Species.EMPTY)
    empty_positions = lattice.positions_nm[empty_ids]
    ir_positions = lattice.positions_nm[ir_ids]
    minimum_distance_squared = np.full(len(empty_ids), np.inf)
    for ir_position in ir_positions:
        offsets = empty_positions - ir_position
        minimum_distance_squared = np.minimum(
            minimum_distance_squared,
            np.einsum("ij,ij->i", offsets, offsets),
        )
    near_ir_ids = empty_ids[
        minimum_distance_squared <= parameters.growth_capture_radius_nm**2
    ]

    ce_candidates = []
    o_candidates = []
    for site_id in near_ir_ids:
        site_id = int(site_id)
        neighbor_ids = lattice.get_ce_o_neighbors(site_id)
        neighbor_occupations = lattice.occupation[neighbor_ids]
        if lattice.site_type[site_id] == SiteType.M:
            if np.any(neighbor_occupations == Species.O):
                ce_candidates.append(site_id)
        elif np.any(
            (neighbor_occupations == Species.CE)
            | (neighbor_occupations == Species.IR)
        ):
            o_candidates.append(site_id)

    rng.shuffle(ce_candidates)
    rng.shuffle(o_candidates)
    number_ce = min(len(ce_candidates), target_count // 3)
    number_o = min(len(o_candidates), target_count - number_ce)
    selected_ce = np.asarray(ce_candidates[:number_ce], dtype=np.int32)
    selected_o = np.asarray(o_candidates[:number_o], dtype=np.int32)
    lattice.occupation[selected_ce] = Species.CE
    lattice.occupation[selected_o] = Species.O
    return np.concatenate((selected_ce, selected_o))

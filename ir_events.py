from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from constants import KB_EV_PER_K, SiteType, Species
from lattice_build import Lattice


class IrEventType(str, Enum):
    IR_ION_ADSORPTION = "Ir_ion_adsorption"
    IR_ION_DESORPTION = "Ir_ion_desorption"
    IR_ION_DIFFUSION = "Ir_ion_diffusion"
    IR_REDUCTION = "Ir_reduction"
    IR_OXIDATION = "Ir_oxidation"


@dataclass(frozen=True)
class IrParameters:
    """Parameters for the Ir part of the lattice-gas model.

    Binding energies are positive stabilization-energy magnitudes.  The
    reduction free energy is G(IR) - G(IR_ION), so a negative value favors
    reduction.  The numerical values must be calibrated from the paper/DFT or
    from experiment before a production simulation is interpreted.
    """

    ir_ir_binding_energy_ev: float
    ir_o_binding_energy_ev: float
    chemical_potential_ir_ion_ev: float
    reduction_free_energy_ev: float
    temperature_k: float = 453.0
    adsorption_prefactor: float = 1.0
    desorption_prefactor: float = 1.0
    diffusion_prefactor: float = 1.0
    reduction_prefactor: float = 1.0
    oxidation_prefactor: float = 1.0
    adsorption_barrier_ev: float = 0.0
    desorption_barrier_ev: float = 0.0
    diffusion_barrier_ev: float = 0.0
    reduction_barrier_ev: float = 0.0
    oxidation_barrier_ev: float = 0.0

    def __post_init__(self):
        if self.temperature_k <= 0.0:
            raise ValueError("temperature_k must be positive")
        if self.ir_ir_binding_energy_ev < 0.0 or self.ir_o_binding_energy_ev < 0.0:
            raise ValueError("Ir binding-energy magnitudes must be non-negative")

        prefactors = (
            self.adsorption_prefactor,
            self.desorption_prefactor,
            self.diffusion_prefactor,
            self.reduction_prefactor,
            self.oxidation_prefactor,
        )
        if any(value < 0.0 or not math.isfinite(value) for value in prefactors):
            raise ValueError("Ir event prefactors must be finite and non-negative")

        barriers = (
            self.adsorption_barrier_ev,
            self.desorption_barrier_ev,
            self.diffusion_barrier_ev,
            self.reduction_barrier_ev,
            self.oxidation_barrier_ev,
        )
        if any(value < 0.0 or not math.isfinite(value) for value in barriers):
            raise ValueError("Ir event barriers must be finite and non-negative")


@dataclass(frozen=True)
class IrEvent:
    event_type: IrEventType
    site_id: int
    target_site_id: int | None
    rate: float
    delta_free_energy_ev: float
    ir_neighbor_count: int
    o_neighbor_count: int


def count_ir_neighbors(
    lattice: Lattice,
    m_site_id: int,
    ignored_site_id: int | None = None,
) -> int:
    neighbor_ids = lattice.get_m_m_neighbors(m_site_id)
    if ignored_site_id is not None:
        neighbor_ids = neighbor_ids[neighbor_ids != ignored_site_id]
    occupations = lattice.occupation[neighbor_ids]
    return int(
        np.count_nonzero(
            (occupations == Species.IR_ION) | (occupations == Species.IR)
        )
    )


def count_o_neighbors(lattice: Lattice, m_site_id: int) -> int:
    neighbor_ids = lattice.get_ce_o_neighbors(m_site_id)
    return int(np.count_nonzero(lattice.occupation[neighbor_ids] == Species.O))


def local_ir_binding_energy(
    lattice: Lattice,
    site_id: int,
    parameters: IrParameters,
    ignored_ir_site_id: int | None = None,
) -> tuple[float, int, int]:
    ir_neighbors = count_ir_neighbors(lattice, site_id, ignored_ir_site_id)
    o_neighbors = count_o_neighbors(lattice, site_id)
    binding_energy_ev = (
        ir_neighbors * parameters.ir_ir_binding_energy_ev
        + o_neighbors * parameters.ir_o_binding_energy_ev
    )
    return binding_energy_ev, ir_neighbors, o_neighbors


def activated_rate(
    delta_free_energy_ev: float,
    prefactor: float,
    base_barrier_ev: float,
    parameters: IrParameters,
) -> float:
    thermal_energy_ev = KB_EV_PER_K * parameters.temperature_k
    activation_energy_ev = base_barrier_ev + max(0.0, delta_free_energy_ev)
    return prefactor * math.exp(-activation_energy_ev / thermal_energy_ev)


def build_ir_ion_adsorption_event(
    lattice: Lattice,
    site_id: int,
    parameters: IrParameters,
) -> IrEvent | None:
    if lattice.site_type[site_id] != SiteType.M:
        return None
    if lattice.occupation[site_id] != Species.EMPTY:
        return None
    if not lattice.reservoir_boundary[site_id]:
        return None

    binding_energy_ev, ir_neighbors, o_neighbors = local_ir_binding_energy(
        lattice, site_id, parameters
    )
    delta_free_energy_ev = (
        -parameters.chemical_potential_ir_ion_ev - binding_energy_ev
    )
    rate = activated_rate(
        delta_free_energy_ev,
        parameters.adsorption_prefactor,
        parameters.adsorption_barrier_ev,
        parameters,
    )
    return IrEvent(
        event_type=IrEventType.IR_ION_ADSORPTION,
        site_id=site_id,
        target_site_id=None,
        rate=rate,
        delta_free_energy_ev=delta_free_energy_ev,
        ir_neighbor_count=ir_neighbors,
        o_neighbor_count=o_neighbors,
    )


def build_ir_ion_desorption_event(
    lattice: Lattice,
    site_id: int,
    parameters: IrParameters,
) -> IrEvent | None:
    if lattice.site_type[site_id] != SiteType.M:
        return None
    if lattice.occupation[site_id] != Species.IR_ION:
        return None
    if not lattice.reservoir_boundary[site_id]:
        return None

    binding_energy_ev, ir_neighbors, o_neighbors = local_ir_binding_energy(
        lattice, site_id, parameters
    )
    delta_free_energy_ev = (
        parameters.chemical_potential_ir_ion_ev + binding_energy_ev
    )
    rate = activated_rate(
        delta_free_energy_ev,
        parameters.desorption_prefactor,
        parameters.desorption_barrier_ev,
        parameters,
    )
    return IrEvent(
        event_type=IrEventType.IR_ION_DESORPTION,
        site_id=site_id,
        target_site_id=None,
        rate=rate,
        delta_free_energy_ev=delta_free_energy_ev,
        ir_neighbor_count=ir_neighbors,
        o_neighbor_count=o_neighbors,
    )


def build_ir_ion_diffusion_event(
    lattice: Lattice,
    source_site_id: int,
    target_site_id: int,
    parameters: IrParameters,
) -> IrEvent | None:
    if lattice.site_type[source_site_id] != SiteType.M:
        return None
    if lattice.occupation[source_site_id] != Species.IR_ION:
        return None
    if lattice.site_type[target_site_id] != SiteType.M:
        return None
    if lattice.occupation[target_site_id] != Species.EMPTY:
        return None

    initial_binding_ev, _, _ = local_ir_binding_energy(
        lattice, source_site_id, parameters
    )
    final_binding_ev, ir_neighbors, o_neighbors = local_ir_binding_energy(
        lattice,
        target_site_id,
        parameters,
        ignored_ir_site_id=source_site_id,
    )
    delta_free_energy_ev = initial_binding_ev - final_binding_ev
    rate = activated_rate(
        delta_free_energy_ev,
        parameters.diffusion_prefactor,
        parameters.diffusion_barrier_ev,
        parameters,
    )
    return IrEvent(
        event_type=IrEventType.IR_ION_DIFFUSION,
        site_id=source_site_id,
        target_site_id=target_site_id,
        rate=rate,
        delta_free_energy_ev=delta_free_energy_ev,
        ir_neighbor_count=ir_neighbors,
        o_neighbor_count=o_neighbors,
    )


def build_ir_reduction_event(
    lattice: Lattice,
    site_id: int,
    parameters: IrParameters,
) -> IrEvent | None:
    if lattice.occupation[site_id] != Species.IR_ION:
        return None

    _, ir_neighbors, o_neighbors = local_ir_binding_energy(
        lattice, site_id, parameters
    )
    delta_free_energy_ev = parameters.reduction_free_energy_ev
    rate = activated_rate(
        delta_free_energy_ev,
        parameters.reduction_prefactor,
        parameters.reduction_barrier_ev,
        parameters,
    )
    return IrEvent(
        event_type=IrEventType.IR_REDUCTION,
        site_id=site_id,
        target_site_id=None,
        rate=rate,
        delta_free_energy_ev=delta_free_energy_ev,
        ir_neighbor_count=ir_neighbors,
        o_neighbor_count=o_neighbors,
    )


def build_ir_oxidation_event(
    lattice: Lattice,
    site_id: int,
    parameters: IrParameters,
) -> IrEvent | None:
    if lattice.occupation[site_id] != Species.IR:
        return None

    _, ir_neighbors, o_neighbors = local_ir_binding_energy(
        lattice, site_id, parameters
    )
    delta_free_energy_ev = -parameters.reduction_free_energy_ev
    rate = activated_rate(
        delta_free_energy_ev,
        parameters.oxidation_prefactor,
        parameters.oxidation_barrier_ev,
        parameters,
    )
    return IrEvent(
        event_type=IrEventType.IR_OXIDATION,
        site_id=site_id,
        target_site_id=None,
        rate=rate,
        delta_free_energy_ev=delta_free_energy_ev,
        ir_neighbor_count=ir_neighbors,
        o_neighbor_count=o_neighbors,
    )


def build_ir_event_catalog(
    lattice: Lattice,
    parameters: IrParameters,
) -> list[IrEvent]:
    events: list[IrEvent] = []

    for site_id in np.flatnonzero(lattice.site_type == SiteType.M):
        site_id = int(site_id)
        species = Species(lattice.occupation[site_id])

        if species == Species.EMPTY:
            event = build_ir_ion_adsorption_event(lattice, site_id, parameters)
            if event is not None:
                events.append(event)
        elif species == Species.IR_ION:
            event = build_ir_ion_desorption_event(lattice, site_id, parameters)
            if event is not None:
                events.append(event)

            for target_site_id in lattice.get_m_m_neighbors(site_id):
                event = build_ir_ion_diffusion_event(
                    lattice, site_id, int(target_site_id), parameters
                )
                if event is not None:
                    events.append(event)

            reduction_event = build_ir_reduction_event(lattice, site_id, parameters)
            if reduction_event is not None:
                events.append(reduction_event)
        elif species == Species.IR:
            oxidation_event = build_ir_oxidation_event(lattice, site_id, parameters)
            if oxidation_event is not None:
                events.append(oxidation_event)

    return events


def apply_ir_event(lattice: Lattice, event: IrEvent):
    if event.event_type == IrEventType.IR_ION_ADSORPTION:
        lattice.occupation[event.site_id] = Species.IR_ION
    elif event.event_type == IrEventType.IR_ION_DESORPTION:
        lattice.occupation[event.site_id] = Species.EMPTY
    elif event.event_type == IrEventType.IR_ION_DIFFUSION:
        if event.target_site_id is None:
            raise ValueError("Ir diffusion event has no target site")
        lattice.occupation[event.site_id] = Species.EMPTY
        lattice.occupation[event.target_site_id] = Species.IR_ION
    elif event.event_type == IrEventType.IR_REDUCTION:
        lattice.occupation[event.site_id] = Species.IR
    elif event.event_type == IrEventType.IR_OXIDATION:
        lattice.occupation[event.site_id] = Species.IR_ION
    else:
        raise ValueError(f"unsupported Ir event type: {event.event_type}")

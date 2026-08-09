from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from constants import KB_EV_PER_K, Species
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

    Positive pair energies stabilize a local configuration and negative pair
    energies destabilize it.  The reduction free energy is
    G(IR) - G(IR_ION), so a negative value favors reduction.  Kinetic values
    not published in the supplement must be calibrated before KMC time is
    interpreted physically.
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
    precursor_ir_to_ce_atom_ratio: float = 0.0

    def __post_init__(self):
        if self.temperature_k <= 0.0:
            raise ValueError("temperature_k must be positive")
        if not math.isfinite(self.ir_ir_binding_energy_ev) or not math.isfinite(
            self.ir_o_binding_energy_ev
        ):
            raise ValueError("Ir pair energies must be finite")

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
        if (
            self.precursor_ir_to_ce_atom_ratio < 0.0
            or not math.isfinite(self.precursor_ir_to_ce_atom_ratio)
        ):
            raise ValueError(
                "precursor_ir_to_ce_atom_ratio must be finite and non-negative"
            )


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


def has_direct_support_contact(lattice: Lattice, site_id: int) -> bool:
    """Whether an M site touches the solid Ce/O support."""
    o_neighbors = lattice.get_ce_o_neighbors(site_id)
    if np.any(lattice.occupation[o_neighbors] == Species.O):
        return True
    m_neighbors = lattice.get_m_m_neighbors(site_id)
    return bool(np.any(lattice.occupation[m_neighbors] == Species.CE))


def has_metallic_ir_contact(lattice: Lattice, site_id: int) -> bool:
    """Whether a site can grow an already nucleated metallic Ir cluster."""
    neighbors = lattice.get_m_m_neighbors(site_id)
    return bool(np.any(lattice.occupation[neighbors] == Species.IR))


def is_ir_attachment_site(lattice: Lattice, site_id: int) -> bool:
    """Sites exposed to solution at the support/anchored-Ir interface."""
    return has_direct_support_contact(lattice, site_id) or has_metallic_ir_contact(
        lattice, site_id
    )


def is_supported_reduction_site(lattice: Lattice, site_id: int) -> bool:
    """Allow heterogeneous Ir nucleation/growth, but not bulk reduction."""
    return is_ir_attachment_site(lattice, site_id)


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

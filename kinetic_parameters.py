"""Serializable kinetic parameters with explicit physical rate units."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

from ceox_events import CeOxParameters
from ir_events import IrParameters
from paper_parameters import (
    DFT_CE_O_BINDING_ENERGY_EV,
    DFT_IR_IR_BINDING_ENERGY_EV,
    DFT_IR_O_BINDING_ENERGY_EV,
    PAPER_DISSOLUTION_PROBABILITY,
    PAPER_CHEMICAL_POTENTIAL_CE_EV,
    PAPER_CHEMICAL_POTENTIAL_O_EV,
    PAPER_RIE_IR_TO_CE_ATOM_RATIO,
    PAPER_SONICATION_RADIUS_NM,
    PAPER_TEMPERATURE_K,
)
from sonication_events import SonicationParameters


@dataclass(frozen=True)
class KineticParameterSet:
    """All fitted quantities needed to interpret KMC time in seconds."""

    # Ce/O exchange must relax the interface between successive acoustic
    # corrosion events; these remain initial estimates pending calibration.
    ce_adsorption_prefactor_s: float = 1.0e-3
    ce_desorption_prefactor_s: float = 1.0e-3
    ce_exchange_barrier_ev: float = 0.0
    # The paper-style comparison uses the fixed high-concentration bath from
    # Figs. S31/S34.  Dissolved Ce/O is diagnostic and does not feed back into
    # either chemical potential during a run.
    chemical_potential_ce_ev: float = PAPER_CHEMICAL_POTENTIAL_CE_EV
    chemical_potential_o_ev: float = PAPER_CHEMICAL_POTENTIAL_O_EV

    # The supplement reports that Ir nanoparticles are already visible after
    # 5 min and that nucleation is essentially complete within 60 min.  The
    # numerical rates are unpublished, so these defaults are constrained by
    # those two observations rather than by the earlier transport-only pilot.
    ir_adsorption_prefactor_s: float = 3.0e-4
    ir_desorption_prefactor_s: float = 1.0e-5
    # Morphology-informed initial estimates: ions must cross the empty M-site
    # network and explore the supported interface before reduction freezes
    # them into a metallic cluster.
    # 2.0 s^-1 per available hop yields a few support arrivals by 5 min in the
    # 20 nm / 5 nm paper geometry without replacing explicit nearest-neighbor
    # transport by direct deposition.  Reduction is fast enough to compete
    # with escape once an ion reaches the support.  Both unpublished values
    # still need a quantitative fit to the time-resolved Ir size data.
    ir_diffusion_prefactor_s: float = 2.0
    ir_reduction_prefactor_s: float = 5.0e-1
    ir_oxidation_prefactor_s: float = 1.0e-6
    ir_adsorption_barrier_ev: float = 0.0
    ir_desorption_barrier_ev: float = 0.0
    ir_diffusion_barrier_ev: float = 0.0
    ir_reduction_barrier_ev: float = 0.0
    ir_oxidation_barrier_ev: float = 0.0
    chemical_potential_ir_ion_ev: float = -0.15
    reduction_free_energy_ev: float = -0.10
    # Table S9 reports Ir retained in the washed final catalyst.  The
    # supplement does not publish a precursor-retention efficiency, so do not
    # inflate the atom inventory with an arbitrary inverse-retention factor.
    # Faster heterogeneous reduction makes the explicit target-sized dose
    # visible at the support instead of leaving most of it in transport.
    precursor_ir_to_ce_atom_ratio: float = PAPER_RIE_IR_TO_CE_ATOM_RATIO
    precursor_retention_fraction: float = 1.0

    # Propensity of one eligible nanoparticle-solution interface center.  The
    # total sonication propensity is this value times the current number of
    # surface Ce/O centers.  Roughly 1,000 centers are present initially in
    # the standard 5 nm particle, preserving an initial total rate near the
    # previous 1e-3 s^-1 estimate without treating the particle as one event.
    sonication_event_rate_s: float = 2.0e-6
    # Sonication releases Ce/O from the population of small, rough particles
    # into the well-mixed bath.  The single explicitly represented particle
    # is the growing member of that population, so this fitted chemical-
    # potential increment represents the experimentally observed enrichment
    # of its shared bath (Tables S3/S5).  This remains an initial estimate for
    # the calibration routine; the published paper does not provide the
    # concentration-to-chemical-potential conversion.
    sonication_chemical_potential_shift_ev: float = 0.10
    calibrated: bool = False
    calibration_objective: float | None = None
    calibration_scope: str = "initial guesses only"

    def ceox_parameters(self, sonication: bool = False) -> CeOxParameters:
        chemical_potential_shift = (
            self.sonication_chemical_potential_shift_ev if sonication else 0.0
        )
        return CeOxParameters(
            temperature_k=PAPER_TEMPERATURE_K,
            ce_o_binding_energy_ev=DFT_CE_O_BINDING_ENERGY_EV,
            ir_o_binding_energy_ev=DFT_IR_O_BINDING_ENERGY_EV,
            chemical_potential_ce_ev=(
                self.chemical_potential_ce_ev + chemical_potential_shift
            ),
            chemical_potential_o_ev=(
                self.chemical_potential_o_ev + chemical_potential_shift
            ),
            adsorption_prefactor=self.ce_adsorption_prefactor_s,
            desorption_prefactor=self.ce_desorption_prefactor_s,
            exchange_barrier_ev=self.ce_exchange_barrier_ev,
        )

    def ir_parameters(self) -> IrParameters:
        return IrParameters(
            ir_ir_binding_energy_ev=DFT_IR_IR_BINDING_ENERGY_EV,
            ir_o_binding_energy_ev=DFT_IR_O_BINDING_ENERGY_EV,
            chemical_potential_ir_ion_ev=self.chemical_potential_ir_ion_ev,
            reduction_free_energy_ev=self.reduction_free_energy_ev,
            temperature_k=PAPER_TEMPERATURE_K,
            adsorption_prefactor=self.ir_adsorption_prefactor_s,
            desorption_prefactor=self.ir_desorption_prefactor_s,
            diffusion_prefactor=self.ir_diffusion_prefactor_s,
            reduction_prefactor=self.ir_reduction_prefactor_s,
            oxidation_prefactor=self.ir_oxidation_prefactor_s,
            adsorption_barrier_ev=self.ir_adsorption_barrier_ev,
            desorption_barrier_ev=self.ir_desorption_barrier_ev,
            diffusion_barrier_ev=self.ir_diffusion_barrier_ev,
            reduction_barrier_ev=self.ir_reduction_barrier_ev,
            oxidation_barrier_ev=self.ir_oxidation_barrier_ev,
            precursor_ir_to_ce_atom_ratio=self.precursor_ir_to_ce_atom_ratio,
            precursor_retention_fraction=self.precursor_retention_fraction,
        )

    def sonication_parameters(self) -> SonicationParameters:
        return SonicationParameters(
            event_rate=self.sonication_event_rate_s,
            radius_nm=PAPER_SONICATION_RADIUS_NM,
            dissolution_probability=PAPER_DISSOLUTION_PROBABILITY,
        )

    def scaled(
        self,
        ce_scale: float = 1.0,
        sonication_scale: float = 1.0,
        sonication_chemical_potential_shift_ev: float | None = None,
    ) -> "KineticParameterSet":
        return replace(
            self,
            ce_adsorption_prefactor_s=self.ce_adsorption_prefactor_s * ce_scale,
            ce_desorption_prefactor_s=self.ce_desorption_prefactor_s * ce_scale,
            sonication_event_rate_s=(
                self.sonication_event_rate_s * sonication_scale
            ),
            sonication_chemical_potential_shift_ev=(
                self.sonication_chemical_potential_shift_ev
                if sonication_chemical_potential_shift_ev is None
                else sonication_chemical_potential_shift_ev
            ),
            calibrated=False,
            calibration_objective=None,
            calibration_scope="candidate under calibration",
        )

    def write(self, filename: Path, metadata: dict | None = None) -> None:
        filename.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "rate_unit": "s^-1",
            "energy_unit": "eV",
            "parameters": asdict(self),
            "metadata": metadata or {},
        }
        filename.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def read(cls, filename: Path) -> "KineticParameterSet":
        payload = json.loads(filename.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported kinetic-parameter schema")
        if payload.get("rate_unit") != "s^-1":
            raise ValueError("kinetic parameter rates must use s^-1")
        parameters = dict(payload["parameters"])
        # Read parameter files written by the former dynamic-chemical-
        # potential implementation without restoring that feedback model.
        parameters.pop("maximum_chemical_potential_ce_o_ev", None)
        parameters.pop("excess_reservoir_saturation_fraction", None)
        return cls(**parameters)

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


# Diagnostic precursor dose requested for the standard 20 nm box / 5 nm
# support.  Table S9 still defines the separate final supported-Ir target
# (about 248 atoms for this particle); 248/600 is therefore the nominal
# capture fraction required to reach that target from the enlarged dose.
STANDARD_5NM_IR_PRECURSOR_ATOMS = 600
STANDARD_5NM_SUPPORTED_IR_TARGET_ATOMS = 248


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
    # With the former 248-atom dose and 2.0 s^-1 diffusion, the corrected
    # standard run captured about 41 Ir on the sonicated main particle
    # (16.5%).  A 600-atom dose needs 248/600 = 41.3% capture to meet the
    # Table-S9 target.  Linear first-order scaling gives ~5.0 s^-1 as the next
    # diagnostic estimate.  This unpublished value still requires a complete
    # 20 nm calibration run.
    ir_diffusion_prefactor_s: float = 5.0
    ir_reduction_prefactor_s: float = 5.0e-1
    ir_oxidation_prefactor_s: float = 1.0e-6
    ir_adsorption_barrier_ev: float = 0.0
    ir_desorption_barrier_ev: float = 0.0
    ir_diffusion_barrier_ev: float = 0.0
    ir_reduction_barrier_ev: float = 0.0
    ir_oxidation_barrier_ev: float = 0.0
    chemical_potential_ir_ion_ev: float = -0.15
    reduction_free_energy_ev: float = -0.10
    # Table S9 defines the retained supported-Ir target.  Keep that target
    # separate from the larger precursor dose: for the standard 5 nm support,
    # 248 target atoms / 600 precursor atoms gives the nominal capture factor.
    # This 600-atom dose is a user-selected diagnostic setting, not a value
    # published in the supplement.
    precursor_ir_to_ce_atom_ratio: float = PAPER_RIE_IR_TO_CE_ATOM_RATIO
    precursor_retention_fraction: float = (
        STANDARD_5NM_SUPPORTED_IR_TARGET_ATOMS
        / STANDARD_5NM_IR_PRECURSOR_ATOMS
    )

    # Independent acoustic-condition clock per eligible interface center.
    # This is deliberately not a chemical KMC reaction propensity.  The
    # user-requested fivefold diagnostic increase gives an initial total rate
    # near 1e-2 s^-1 for roughly 1,000 surface centers.
    sonication_event_rate_s: float = 1.0e-5
    # Retained only for backward-compatible parameter-file loading.  The
    # formal model fixes both Ce/O chemical potentials at -0.60 eV, so this
    # deprecated offset has no effect.
    sonication_chemical_potential_shift_ev: float = 0.0
    calibrated: bool = False
    calibration_objective: float | None = None
    calibration_scope: str = "initial guesses only"

    def ceox_parameters(self, sonication: bool = False) -> CeOxParameters:
        # Sonication is an external condition clock, not a bath-composition
        # change.  Accept the argument for API compatibility while using the
        # same fixed chemical potentials for both conditions.
        return CeOxParameters(
            temperature_k=PAPER_TEMPERATURE_K,
            ce_o_binding_energy_ev=DFT_CE_O_BINDING_ENERGY_EV,
            ir_o_binding_energy_ev=DFT_IR_O_BINDING_ENERGY_EV,
            chemical_potential_ce_ev=self.chemical_potential_ce_ev,
            chemical_potential_o_ev=self.chemical_potential_o_ev,
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

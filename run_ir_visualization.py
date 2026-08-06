"""Generate an Ir-containing XYZ trajectory for visual inspection in OVITO.

The Ir kinetic parameters below are deliberately chosen only to make every Ir
event visible in a short demonstration.  They are not fitted paper parameters
and must not be used for scientific interpretation.
"""

from datetime import datetime
from pathlib import Path

import numpy as np

from ceox_events import CeOxParameters
from generation import initialize_sphere, roughen_surface
from ir_events import IrParameters
from kmc_engine import KMCRunConfig, run_KMC
from lattice_build import build_fluorite_lattice
from paper_parameters import (
    DFT_CE_O_BINDING_ENERGY_EV,
    PAPER_CHEMICAL_POTENTIAL_CE_EV,
    PAPER_CHEMICAL_POTENTIAL_O_EV,
    PAPER_TEMPERATURE_K,
)


def main():
    random_seed = 2026
    geometry_rng = np.random.default_rng(random_seed)

    # A compact box puts the solution boundary close to the 5 nm particle, so
    # Ir ions can reach the support within a few thousand demonstration steps.
    lattice = build_fluorite_lattice(ncells=11)
    initialize_sphere(
        lattice=lattice,
        diameter_nm=5.0,
        oxygen_x=2.0,
        rng=geometry_rng,
    )
    roughen_surface(lattice=lattice, fraction=0.05, rng=geometry_rng)

    ceox_parameters = CeOxParameters(
        temperature_k=PAPER_TEMPERATURE_K,
        ce_o_binding_energy_ev=DFT_CE_O_BINDING_ENERGY_EV,
        chemical_potential_ce_ev=PAPER_CHEMICAL_POTENTIAL_CE_EV,
        chemical_potential_o_ev=PAPER_CHEMICAL_POTENTIAL_O_EV,
        adsorption_prefactor=1.0,
        desorption_prefactor=1.0,
        exchange_barrier_ev=0.0,
    )

    # VISUALIZATION-ONLY values.  Fast ion diffusion and slower reduction let
    # Ir ions travel from the reservoir boundary toward the CeOx surface before
    # becoming immobile metallic Ir.
    ir_parameters = IrParameters(
        ir_ir_binding_energy_ev=0.10,
        ir_o_binding_energy_ev=0.20,
        chemical_potential_ir_ion_ev=-0.15,
        reduction_free_energy_ev=-0.10,
        temperature_k=PAPER_TEMPERATURE_K,
        adsorption_prefactor=1.0,
        desorption_prefactor=0.1,
        diffusion_prefactor=100.0,
        reduction_prefactor=0.1,
        oxidation_prefactor=0.01,
        adsorption_barrier_ev=0.0,
        desorption_barrier_ev=0.0,
        diffusion_barrier_ev=0.0,
        reduction_barrier_ev=0.0,
        oxidation_barrier_ev=0.0,
    )

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = Path("kmc_output") / f"ir_visualization_{run_stamp}"
    run_config = KMCRunConfig(
        number_of_steps=3000,
        snapshot_every=50,
        metrics_every=50,
        random_seed=random_seed,
        output_directory=str(output_directory),
    )

    state, metrics = run_KMC(
        lattice=lattice,
        parameters=ceox_parameters,
        run_config=run_config,
        ir_parameters=ir_parameters,
    )

    final_metrics = metrics[-1]
    print(f"Output directory: {output_directory.resolve()}")
    print(f"Completed steps: {state.step}")
    print(f"Stopped reason: {state.stopped_reason or 'completed normally'}")
    print(f"Final Ir ions: {final_metrics['number_Ir_ion']}")
    print(f"Final metallic Ir: {final_metrics['number_Ir']}")
    print("Open snapshot_00000000.xyz as a file sequence in OVITO.")


if __name__ == "__main__":
    main()

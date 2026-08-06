"""Parameters traceable to Shi et al., Science 387 (2025), adr3149.

Keep paper/DFT-derived values here so demonstration-only settings do not get
mistaken for calibrated inputs.  A value being reported by the paper does not
necessarily mean it was obtained by DFT; the provenance category is explicit.
"""

PAPER_DOI = "10.1126/science.adr3149"

# DFT-derived lattice-gas energy used by the paper's KMC description.
DFT_CE_O_BINDING_ENERGY_EV = 0.30

# Reported KMC/synthesis settings; these are not labelled as DFT energies.
PAPER_TEMPERATURE_K = 453.0
PAPER_CHEMICAL_POTENTIAL_CE_EV = -0.60
PAPER_CHEMICAL_POTENTIAL_O_EV = -0.60
PAPER_BOX_NM = 20.0
PAPER_PARTICLE_DIAMETER_NM = 5.0
PAPER_SNAPSHOT_STEPS = (0, 1_000_000, 3_000_000, 5_000_000)
PAPER_SONICATION_RADIUS_NM = 1.0
PAPER_DISSOLUTION_PROBABILITY = 0.10

# No directly mappable DFT values were found in the accessible paper record
# for Ir-Ir/Ir-O binding, Ir adsorption/diffusion/redox barriers, or event
# prefactors.  Those must remain explicitly labelled demonstration parameters.
UNRESOLVED_DFT_PARAMETERS = (
    "ir_ir_binding_energy_ev",
    "ir_o_binding_energy_ev",
    "chemical_potential_ir_ion_ev",
    "reduction_free_energy_ev",
    "ir_adsorption_barrier_ev",
    "ir_desorption_barrier_ev",
    "ir_diffusion_barrier_ev",
    "ir_reduction_barrier_ev",
    "ir_oxidation_barrier_ev",
)

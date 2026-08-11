"""Parameters traceable to Shi et al., Science 387 (2025), adr3149.

Keep paper/DFT-derived values here so demonstration-only settings do not get
mistaken for calibrated inputs.  A value being reported by the paper does not
necessarily mean it was obtained by DFT; the provenance category is explicit.
"""

PAPER_DOI = "10.1126/science.adr3149"

# Values reported in the supplementary KMC/DFT methods.  The supplement uses
# both binding-strength and Hamiltonian-energy sign conventions: it reports
# positive average Ir-Ir/Ce-O binding strengths, but an Ir-O interface value
# of -0.05 eV.  This implementation stores positive stabilization strengths
# and subtracts them when forming a bond, so the reported Ir-O value is
# sign-converted before entering the local binding expression.
DFT_CE_O_BINDING_ENERGY_EV = 0.30
DFT_IR_IR_BINDING_ENERGY_EV = 0.32
PAPER_REPORTED_IR_O_BINDING_ENERGY_EV = -0.05
DFT_IR_O_BINDING_ENERGY_EV = abs(PAPER_REPORTED_IR_O_BINDING_ENERGY_EV)

# Reported KMC/synthesis settings; these are not labelled as DFT energies.
PAPER_TEMPERATURE_K = 453.0
PAPER_CHEMICAL_POTENTIAL_CE_EV = -0.60
PAPER_CHEMICAL_POTENTIAL_O_EV = -0.60
PAPER_BOX_NM = 20.0
PAPER_PARTICLE_DIAMETER_NM = 5.0
PAPER_TARGET_TIMES_MIN = (0.0, 5.0, 30.0, 60.0, 120.0, 180.0)
PAPER_SNAPSHOT_STEPS = (0, 1_000_000, 3_000_000, 5_000_000)
PAPER_SONICATION_RADIUS_NM = 1.0
PAPER_DISSOLUTION_PROBABILITY = 0.10

# ICP-OES composition of the final RIE-Ir/CeOx catalyst (Table S9).  The
# atom-number ratio is used to scale the finite Ir precursor inventory to the
# amount of CeOx represented by a simulation particle.  Using the measured Ce
# fraction avoids assuming that the support is exactly stoichiometric CeO2.
PAPER_RIE_IR_MASS_PERCENT = 15.60
PAPER_RIE_CE_MASS_PERCENT = 74.96
IR_ATOMIC_MASS = 192.217
CE_ATOMIC_MASS = 140.116
PAPER_RIE_IR_TO_CE_ATOM_RATIO = (
    (PAPER_RIE_IR_MASS_PERCENT / IR_ATOMIC_MASS)
    / (PAPER_RIE_CE_MASS_PERCENT / CE_ATOMIC_MASS)
)

# The supplement says these kinetic quantities were fitted, but does not
# publish their numerical values.  They remain explicit calibration inputs.
UNPUBLISHED_FITTED_PARAMETERS = (
    "ir_adsorption_prefactor_s",
    "ir_desorption_prefactor_s",
    "ir_diffusion_prefactor_s",
    "ir_reduction_prefactor_s",
    "ir_oxidation_prefactor_s",
    "chemical_potential_ir_ion_ev",
    "reduction_free_energy_ev",
    "ir_adsorption_barrier_ev",
    "ir_desorption_barrier_ev",
    "ir_diffusion_barrier_ev",
    "ir_reduction_barrier_ev",
    "ir_oxidation_barrier_ev",
    "sonication_event_rate_s",
    "sonication_chemical_potential_shift_ev",
)

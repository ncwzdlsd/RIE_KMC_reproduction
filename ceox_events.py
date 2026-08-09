from dataclasses import dataclass
from enum import Enum
import math

from paper_parameters import (
    DFT_CE_O_BINDING_ENERGY_EV,
    DFT_IR_O_BINDING_ENERGY_EV,
    PAPER_CHEMICAL_POTENTIAL_CE_EV,
    PAPER_CHEMICAL_POTENTIAL_O_EV,
    PAPER_TEMPERATURE_K,
)

K_EV_PER_K = 8.617333262145e-5


class EventType(str, Enum):
    CE_ADSORPTION = "Ce_ADSORPTION"
    CE_DESORPTION = "Ce_DESORPTION"
    O_ADSORPTION = "O_ADSORPTION"
    O_DESORPTION = "O_DESORPTION"


@dataclass(frozen=True)
class CeOxParameters:
    temperature_k: float = PAPER_TEMPERATURE_K
    ce_o_binding_energy_ev: float = DFT_CE_O_BINDING_ENERGY_EV
    ir_o_binding_energy_ev: float = DFT_IR_O_BINDING_ENERGY_EV
    chemical_potential_ce_ev: float = PAPER_CHEMICAL_POTENTIAL_CE_EV
    chemical_potential_o_ev: float = PAPER_CHEMICAL_POTENTIAL_O_EV
    adsorption_prefactor: float = 1.0
    desorption_prefactor: float = 1.0
    exchange_barrier_ev: float = 0.0


def transition_rate(
    delta_omega_ev: float,
    prefactor: float,
    parameters: CeOxParameters,
) -> float:
    thermal_energy_ev = K_EV_PER_K * parameters.temperature_k
    total_barrier_ev = max(0.0, delta_omega_ev) + parameters.exchange_barrier_ev
    return prefactor * math.exp(-total_barrier_ev / thermal_energy_ev)


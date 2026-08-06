import numpy as np
from collections import Counter
from enum import Enum
from dataclasses import dataclass
import math
from constants import SiteType,Species
from lattice_build import (Lattice,build_fluorite_lattice)
from generation import (initialize_sphere,find_external_surface,find_accessible_empty_sites)
from paper_parameters import (
    DFT_CE_O_BINDING_ENERGY_EV,
    PAPER_CHEMICAL_POTENTIAL_CE_EV,
    PAPER_CHEMICAL_POTENTIAL_O_EV,
    PAPER_TEMPERATURE_K,
)

k_ev_per_k=8.617333262145e-5

class EventType(str,Enum):
    CE_ADSORPTION="Ce_ADSORPTION"
    CE_DESORPTION="Ce_DESORPTION"
    O_ADSORPTION="O_ADSORPTION"
    O_DESORPTION="O_DESORPTION"

@dataclass(frozen=True)
class CeOxParameters:
    temperature_k:float=PAPER_TEMPERATURE_K
    ce_o_binding_energy_ev:float=DFT_CE_O_BINDING_ENERGY_EV
    chemical_potential_ce_ev:float=PAPER_CHEMICAL_POTENTIAL_CE_EV
    chemical_potential_o_ev:float=PAPER_CHEMICAL_POTENTIAL_O_EV
    adsorption_prefactor:float=1.0
    desorption_prefactor:float=1.0
    exchange_barrier_ev:float=0.0

@dataclass(frozen=True)
class CeOxEvent:
    event_type:EventType
    site_id:int
    species:Species
    coordination:int
    binding_energy_ev:float
    rate:float
    delta_omega_ev:float # change in grand potential for the event

def count_o_neighbors(lattice:Lattice,m_site_id:int) -> int:
    neighbor_ids=lattice.get_ce_o_neighbors(m_site_id)
    return np.count_nonzero(lattice.occupation[neighbor_ids]==Species.O)

def count_ce_neighbors(lattice:Lattice,o_site_id:int) -> int:
    neighbor_ids=lattice.get_ce_o_neighbors(o_site_id)
    return np.count_nonzero(lattice.occupation[neighbor_ids]==Species.CE)

def local_coordination(lattice:Lattice,site_id:int,species:Species):
    if species==Species.CE:
        return count_o_neighbors(lattice,site_id)
    if species==Species.O:
        return count_ce_neighbors(lattice,site_id)

def local_binding_energy(coordination:int,parameters:CeOxParameters) -> float:
    return coordination*parameters.ce_o_binding_energy_ev

def chemical_potential(species:Species,parameters:CeOxParameters) -> float:
    if species==Species.CE:
        return parameters.chemical_potential_ce_ev
    if species==Species.O:
        return parameters.chemical_potential_o_ev

def adsorption_delta_omega(binding_energy_ev:float,chemical_potential_ev:float) -> float:
    return (-chemical_potential_ev-binding_energy_ev)

def desorption_delta_omega(binding_energy_ev:float,chemical_potential_ev:float) -> float:
    return (binding_energy_ev+chemical_potential_ev)

def transition_rate(delta_omega_ev:float,prefactor:float,parameters:CeOxParameters) -> float:
    thermal_energy_ev=k_ev_per_k*parameters.temperature_k
    total_barrier_ev=max(0.0,delta_omega_ev)+parameters.exchange_barrier_ev
    rate=prefactor*math.exp(-total_barrier_ev/thermal_energy_ev)
    return rate

def build_ce_adsorption_event(lattice:Lattice,site_id:int,parameters:CeOxParameters,accessible:np.ndarray) -> CeOxEvent:
    if(lattice.site_type[site_id]!=SiteType.M):return None
    if(lattice.occupation[site_id]!=Species.EMPTY):return None
    if not accessible[site_id]:return None
    binding_energy_ev=local_binding_energy(count_o_neighbors(lattice,site_id),parameters)
    delta_omega_ev=adsorption_delta_omega(binding_energy_ev,chemical_potential(Species.CE,parameters))
    rate=transition_rate(delta_omega_ev,parameters.adsorption_prefactor,parameters)
    return CeOxEvent(
        event_type=EventType.CE_ADSORPTION,
        site_id=site_id,
        species=Species.CE,
        coordination=count_o_neighbors(lattice,site_id),
        binding_energy_ev=binding_energy_ev,
        rate=rate,
        delta_omega_ev=delta_omega_ev
    )

def build_ce_desorption_event(lattice:Lattice,site_id:int,parameters:CeOxParameters,accessible:np.ndarray) -> CeOxEvent:
    if(lattice.site_type[site_id]!=SiteType.M):return None
    if(lattice.occupation[site_id]!=Species.CE):return None
    if not accessible[site_id]:return None
    binding_energy_ev=local_binding_energy(count_o_neighbors(lattice,site_id),parameters)
    delta_omega_ev=desorption_delta_omega(binding_energy_ev,chemical_potential(Species.CE,parameters))
    rate=transition_rate(delta_omega_ev,parameters.desorption_prefactor,parameters)
    return CeOxEvent(
        event_type=EventType.CE_DESORPTION,
        site_id=site_id,
        species=Species.CE,
        coordination=count_o_neighbors(lattice,site_id),
        binding_energy_ev=binding_energy_ev,
        rate=rate,
        delta_omega_ev=delta_omega_ev
    )

def build_o_adsorption_event(lattice:Lattice,site_id:int,parameters:CeOxParameters,accessible:np.ndarray) -> CeOxEvent:
    if(lattice.site_type[site_id]!=SiteType.O):return None
    if(lattice.occupation[site_id]!=Species.EMPTY):return None
    if not accessible[site_id]:return None
    binding_energy_ev=local_binding_energy(count_ce_neighbors(lattice,site_id),parameters)
    delta_omega_ev=adsorption_delta_omega(binding_energy_ev,chemical_potential(Species.O,parameters))
    rate=transition_rate(delta_omega_ev,parameters.adsorption_prefactor,parameters)
    return CeOxEvent(
        event_type=EventType.O_ADSORPTION,
        site_id=site_id,
        species=Species.O,
        coordination=count_ce_neighbors(lattice,site_id),
        binding_energy_ev=binding_energy_ev,
        rate=rate,
        delta_omega_ev=delta_omega_ev
    )

def build_o_desorption_event(lattice:Lattice,site_id:int,parameters:CeOxParameters,accessible:np.ndarray) -> CeOxEvent:
    if(lattice.site_type[site_id]!=SiteType.O):return None
    if(lattice.occupation[site_id]!=Species.O):return None
    if not accessible[site_id]:return None
    binding_energy_ev=local_binding_energy(count_ce_neighbors(lattice,site_id),parameters)
    delta_omega_ev=desorption_delta_omega(binding_energy_ev,chemical_potential(Species.O,parameters))
    rate=transition_rate(delta_omega_ev,parameters.desorption_prefactor,parameters)
    return CeOxEvent(
        event_type=EventType.O_DESORPTION,
        site_id=site_id,
        species=Species.O,
        coordination=count_ce_neighbors(lattice,site_id),
        binding_energy_ev=binding_energy_ev,
        rate=rate,
        delta_omega_ev=delta_omega_ev
    )

def build_CeOx_event_catalog(
    lattice:Lattice,
    parameters:CeOxParameters,
    accessible:np.ndarray | None=None,
    external_surface:np.ndarray | None=None,
) -> list[CeOxEvent]:
    if accessible is None:
        accessible=find_accessible_empty_sites(lattice)
    if external_surface is None:
        external_surface=find_external_surface(lattice)
    events=[]
    for site_id in range(lattice.nsites):
        if lattice.site_type[site_id]==SiteType.M:
            if lattice.occupation[site_id]==Species.EMPTY:
                event=build_ce_adsorption_event(lattice,site_id,parameters,accessible)
                if event is not None:
                    events.append(event)
            elif lattice.occupation[site_id]==Species.CE:
                event=build_ce_desorption_event(lattice,site_id,parameters,external_surface)
                if event is not None:
                    events.append(event)
        elif lattice.site_type[site_id]==SiteType.O:
            if lattice.occupation[site_id]==Species.EMPTY:
                event=build_o_adsorption_event(lattice,site_id,parameters,accessible)
                if event is not None:
                    events.append(event)
            elif lattice.occupation[site_id]==Species.O:
                event=build_o_desorption_event(lattice,site_id,parameters,external_surface)
                if event is not None:
                    events.append(event)
    return events

def apply_CeOx_event(lattice:Lattice,event:CeOxEvent):
    if(event.event_type==EventType.CE_ADSORPTION):
        lattice.occupation[event.site_id]=Species.CE
    elif(event.event_type==EventType.CE_DESORPTION):
        lattice.occupation[event.site_id]=Species.EMPTY
    elif(event.event_type==EventType.O_ADSORPTION):
        lattice.occupation[event.site_id]=Species.O
    elif(event.event_type==EventType.O_DESORPTION):
        lattice.occupation[event.site_id]=Species.EMPTY


from collections import Counter
from dataclasses import dataclass,field
from dataclasses import replace
from constants import Species
from pathlib import Path
import csv
import json
import numpy as np
import math
from lattice_build import Lattice,build_fluorite_lattice
from generation import initialize_sphere,roughen_surface,find_accessible_empty_sites,write_xyz,find_external_surface
from ceox_events import (CeOxParameters,CeOxEvent,EventType,build_CeOx_event_catalog,apply_CeOx_event)
from ir_events import (
    IrEvent,
    IrEventType,
    IrParameters,
    apply_ir_event,
    build_ir_event_catalog,
)
from sonication_events import (
    SonicationEvent,
    SonicationEventType,
    SonicationParameters,
    apply_mean_field_ripening_growth,
    apply_sonication_event,
    build_sonication_event,
)
from paper_parameters import (
    DFT_CE_O_BINDING_ENERGY_EV,
    PAPER_CHEMICAL_POTENTIAL_CE_EV,
    PAPER_CHEMICAL_POTENTIAL_O_EV,
    PAPER_TEMPERATURE_K,
)

KMCEvent = CeOxEvent | IrEvent | SonicationEvent

@dataclass(frozen=True)
class KMCRunConfig:
    number_of_steps:int=100
    snapshot_every:int=10
    metrics_every:int=10
    random_seed:int=2026
    output_directory:str="kmc_output"

    def __post_init__(self):
        if self.number_of_steps < 0:
            raise ValueError("number_of_steps must be non-negative")
        if self.snapshot_every <= 0:
            raise ValueError("snapshot_every must be positive")
        if self.metrics_every <= 0:
            raise ValueError("metrics_every must be positive")

@dataclass
class KMCState:
    step:int=0
    kmc_time:float=0.0
    event_counts:Counter=field(default_factory=Counter)
    sonication_removed_atoms:int=0
    sonication_redeposited_atoms:int=0
    solution_chemical_potential_boost_ev:float=0.0
    stopped_reason:str=""
    def __post_init__(self):
        if self.event_counts is None:
            self.event_counts=Counter

def select_event(events:list[KMCEvent],rng:np.random.Generator):
    if not events:
        raise RuntimeError("no available KMC events")

    rates=np.fromiter((event.rate for event in events),dtype=np.float64,count=len(events))
    if np.any(~np.isfinite(rates)) or np.any(rates < 0.0):
        raise RuntimeError("KMC event rates must be finite and non-negative")

    total_rate=float(rates.sum())
    if not math.isfinite(total_rate) or total_rate <= 0.0:
        raise RuntimeError("total KMC event rate must be finite and positive")

    cumulative_rates=np.cumsum(rates)
    target=(rng.random()*total_rate)
    selected_index=int(np.searchsorted(cumulative_rates,target,side="right"))
    selected_index=min(selected_index,len(events)-1)
    selected_event=events[selected_index]
    return selected_event,total_rate

def sample_time_increment(total_rate:float,rng:np.random.Generator):
    random_number=max(rng.random(),np.finfo(np.float64).tiny)
    delta_time=(-math.log(random_number)/total_rate)
    return delta_time

def calculate_current(lattice:Lattice,state:KMCState):
    number_ce=int(np.count_nonzero(lattice.occupation==Species.CE))
    number_o=int(np.count_nonzero(lattice.occupation==Species.O))
    number_ir_ion=int(np.count_nonzero(lattice.occupation==Species.IR_ION))
    number_ir=int(np.count_nonzero(lattice.occupation==Species.IR))
    number_tot=number_ce+number_o+number_ir_ion+number_ir
    external_surface=find_external_surface(lattice)
    surface_ce=int(np.count_nonzero(external_surface&(lattice.occupation==Species.CE)))
    surface_o=int(np.count_nonzero(external_surface&(lattice.occupation==Species.O)))
    surface_ir_ion=int(np.count_nonzero(external_surface&(lattice.occupation==Species.IR_ION)))
    surface_ir=int(np.count_nonzero(external_surface&(lattice.occupation==Species.IR)))
    embedded_ir_ion=number_ir_ion-surface_ir_ion
    embedded_ir=number_ir-surface_ir
    total_ir=number_ir_ion+number_ir
    embedded_ir_total=embedded_ir_ion+embedded_ir
    ir_site_ids=np.flatnonzero(
        (lattice.occupation==Species.IR_ION)
        |(lattice.occupation==Species.IR)
    )
    ir_support_contact_counts=[]
    for site_id in ir_site_ids:
        site_id=int(site_id)
        o_neighbors=lattice.get_ce_o_neighbors(site_id)
        m_neighbors=lattice.get_m_m_neighbors(site_id)
        ir_support_contact_counts.append(
            int(np.count_nonzero(lattice.occupation[o_neighbors]==Species.O))
            +int(np.count_nonzero(lattice.occupation[m_neighbors]==Species.CE))
        )
    mean_ir_support_contacts=(
        float(np.mean(ir_support_contact_counts))
        if ir_support_contact_counts else 0.0
    )
    highly_covered_ir_count=int(
        np.count_nonzero(np.asarray(ir_support_contact_counts)>=8)
    )

    volume_per_Ce_nm3=lattice.lattice_constant_nm**3/4.0
    particle_volume_nm3=number_ce*volume_per_Ce_nm3
    equivalent_radius_nm=(3.0*particle_volume_nm3/(4.0*math.pi))**(1.0/3.0)
    equivalent_diameter_nm=2.0*equivalent_radius_nm

    return{
        "step":state.step,
        "KMC_time":state.kmc_time,
        "number_Ce":number_ce,
        "number_O":number_o,
        "number_Ir_ion":number_ir_ion,
        "number_Ir":number_ir,
        "number_total":number_tot,
        "surface_Ce":surface_ce,
        "surface_O":surface_o,
        "surface_Ir_ion":surface_ir_ion,
        "surface_Ir":surface_ir,
        "embedded_Ir_ion":embedded_ir_ion,
        "embedded_Ir":embedded_ir,
        "embedded_Ir_total":embedded_ir_total,
        "Ir_embedding_fraction":embedded_ir_total/total_ir if total_ir else 0.0,
        "mean_Ir_support_contacts":mean_ir_support_contacts,
        "highly_covered_Ir_count":highly_covered_ir_count,
        "highly_covered_Ir_fraction":highly_covered_ir_count/total_ir if total_ir else 0.0,
        "equivalent_diameter_nm":equivalent_diameter_nm,
        "Ce_adsorption_count":state.event_counts[EventType.CE_ADSORPTION.value],
        "Ce_desorption_count":state.event_counts[EventType.CE_DESORPTION.value],
        "O_adsorption_count":state.event_counts[EventType.O_ADSORPTION.value],
        "O_desorption_count":state.event_counts[EventType.O_DESORPTION.value],
        "Ir_ion_adsorption_count":state.event_counts[IrEventType.IR_ION_ADSORPTION.value],
        "Ir_ion_desorption_count":state.event_counts[IrEventType.IR_ION_DESORPTION.value],
        "Ir_ion_diffusion_count":state.event_counts[IrEventType.IR_ION_DIFFUSION.value],
        "Ir_reduction_count":state.event_counts[IrEventType.IR_REDUCTION.value],
        "Ir_oxidation_count":state.event_counts[IrEventType.IR_OXIDATION.value],
        "sonication_event_count":state.event_counts[SonicationEventType.CORROSION.value],
        "sonication_removed_atoms":state.sonication_removed_atoms,
        "sonication_redeposited_atoms":state.sonication_redeposited_atoms,
        "solution_chemical_potential_boost_ev":state.solution_chemical_potential_boost_ev,
    }

def write_snapshot(output_directory,lattice:Lattice,state:KMCState):
    output_directory=Path(output_directory)
    filename=(output_directory/(f"snapshot_"f"{state.step:08d}.xyz"))
    write_xyz(filename,lattice,comment=(f"step={state.step} "f"KMC_time={state.kmc_time:.10e}"))

def write_metrics_csv(filename,metrics_rows):
    filename=Path(filename)
    filename.parent.mkdir(parents=True,exist_ok=True)
    with filename.open("w",encoding="utf-8",newline="") as file:
        field_names=list(metrics_rows[0].keys())
        writer=csv.DictWriter(file,fieldnames=field_names)
        writer.writeheader()
        writer.writerows(metrics_rows)

def write_event_counts(filename,state:KMCState):
    filename=Path(filename)
    filename.parent.mkdir(parents=True,exist_ok=True)
    data={
        "step":state.step,
        "KMC_time":state.kmc_time,
        "stopped_reason":state.stopped_reason,
        "sonication_removed_atoms":state.sonication_removed_atoms,
        "sonication_redeposited_atoms":state.sonication_redeposited_atoms,
        "solution_chemical_potential_boost_ev":state.solution_chemical_potential_boost_ev,
        "event_counts":dict(state.event_counts),
    }
    filename.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding="utf-8")

def run_KMC(
    lattice:Lattice,
    parameters:CeOxParameters,
    run_config:KMCRunConfig,
    ir_parameters:IrParameters | None=None,
    sonication_parameters:SonicationParameters | None=None,
):
    rng=np.random.default_rng(run_config.random_seed)
    output_directory=Path(run_config.output_directory)
    output_directory.mkdir(parents=True,exist_ok=True)
    state=KMCState()
    metrics_rows=[]
    initial_metrics=(calculate_current(lattice,state))
    metrics_rows.append(initial_metrics)
    write_snapshot(output_directory,lattice,state)

    for target_step in range(1,run_config.number_of_steps+1):
        current_parameters=parameters
        if sonication_parameters is not None:
            exposure_fraction=min(
                1.0,
                state.event_counts[SonicationEventType.CORROSION.value]
                / sonication_parameters.events_for_maximum_boost,
            )
            state.solution_chemical_potential_boost_ev=(
                exposure_fraction
                * sonication_parameters.maximum_chemical_potential_boost_ev
            )
            current_parameters=replace(
                parameters,
                chemical_potential_ce_ev=(
                    parameters.chemical_potential_ce_ev
                    + state.solution_chemical_potential_boost_ev
                ),
                chemical_potential_o_ev=(
                    parameters.chemical_potential_o_ev
                    + state.solution_chemical_potential_boost_ev
                ),
            )
        accessible=find_accessible_empty_sites(lattice)
        external_surface=find_external_surface(lattice, accessible)
        events: list[KMCEvent]=list(
            build_CeOx_event_catalog(
                lattice,
                current_parameters,
                accessible=accessible,
                external_surface=external_surface,
            )
        )
        if ir_parameters is not None:
            events.extend(build_ir_event_catalog(lattice,ir_parameters))
        if sonication_parameters is not None:
            sonication_event=build_sonication_event(
                lattice,
                sonication_parameters,
                external_surface,
            )
            if sonication_event is not None:
                events.append(sonication_event)
        try:
            selected_event,total_rate=(select_event(events,rng))
        except RuntimeError as error:
            state.stopped_reason=str(error)
            break
        delta_time=(sample_time_increment(total_rate,rng))
        if isinstance(selected_event,CeOxEvent):
            apply_CeOx_event(lattice,selected_event)
        elif isinstance(selected_event,IrEvent):
            apply_ir_event(lattice,selected_event)
        elif isinstance(selected_event,SonicationEvent):
            removed_site_ids=apply_sonication_event(
                lattice,
                selected_event,
                sonication_parameters,
                rng,
            )
            state.sonication_removed_atoms+=len(removed_site_ids)
            added_site_ids=apply_mean_field_ripening_growth(
                lattice,
                sonication_parameters,
                rng,
            )
            state.sonication_redeposited_atoms+=len(added_site_ids)
        else:
            raise TypeError(f"unsupported KMC event: {type(selected_event).__name__}")
        state.step=target_step
        state.kmc_time+=delta_time
        state.event_counts[selected_event.event_type.value]+=1
        if(state.step%run_config.metrics_every==0):
            metrics_rows.append(calculate_current(lattice,state))
        if(state.step%run_config.snapshot_every==0):
            write_snapshot(output_directory,lattice,state)

    if (len(metrics_rows)==0 or metrics_rows[-1]["step"]!=state.step):
        metrics_rows.append(calculate_current(lattice,state))
    final_snapshot=(output_directory/(f"snapshot_"f"{state.step:08d}.xyz"))
    if not final_snapshot.exists():
        write_snapshot(output_directory,lattice,state)

    write_metrics_csv(output_directory/"metrics.csv",metrics_rows)
    write_event_counts(output_directory/"event_counts.json",state)

    return state,metrics_rows

def main():
    geometry_rng=np.random.default_rng(2026)
    lattice=build_fluorite_lattice(ncells=20)
    initialize_sphere(lattice=lattice,diameter_nm=5.0,oxygen_x=2.0,rng=geometry_rng)
    roughen_surface(lattice=lattice,fraction=0.05,rng=geometry_rng)
    parameters=CeOxParameters(
        temperature_k=PAPER_TEMPERATURE_K,
        ce_o_binding_energy_ev=DFT_CE_O_BINDING_ENERGY_EV,
        chemical_potential_ce_ev=PAPER_CHEMICAL_POTENTIAL_CE_EV,
        chemical_potential_o_ev=PAPER_CHEMICAL_POTENTIAL_O_EV,
        adsorption_prefactor=1.0,
        desorption_prefactor=1.0,
        exchange_barrier_ev=0.0,
    )
    run_config=KMCRunConfig(number_of_steps=200,snapshot_every=50,metrics_every=50,random_seed=2026,output_directory=("kmc_output/""test"))
    run_KMC(lattice=lattice,parameters=parameters,run_config=run_config)

if __name__=="__main__":
    main()

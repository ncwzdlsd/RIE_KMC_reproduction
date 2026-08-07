from collections import Counter
from dataclasses import dataclass,field
from constants import Species
from pathlib import Path
import csv
import json
import numpy as np
import math
from lattice_build import Lattice
from generation import (
    find_accessible_empty_sites,
    find_external_surface,
    find_supported_ir_sites,
    ir_support_contact_count,
    write_xyz,
)
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
    apply_sonication_event,
    build_sonication_event,
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
    supported_ir=find_supported_ir_sites(lattice)
    attached_ir_ion=int(np.count_nonzero(supported_ir&(lattice.occupation==Species.IR_ION)))
    attached_ir=int(np.count_nonzero(supported_ir&(lattice.occupation==Species.IR)))
    attached_ir_total=attached_ir_ion+attached_ir
    ir_support_contact_counts=[]
    for site_id in ir_site_ids:
        site_id=int(site_id)
        ir_support_contact_counts.append(ir_support_contact_count(lattice,site_id))
    mean_ir_support_contacts=(
        float(np.mean(ir_support_contact_counts))
        if ir_support_contact_counts else 0.0
    )
    highly_covered_ir_count=int(
        np.count_nonzero(np.asarray(ir_support_contact_counts)>=8)
    )

    ir_mask=(lattice.occupation==Species.IR_ION)|(lattice.occupation==Species.IR)
    ir_neighbor_counts=[]
    for site_id in ir_site_ids:
        neighbors=lattice.get_m_m_neighbors(int(site_id))
        ir_neighbor_counts.append(int(np.count_nonzero(ir_mask[neighbors])))

    visited=np.zeros(lattice.nsites,dtype=bool)
    cluster_site_ids=[]
    for start in ir_site_ids:
        start=int(start)
        if visited[start]:
            continue
        visited[start]=True
        stack=[start]
        component=[]
        while stack:
            site_id=stack.pop()
            component.append(site_id)
            for neighbor in lattice.get_m_m_neighbors(site_id):
                neighbor=int(neighbor)
                if ir_mask[neighbor] and not visited[neighbor]:
                    visited[neighbor]=True
                    stack.append(neighbor)
        cluster_site_ids.append(component)

    largest_cluster=max(cluster_site_ids,key=len,default=[])
    largest_cluster_radius_gyration_nm=0.0
    largest_cluster_shape_anisotropy=0.0
    if len(largest_cluster)>1:
        positions=lattice.positions_nm[np.asarray(largest_cluster,dtype=np.int32)]
        centered=positions-positions.mean(axis=0)
        gyration_tensor=centered.T@centered/len(positions)
        eigenvalues=np.maximum(np.linalg.eigvalsh(gyration_tensor),0.0)
        eigenvalue_sum=float(eigenvalues.sum())
        largest_cluster_radius_gyration_nm=math.sqrt(eigenvalue_sum)
        if eigenvalue_sum>0.0:
            mean_eigenvalue=eigenvalue_sum/3.0
            largest_cluster_shape_anisotropy=float(
                1.5
                *np.square(eigenvalues-mean_eigenvalue).sum()
                /eigenvalue_sum**2
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
        "attached_Ir_ion":attached_ir_ion,
        "attached_Ir":attached_ir,
        "attached_Ir_total":attached_ir_total,
        "attached_Ir_fraction":attached_ir_total/total_ir if total_ir else 0.0,
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
        "Ir_cluster_count":len(cluster_site_ids),
        "Ir_nanoparticle_count_ge_3":sum(
            len(component)>=3 for component in cluster_site_ids
        ),
        "largest_Ir_cluster_atoms":len(largest_cluster),
        "mean_Ir_Ir_coordination":(
            float(np.mean(ir_neighbor_counts)) if ir_neighbor_counts else 0.0
        ),
        "largest_Ir_cluster_radius_gyration_nm":largest_cluster_radius_gyration_nm,
        "largest_Ir_cluster_shape_anisotropy":largest_cluster_shape_anisotropy,
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
        accessible=find_accessible_empty_sites(lattice)
        external_surface=find_external_surface(lattice, accessible)
        events: list[KMCEvent]=list(
            build_CeOx_event_catalog(
                lattice,
                parameters,
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

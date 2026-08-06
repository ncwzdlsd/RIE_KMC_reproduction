from collections import deque
from pathlib import Path
import numpy as np

from lattice_build import Lattice,SiteType,Species,build_fluorite_lattice

def initialize_sphere(lattice:Lattice,diameter_nm:float=5.0,oxygen_x:float=2.0,rng:np.random.Generator=None):
    if rng is None:
        rng=np.random.default_rng()

    center=lattice.center_nm
    radius=0.5*diameter_nm

    relative_positions=lattice.positions_nm-lattice.center_nm
    distances=np.linalg.norm(relative_positions,axis=1)
    inside_sphere=distances<=radius

    lattice.occupation[:]=Species.EMPTY

    ce_mask=(inside_sphere)&(lattice.site_type==SiteType.M)
    lattice.occupation[ce_mask]=Species.CE

    oxygen_site_ids=np.flatnonzero(inside_sphere&(lattice.site_type==SiteType.O))
    oxygen_occupancy_probability=oxygen_x/2.0
    oxygen_occupied=oxygen_site_ids[rng.random(len(oxygen_site_ids))<oxygen_occupancy_probability]

    lattice.occupation[oxygen_occupied]=Species.O

def support_cordination(lattice:Lattice,site_id:int)->int:
    current_species=Species(lattice.occupation[site_id])
    neighbor_ids=(lattice.get_ce_o_neighbors(site_id))
    neighbor_species=Species(lattice.occupation[neighbor_ids])

    if current_species==Species.CE:
        return np.count_nonzero(neighbor_species==Species.O)
    if current_species==Species.O:
        return np.count_nonzero(neighbor_species==Species.CE)

def find_accessible_empty_sites(lattice:Lattice):
    empty=(lattice.occupation==Species.EMPTY)
    accessible=np.zeros(lattice.nsites,dtype=bool)

    q_min=lattice.q.min(axis=0)
    q_max=lattice.q.max(axis=0)

    boudary_layer=np.any((lattice.q<=q_min+3)|(lattice.q>=q_max-3),axis=1)
    starting_ids=np.flatnonzero(boudary_layer&empty)

    queue=deque()

    for site_id in starting_ids:
        queue.append(site_id)
        accessible[site_id]=True

    while queue:
        site_id=queue.popleft()
        candidate_neighbors=[]
        # M-O cordination
        candidate_neighbors.extend(map(int,lattice.get_ce_o_neighbors(site_id)))
        # M-M cordination
        if(lattice.site_type[site_id]==SiteType.M):
            candidate_neighbors.extend(map(int,lattice.get_m_m_neighbors(site_id)))

        for neighbor_id in candidate_neighbors:
            if empty[neighbor_id] and not accessible[neighbor_id]:
                accessible[neighbor_id]=True
                queue.append(neighbor_id)

    return accessible

def find_external_surface(lattice:Lattice, accessible_empty:np.ndarray | None=None):
    if accessible_empty is None:
        accessible_empty=find_accessible_empty_sites(lattice)
    surface=np.zeros(lattice.nsites,dtype=bool)
    occupied = lattice.occupation != Species.EMPTY
    support_atom_ids=np.flatnonzero(occupied)
    for site_id in support_atom_ids:
        neighbor_ids=(lattice.get_ce_o_neighbors(int(site_id)))
        if(np.any(accessible_empty[neighbor_ids])):
            surface[site_id]=True
    return surface

def roughen_surface(lattice:Lattice,fraction:float=0.05,rng:np.random.Generator=None):
    if rng is None:
        rng=np.random.default_rng()
    if fraction<0.0 or fraction>1.0:
        raise ValueError("fraction must be between 0.0 and 1.0")

    surface=find_external_surface(lattice)
    surface_atom_ids=np.flatnonzero(surface)
    number_to_remove=int(np.ceil(len(surface_atom_ids)*fraction))

    if number_to_remove==0:
        return np.empty(0,dtype=np.int32)

    removal_ids=rng.choice(surface_atom_ids,size=number_to_remove,replace=False)
    lattice.occupation[removal_ids]=Species.EMPTY
    return np.asarray(removal_ids,dtype=np.int32)


def seed_ir_nanoparticle(
    lattice:Lattice,
    center_nm:np.ndarray,
    diameter_nm:float,
) -> np.ndarray:
    """Place a pre-nucleated metallic Ir particle on currently empty M sites."""
    if diameter_nm <= 0.0:
        raise ValueError("diameter_nm must be positive")
    relative_positions=lattice.positions_nm-np.asarray(center_nm,dtype=float)
    within_particle=np.einsum(
        "ij,ij->i",relative_positions,relative_positions
    ) <= (0.5*diameter_nm)**2
    ir_site_ids=np.flatnonzero(
        within_particle
        &(lattice.site_type==SiteType.M)
        &(lattice.occupation==Species.EMPTY)
    )
    lattice.occupation[ir_site_ids]=Species.IR
    return ir_site_ids.astype(np.int32,copy=False)

def geometry_summart(lattice:Lattice):
    number_ce=int(np.count_nonzero(lattice.occupation==Species.CE))
    number_o=int(np.count_nonzero(lattice.occupation==Species.O))
    surface=find_external_surface(lattice)
    number_surface_ce=int(np.count_nonzero(surface&(lattice.occupation==Species.CE)))
    number_surface_o=int(np.count_nonzero(surface&(lattice.occupation==Species.O)))
    occupied_ids=np.flatnonzero((lattice.occupation==Species.CE)|(lattice.occupation==Species.O))

    if len(occupied_ids)>0:
        distances=np.linalg.norm(lattice.positions_nm[occupied_ids]-lattice.center_nm,axis=1)
        geometric_diameter_nm=2.0*distances.max()    
    else:geometric_diameter_nm=0.0

    return {
        "number_ce":number_ce,
        "number_o":number_o,
        "o_to_ce":number_o/number_ce if number_ce>0 else 0.0,
        "surface_ce":number_surface_ce,
        "surface_o":number_surface_o,
        "geometric_diameter_nm":geometric_diameter_nm
    }

def write_xyz(filename,lattice:Lattice,comment=""):
    filename=Path(filename)
    filename.parent.mkdir(parents=True,exist_ok=True)
    occupied_ids=np.flatnonzero(lattice.occupation!=Species.EMPTY)
    surface=find_external_surface(lattice)
    species_name={
        Species.CE:"Ce",
        Species.O:"O",
        Species.IR_ION:"Ir",
        Species.IR:"Ir"
    }
    with filename.open("w",encoding="utf-8",newline="\n") as f:
        f.write(f"{len(occupied_ids)}\n")
        f.write(
            "Properties="
            "species:S:1:"
            "pos:R:3:"
            "surface:I:1:"
            "ir_state:I:1:"
            "embedded:I:1:"
            "support_contacts:I:1 "
            f"{comment}\n"
        )
        for site_id in occupied_ids:
            species=Species(lattice.occupation[site_id])
            name=species_name[species]
            x,y,z=lattice.positions_nm[site_id]
            is_surface=1 if surface[site_id] else 0
            if species==Species.IR_ION:
                ir_state=1
            elif species==Species.IR:
                ir_state=2
            else:
                ir_state=0
            embedded = int(species in (Species.IR_ION, Species.IR) and not surface[site_id])
            if species in (Species.IR_ION,Species.IR):
                o_neighbors=lattice.get_ce_o_neighbors(int(site_id))
                m_neighbors=lattice.get_m_m_neighbors(int(site_id))
                support_contacts=(
                    int(np.count_nonzero(lattice.occupation[o_neighbors]==Species.O))
                    +int(np.count_nonzero(lattice.occupation[m_neighbors]==Species.CE))
                )
            else:
                support_contacts=0
            f.write(
                f"{name} {x:.6f} {y:.6f} {z:.6f} "
                f"{is_surface} {ir_state} {embedded} {support_contacts}\n"
            )

def main():
    rng=np.random.default_rng(2026)

    lattice=build_fluorite_lattice(ncells=20)

    initialize_sphere(lattice=lattice,diameter_nm=5.0,oxygen_x=2.0,rng=rng)
    initial_summary=geometry_summart(lattice)
    write_xyz("output/initial_sphere.xyz",lattice)

if __name__=="__main__":
    main()

from collections import deque
from pathlib import Path
import numpy as np

from constants import SiteType, Species
from lattice_build import Lattice, build_fluorite_lattice


def build_initial_lattice(
    random_seed: int,
    box_nm: float,
    particle_diameter_nm: float,
    roughness_fraction: float,
) -> Lattice:
    """Build the shared rough spherical CeO2 starting configuration."""
    rng = np.random.default_rng(random_seed)
    lattice = build_fluorite_lattice(ncells=int(np.ceil(box_nm / 0.541)))
    initialize_sphere(
        lattice,
        diameter_nm=particle_diameter_nm,
        oxygen_x=2.0,
        rng=rng,
    )
    roughen_surface(lattice, fraction=roughness_fraction, rng=rng)
    return lattice

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


def is_external_surface(
    lattice: Lattice,
    accessible_empty: np.ndarray,
    site_id: int,
) -> bool:
    """Return whether a site touches the externally accessible solution."""
    neighbors = lattice.get_ce_o_neighbors(site_id)
    return bool(np.any(accessible_empty[neighbors]))

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


def find_main_support_component(lattice: Lattice) -> np.ndarray:
    """Return the largest Ce/O-connected component (the represented particle)."""
    support_mask = (
        (lattice.occupation == Species.CE)
        | (lattice.occupation == Species.O)
    )
    visited = np.zeros(lattice.nsites, dtype=bool)
    largest_component: list[int] = []

    for start in np.flatnonzero(support_mask):
        start = int(start)
        if visited[start]:
            continue
        visited[start] = True
        queue = deque([start])
        component: list[int] = []
        while queue:
            site_id = queue.popleft()
            component.append(site_id)
            for neighbor_id in lattice.get_ce_o_neighbors(site_id):
                neighbor_id = int(neighbor_id)
                if support_mask[neighbor_id] and not visited[neighbor_id]:
                    visited[neighbor_id] = True
                    queue.append(neighbor_id)
        if len(component) > len(largest_component):
            largest_component = component

    main_support = np.zeros(lattice.nsites, dtype=bool)
    if largest_component:
        main_support[np.asarray(largest_component, dtype=np.int32)] = True
    return main_support


def ir_support_contact_count(
    lattice: Lattice,
    site_id: int,
    support_mask: np.ndarray | None = None,
) -> int:
    """Count direct Ce/O support contacts of an Ir lattice site."""
    o_neighbors=lattice.get_ce_o_neighbors(site_id)
    m_neighbors=lattice.get_m_m_neighbors(site_id)
    if support_mask is None:
        return (
            int(np.count_nonzero(lattice.occupation[o_neighbors] == Species.O))
            + int(np.count_nonzero(lattice.occupation[m_neighbors] == Species.CE))
        )
    return (
        int(np.count_nonzero(
            (lattice.occupation[o_neighbors] == Species.O)
            & support_mask[o_neighbors]
        ))
        + int(np.count_nonzero(
            (lattice.occupation[m_neighbors] == Species.CE)
            & support_mask[m_neighbors]
        ))
    )


def find_supported_ir_sites(
    lattice: Lattice,
    support_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return Ir sites belonging to a cluster anchored to the CeO2 support."""
    if support_mask is None:
        support_mask = (
            (lattice.occupation == Species.CE)
            | (lattice.occupation == Species.O)
        )
    ir_mask=(lattice.occupation==Species.IR_ION)|(lattice.occupation==Species.IR)
    supported=np.zeros(lattice.nsites,dtype=bool)
    queue=deque()

    for site_id in np.flatnonzero(ir_mask):
        site_id=int(site_id)
        if ir_support_contact_count(lattice,site_id,support_mask)>0:
            supported[site_id]=True
            queue.append(site_id)

    while queue:
        site_id=queue.popleft()
        for neighbor_id in lattice.get_m_m_neighbors(site_id):
            neighbor_id=int(neighbor_id)
            if ir_mask[neighbor_id] and not supported[neighbor_id]:
                supported[neighbor_id]=True
                queue.append(neighbor_id)

    return supported


def write_xyz(filename,lattice:Lattice,comment="",supported_ir_only:bool=False):
    filename=Path(filename)
    filename.parent.mkdir(parents=True,exist_ok=True)
    occupied=lattice.occupation!=Species.EMPTY
    main_support = None
    if supported_ir_only:
        main_support = find_main_support_component(lattice)
        ir_mask=(lattice.occupation==Species.IR_ION)|(lattice.occupation==Species.IR)
        main_supported_ir = find_supported_ir_sites(lattice, main_support)
        occupied = main_support | (ir_mask & main_supported_ir)
    occupied_ids=np.flatnonzero(occupied)
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
                support_contacts=ir_support_contact_count(
                    lattice,
                    int(site_id),
                    main_support,
                )
            else:
                support_contacts=0
            f.write(
                f"{name} {x:.6f} {y:.6f} {z:.6f} "
                f"{is_surface} {ir_state} {embedded} {support_contacts}\n"
            )

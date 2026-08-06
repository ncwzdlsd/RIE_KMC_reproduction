from dataclasses import dataclass
from itertools import product,permutations

import numpy as np

from constants import SiteType,Species

CE_BASIS=np.array([[0,0,0],[0,2,2],[2,0,2],[2,2,0]],dtype=np.int16)
O_BASIS=np.array(list(product((1,3),repeat=3)),dtype=np.int16)
CE_O_OFFSETS=np.array(list(product((-1,1),repeat=3)),dtype=np.int16)

def build_m_m_offsets():
    offsets=set()
    for base in set(permutations((0,2,2))):
        nonzero_indices=[i for i,v in enumerate(base) if v!=0]
        for signs in product((-1,1),repeat=2):
            offset=list(base)
            for i,sign in zip(nonzero_indices,signs):
                offset[i]=sign*offset[i]
            offsets.add(tuple(offset))
    return np.array(sorted(offsets),dtype=np.int16)

M_M_OFFSETS=build_m_m_offsets()


@dataclass
class Lattice:
    q:np.ndarray
    positions_nm:np.ndarray
    site_type:np.ndarray
    occupation:np.ndarray
    ce_o_neighbors:np.ndarray
    ce_o_neighbor_count:np.ndarray
    m_m_neighbors:np.ndarray
    m_m_neighbor_count:np.ndarray
    reservoir_boundary:np.ndarray
    center_nm:np.ndarray
    lattice_constant_nm:float

    @property
    def nsites(self) -> int:
        return len(self.occupation)

    def get_ce_o_neighbors(self,site_id) -> np.ndarray:
        return self.ce_o_neighbors[site_id,:self.ce_o_neighbor_count[site_id]]

    def get_m_m_neighbors(self,site_id) -> np.ndarray:
        return self.m_m_neighbors[site_id,:self.m_m_neighbor_count[site_id]]

def pack_neighbor_lists(neighbor_lists,max_width):
    number_of_sites=len(neighbor_lists)
    neighbors=np.full((number_of_sites,max_width),-1,dtype=np.int32)

    counts=np.zeros(number_of_sites,dtype=np.uint8)
    for site_id,ids in enumerate(neighbor_lists):
        counts[site_id]=len(ids)
        neighbors[site_id,:len(ids)]=ids

    return neighbors,counts

# ncells is the number of conventional unit cells along each axis
def build_fluorite_lattice(ncells,lattice_constant_nm:float=0.541)->Lattice:
    if ncells <= 0:
        raise ValueError("ncells must be positive")

    # Construct the regular fluorite lattice in vectorized form.  The former
    # dictionary/list implementation was convenient for small demonstrations,
    # but required several million Python objects for the 20 nm paper box.
    cells=np.indices((ncells,ncells,ncells),dtype=np.int16).reshape(3,-1).T
    basis=np.vstack((CE_BASIS,O_BASIS))
    q=(4*cells[:,None,:]+basis[None,:,:]).reshape(-1,3)
    cells=np.repeat(cells,len(basis),axis=0)
    cell_site_types=np.concatenate(
        (
            np.full(len(CE_BASIS),SiteType.M,dtype=np.uint8),
            np.full(len(O_BASIS),SiteType.O,dtype=np.uint8),
        )
    )
    site_type=np.tile(cell_site_types,ncells**3)
    positions_nm=q.astype(np.float64)*lattice_constant_nm/4.0
    center_nm=0.5*(positions_nm.min(axis=0)+positions_nm.max(axis=0))

    grid_width=4*ncells
    coordinate_to_id=np.full(
        (grid_width,grid_width,grid_width),-1,dtype=np.int32
    )
    coordinate_to_id[q[:,0],q[:,1],q[:,2]]=np.arange(len(q),dtype=np.int32)

    def neighbor_table(site_ids,offsets,width):
        table=np.full((len(q),width),-1,dtype=np.int32)
        for column,offset in enumerate(offsets):
            candidates=q[site_ids]+offset
            in_box=np.all((candidates>=0)&(candidates<grid_width),axis=1)
            valid_ids=site_ids[in_box]
            valid_coordinates=candidates[in_box]
            table[valid_ids,column]=coordinate_to_id[
                valid_coordinates[:,0],
                valid_coordinates[:,1],
                valid_coordinates[:,2],
            ]
        # Lattice.get_*_neighbors slices the first ``count`` columns, so keep
        # all real neighbors packed before the -1 padding on boundary rows.
        order=np.argsort(table<0,axis=1,kind="stable")
        return np.take_along_axis(table,order,axis=1)

    all_site_ids=np.arange(len(q),dtype=np.int32)
    ce_o_neighbors=neighbor_table(all_site_ids,CE_O_OFFSETS,8)
    m_site_ids=np.flatnonzero(site_type==SiteType.M).astype(np.int32)
    m_m_neighbors=neighbor_table(m_site_ids,M_M_OFFSETS,12)
    ce_o_count=np.count_nonzero(ce_o_neighbors>=0,axis=1).astype(np.uint8)
    m_m_count=np.count_nonzero(m_m_neighbors>=0,axis=1).astype(np.uint8)

    on_outer_cell=np.any((cells==0)|(cells==ncells-1),axis=1)
    reservoir_boundary=on_outer_cell&(site_type==SiteType.M)

    return Lattice(
        q=q,
        positions_nm=positions_nm,
        site_type=site_type,
        occupation=np.full(len(q),Species.EMPTY,dtype=np.uint8),
        ce_o_neighbors=ce_o_neighbors,
        ce_o_neighbor_count=ce_o_count,
        m_m_neighbors=m_m_neighbors,
        m_m_neighbor_count=m_m_count,
        reservoir_boundary=reservoir_boundary,
        center_nm=center_nm,
        lattice_constant_nm=lattice_constant_nm,
    )

# def validate_lattice(lattice:Lattice) -> dict[str,int]:
#     m = lattice.site_type == SiteType.M
#     o = lattice.site_type == SiteType.O
#     result = {
#         "sites": lattice.nsites,
#         "m_sites": int(np.count_nonzero(m)),
#         "o_sites": int(np.count_nonzero(o)),
#         "max_ce_coordination": int(lattice.ce_o_neighbor_count[m].max()),
#         "max_o_coordination": int(lattice.ce_o_neighbor_count[o].max()),
#         "max_m_neighbors": int(lattice.m_m_neighbor_count[m].max()),
#     }
#     assert result["max_ce_coordination"] == 8
#     assert result["max_o_coordination"] == 4
#     assert result["max_m_neighbors"] == 12
#     return result

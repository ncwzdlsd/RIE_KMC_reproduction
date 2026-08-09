"""Complete local-update KMC engine for Ce/O, Ir, and sonication events."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Hashable

import numpy as np

from ceox_events import CeOxParameters, EventType, transition_rate
from constants import SiteType, Species
from generation import (
    find_accessible_empty_sites,
    find_external_surface,
    find_supported_ir_sites,
    write_xyz,
)
from ir_events import (
    IrEventType,
    IrParameters,
    activated_rate,
    is_supported_reduction_site,
    local_ir_binding_energy,
)
from kmc_engine import calculate_current
from lattice_build import Lattice
from sonication_events import SonicationEventType, SonicationParameters


CE_EVENT_TYPES = (
    EventType.CE_ADSORPTION,
    EventType.CE_DESORPTION,
    EventType.O_ADSORPTION,
    EventType.O_DESORPTION,
)


@dataclass
class LocalKMCState:
    step: int = 0
    kmc_time: float = 0.0
    event_counts: Counter = field(default_factory=Counter)
    sonication_removed_atoms: int = 0
    sonication_removed_ce_atoms: int = 0
    sonication_removed_o_atoms: int = 0
    solution_excess_ce_atoms: int = 0
    solution_excess_o_atoms: int = 0
    solution_ir_precursor_atoms: int | None = None
    stopped_reason: str = ""


class IndexedSet:
    """O(1) dense integer set used for current support-surface sites."""

    def __init__(self, size: int):
        self.members: list[int] = []
        self.position = np.full(size, -1, dtype=np.int32)

    def add(self, member: int) -> None:
        if self.position[member] >= 0:
            return
        self.position[member] = len(self.members)
        self.members.append(member)

    def discard(self, member: int) -> None:
        position = int(self.position[member])
        if position < 0:
            return
        last = self.members.pop()
        if position < len(self.members):
            self.members[position] = last
            self.position[last] = position
        self.position[member] = -1


class DenseRateBuckets:
    """Rate buckets for at most one event in this family per lattice site."""

    def __init__(self, size: int):
        self.keys: list[Hashable] = []
        self.rates: list[float] = []
        self.members: list[list[int]] = []
        self.code_for_key: dict[Hashable, int] = {}
        self.member_code = np.full(size, -1, dtype=np.int16)
        self.member_position = np.full(size, -1, dtype=np.int32)
        self.total_rate = 0.0

    def _code(self, key: Hashable, rate: float) -> int:
        code = self.code_for_key.get(key)
        if code is not None:
            if not math.isclose(self.rates[code], rate, rel_tol=1e-12, abs_tol=0.0):
                raise RuntimeError(f"rate changed for bucket {key}")
            return code
        code = len(self.keys)
        self.code_for_key[key] = code
        self.keys.append(key)
        self.rates.append(rate)
        self.members.append([])
        return code

    def add(self, member: int, key: Hashable, rate: float) -> None:
        if rate <= 0.0:
            return
        if self.member_code[member] >= 0:
            raise RuntimeError(f"member {member} already has an event in this family")
        code = self._code(key, rate)
        self.member_code[member] = code
        self.member_position[member] = len(self.members[code])
        self.members[code].append(member)
        self.total_rate += rate

    def discard(self, member: int) -> None:
        code = int(self.member_code[member])
        if code < 0:
            return
        position = int(self.member_position[member])
        bucket = self.members[code]
        last = bucket.pop()
        if position < len(bucket):
            bucket[position] = last
            self.member_position[last] = position
        self.member_code[member] = -1
        self.member_position[member] = -1
        self.total_rate -= self.rates[code]

    def select(self, target: float, rng: np.random.Generator) -> tuple[Hashable, int]:
        cumulative = 0.0
        for code, (rate, members) in enumerate(zip(self.rates, self.members)):
            cumulative += rate * len(members)
            if target < cumulative and members:
                member = members[int(rng.integers(len(members)))]
                return self.keys[code], member
        raise RuntimeError("failed to select dense-bucket event")


class SparseRateBuckets:
    """Rate buckets for directed Ir diffusion edges."""

    def __init__(self):
        self.keys: list[Hashable] = []
        self.rates: list[float] = []
        self.members: list[list[tuple[int, int]]] = []
        self.code_for_key: dict[Hashable, int] = {}
        self.location: dict[tuple[int, int], tuple[int, int]] = {}
        self.total_rate = 0.0

    def _code(self, key: Hashable, rate: float) -> int:
        code = self.code_for_key.get(key)
        if code is not None:
            if not math.isclose(self.rates[code], rate, rel_tol=1e-12, abs_tol=0.0):
                raise RuntimeError(f"rate changed for diffusion bucket {key}")
            return code
        code = len(self.keys)
        self.code_for_key[key] = code
        self.keys.append(key)
        self.rates.append(rate)
        self.members.append([])
        return code

    def add(self, member: tuple[int, int], key: Hashable, rate: float) -> None:
        if rate <= 0.0:
            return
        if member in self.location:
            raise RuntimeError(f"diffusion edge {member} already exists")
        code = self._code(key, rate)
        position = len(self.members[code])
        self.members[code].append(member)
        self.location[member] = (code, position)
        self.total_rate += rate

    def discard(self, member: tuple[int, int]) -> None:
        location = self.location.pop(member, None)
        if location is None:
            return
        code, position = location
        bucket = self.members[code]
        last = bucket.pop()
        if position < len(bucket):
            bucket[position] = last
            self.location[last] = (code, position)
        self.total_rate -= self.rates[code]

    def select(
        self, target: float, rng: np.random.Generator
    ) -> tuple[Hashable, tuple[int, int]]:
        cumulative = 0.0
        for code, (rate, members) in enumerate(zip(self.rates, self.members)):
            cumulative += rate * len(members)
            if target < cumulative and members:
                member = members[int(rng.integers(len(members)))]
                return self.keys[code], member
        raise RuntimeError("failed to select diffusion event")


class LocalKMC:
    """Incremental n-fold-way engine covering every reported event family."""

    def __init__(
        self,
        lattice: Lattice,
        ceox_parameters: CeOxParameters,
        ir_parameters: IrParameters,
        sonication_parameters: SonicationParameters | None = None,
        random_seed: int = 2026,
        state: LocalKMCState | None = None,
        accessible_empty: np.ndarray | None = None,
        rng_state: dict | None = None,
        initial_ce_atoms: int | None = None,
        initial_ir_precursor_atoms: int | None = None,
    ):
        self.lattice = lattice
        self.ceox_parameters = ceox_parameters
        self.ir_parameters = ir_parameters
        self.sonication_parameters = sonication_parameters
        self.state = state or LocalKMCState()
        self.rng = np.random.default_rng(random_seed)
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state
        self.accessible_empty = (
            find_accessible_empty_sites(lattice)
            if accessible_empty is None
            else np.asarray(accessible_empty, dtype=bool)
        )
        self.initial_ce_atoms = (
            int(np.count_nonzero(lattice.occupation == Species.CE))
            if initial_ce_atoms is None
            else int(initial_ce_atoms)
        )
        solid_ir_atoms = int(
            np.count_nonzero(
                (lattice.occupation == Species.IR_ION)
                | (lattice.occupation == Species.IR)
            )
        )
        if initial_ir_precursor_atoms is None:
            scaled_inventory = round(
                self.initial_ce_atoms
                * self.ir_parameters.precursor_ir_to_ce_atom_ratio
            )
            initial_ir_precursor_atoms = max(
                solid_ir_atoms,
                1 if self.initial_ce_atoms and self.ir_parameters.precursor_ir_to_ce_atom_ratio > 0.0 else 0,
                scaled_inventory,
            )
        self.initial_ir_precursor_atoms = int(initial_ir_precursor_atoms)
        if self.initial_ir_precursor_atoms < solid_ir_atoms:
            raise ValueError(
                "initial Ir precursor inventory cannot be smaller than solid Ir count"
            )
        if self.state.solution_ir_precursor_atoms is None:
            self.state.solution_ir_precursor_atoms = (
                self.initial_ir_precursor_atoms - solid_ir_atoms
            )
        if not 0 <= self.state.solution_ir_precursor_atoms <= self.initial_ir_precursor_atoms:
            raise ValueError("solution Ir precursor count is outside inventory bounds")

        self.ce_buckets = DenseRateBuckets(lattice.nsites)
        self.ir_exchange_buckets = DenseRateBuckets(lattice.nsites)
        self.ir_redox_buckets = DenseRateBuckets(lattice.nsites)
        self.diffusion_buckets = SparseRateBuckets()
        self.surface_support = IndexedSet(lattice.nsites)
        self._build_all_events()

    def _ce_o_coordination(self, site_id: int) -> tuple[int, int]:
        neighbors = self.lattice.get_ce_o_neighbors(site_id)
        occupations = self.lattice.occupation[neighbors]
        if self.lattice.site_type[site_id] == SiteType.M:
            return int(np.count_nonzero(occupations == Species.O)), 0
        ce_count = int(np.count_nonzero(occupations == Species.CE))
        ir_count = int(
            np.count_nonzero(
                (occupations == Species.IR_ION)
                | (occupations == Species.IR)
            )
        )
        return ce_count, ir_count

    def _ce_o_binding(self, coordination: int, ir_coordination: int) -> float:
        return (
            coordination * self.ceox_parameters.ce_o_binding_energy_ev
            + ir_coordination * self.ceox_parameters.ir_o_binding_energy_ev
        )

    def _is_external_surface(self, site_id: int) -> bool:
        neighbors = self.lattice.get_ce_o_neighbors(site_id)
        return bool(np.any(self.accessible_empty[neighbors]))

    def _reservoir_saturation_count(self, species: Species) -> float:
        ce_capacity = max(
            1.0,
            self.initial_ce_atoms
            * self.ceox_parameters.excess_reservoir_saturation_fraction,
        )
        return ce_capacity if species == Species.CE else 2.0 * ce_capacity

    def _solution_chemical_potential(self, species: Species) -> float:
        if species == Species.CE:
            baseline = self.ceox_parameters.chemical_potential_ce_ev
            maximum = self.ceox_parameters.maximum_chemical_potential_ce_ev
            excess_atoms = max(
                self.state.solution_excess_ce_atoms,
                self.state.sonication_removed_ce_atoms,
            )
        else:
            baseline = self.ceox_parameters.chemical_potential_o_ev
            maximum = self.ceox_parameters.maximum_chemical_potential_o_ev
            excess_atoms = max(
                self.state.solution_excess_o_atoms,
                self.state.sonication_removed_o_atoms,
            )
        fraction = min(
            1.0,
            excess_atoms / self._reservoir_saturation_count(species),
        )
        return baseline + max(0.0, maximum - baseline) * fraction

    def _ce_rate_for_key(self, key: Hashable) -> float:
        kind, coordination, ir_coordination = key
        if kind in (EventType.CE_ADSORPTION, EventType.CE_DESORPTION):
            species = Species.CE
        else:
            species = Species.O
        chemical_potential = self._solution_chemical_potential(species)
        binding = self._ce_o_binding(coordination, ir_coordination)
        if kind in (EventType.CE_ADSORPTION, EventType.O_ADSORPTION):
            delta = -chemical_potential - binding
            prefactor = self.ceox_parameters.adsorption_prefactor
        else:
            delta = binding + chemical_potential
            prefactor = self.ceox_parameters.desorption_prefactor
        return transition_rate(delta, prefactor, self.ceox_parameters)

    def _refresh_ce_bucket_rates(self) -> None:
        total_rate = 0.0
        for code, key in enumerate(self.ce_buckets.keys):
            rate = self._ce_rate_for_key(key)
            self.ce_buckets.rates[code] = rate
            total_rate += rate * len(self.ce_buckets.members[code])
        self.ce_buckets.total_rate = total_rate

    def _ce_event(self, site_id: int) -> tuple[Hashable, float] | None:
        site_type = self.lattice.site_type[site_id]
        occupation = self.lattice.occupation[site_id]
        coordination, ir_coordination = self._ce_o_coordination(site_id)
        if occupation == Species.EMPTY:
            if not self.accessible_empty[site_id]:
                return None
            if coordination + ir_coordination == 0:
                return None
            kind = EventType.CE_ADSORPTION if site_type == SiteType.M else EventType.O_ADSORPTION
            species = Species.CE if site_type == SiteType.M else Species.O
            chemical_potential = self._solution_chemical_potential(species)
            binding = self._ce_o_binding(coordination, ir_coordination)
            delta = -chemical_potential - binding
            rate = transition_rate(delta, self.ceox_parameters.adsorption_prefactor, self.ceox_parameters)
            return (kind, coordination, ir_coordination), rate
        if not self._is_external_surface(site_id):
            return None
        if occupation == Species.CE:
            kind = EventType.CE_DESORPTION
            chemical_potential = self._solution_chemical_potential(Species.CE)
        elif occupation == Species.O:
            kind = EventType.O_DESORPTION
            chemical_potential = self._solution_chemical_potential(Species.O)
        else:
            return None
        binding = self._ce_o_binding(coordination, ir_coordination)
        delta = binding + chemical_potential
        rate = transition_rate(delta, self.ceox_parameters.desorption_prefactor, self.ceox_parameters)
        return (kind, coordination, ir_coordination), rate

    def _ir_exchange_event(self, site_id: int) -> tuple[Hashable, float] | None:
        occupation = self.lattice.occupation[site_id]
        if occupation not in (Species.EMPTY, Species.IR_ION):
            return None
        if not self.lattice.reservoir_boundary[site_id]:
            return None
        binding, ir_count, o_count = local_ir_binding_energy(
            self.lattice, site_id, self.ir_parameters
        )
        if occupation == Species.EMPTY:
            if not self.accessible_empty[site_id]:
                return None
            kind = IrEventType.IR_ION_ADSORPTION
            delta = -self.ir_parameters.chemical_potential_ir_ion_ev - binding
            prefactor = self.ir_parameters.adsorption_prefactor
            barrier = self.ir_parameters.adsorption_barrier_ev
        else:
            kind = IrEventType.IR_ION_DESORPTION
            delta = self.ir_parameters.chemical_potential_ir_ion_ev + binding
            prefactor = self.ir_parameters.desorption_prefactor
            barrier = self.ir_parameters.desorption_barrier_ev
        rate = activated_rate(delta, prefactor, barrier, self.ir_parameters)
        return (kind, ir_count, o_count), rate

    def _ir_precursor_fraction(self) -> float:
        if self.initial_ir_precursor_atoms <= 0:
            return 0.0
        return (
            self.state.solution_ir_precursor_atoms
            / self.initial_ir_precursor_atoms
        )

    def _ir_exchange_rate_scale(self, key: Hashable) -> float:
        kind = key[0]
        if kind == IrEventType.IR_ION_ADSORPTION:
            # Ideal well-mixed finite bath: collision/adsorption propensity is
            # proportional to the remaining precursor concentration.
            return self._ir_precursor_fraction()
        return 1.0

    def _ir_exchange_total_rate(self) -> float:
        return sum(
            rate * len(members) * self._ir_exchange_rate_scale(key)
            for key, rate, members in zip(
                self.ir_exchange_buckets.keys,
                self.ir_exchange_buckets.rates,
                self.ir_exchange_buckets.members,
            )
        )

    def _select_ir_exchange_event(
        self, target: float
    ) -> tuple[Hashable, int]:
        cumulative = 0.0
        for key, rate, members in zip(
            self.ir_exchange_buckets.keys,
            self.ir_exchange_buckets.rates,
            self.ir_exchange_buckets.members,
        ):
            cumulative += (
                rate * len(members) * self._ir_exchange_rate_scale(key)
            )
            if target < cumulative and members:
                member = members[int(self.rng.integers(len(members)))]
                return key, member
        raise RuntimeError("failed to select finite-reservoir Ir exchange event")

    def _ir_redox_event(self, site_id: int) -> tuple[Hashable, float] | None:
        occupation = self.lattice.occupation[site_id]
        if occupation == Species.IR_ION:
            if not is_supported_reduction_site(self.lattice, site_id):
                return None
            kind = IrEventType.IR_REDUCTION
            delta = self.ir_parameters.reduction_free_energy_ev
            prefactor = self.ir_parameters.reduction_prefactor
            barrier = self.ir_parameters.reduction_barrier_ev
        elif occupation == Species.IR:
            kind = IrEventType.IR_OXIDATION
            delta = -self.ir_parameters.reduction_free_energy_ev
            prefactor = self.ir_parameters.oxidation_prefactor
            barrier = self.ir_parameters.oxidation_barrier_ev
        else:
            return None
        rate = activated_rate(delta, prefactor, barrier, self.ir_parameters)
        return kind, rate

    def _diffusion_event(
        self, source: int, target: int
    ) -> tuple[Hashable, float] | None:
        if self.lattice.occupation[source] != Species.IR_ION:
            return None
        if self.lattice.occupation[target] != Species.EMPTY:
            return None
        if not self.accessible_empty[target]:
            return None
        initial_binding, initial_ir, initial_o = local_ir_binding_energy(
            self.lattice, source, self.ir_parameters
        )
        final_binding, final_ir, final_o = local_ir_binding_energy(
            self.lattice, target, self.ir_parameters, ignored_ir_site_id=source
        )
        delta = initial_binding - final_binding
        rate = activated_rate(
            delta,
            self.ir_parameters.diffusion_prefactor,
            self.ir_parameters.diffusion_barrier_ev,
            self.ir_parameters,
        )
        key = (initial_ir, initial_o, final_ir, final_o)
        return key, rate

    def _refresh_ce(self, site_id: int) -> None:
        self.ce_buckets.discard(site_id)
        event = self._ce_event(site_id)
        if event is not None:
            self.ce_buckets.add(site_id, *event)

    def _refresh_ir_site(self, site_id: int) -> None:
        self.ir_exchange_buckets.discard(site_id)
        self.ir_redox_buckets.discard(site_id)
        exchange = self._ir_exchange_event(site_id)
        if exchange is not None:
            self.ir_exchange_buckets.add(site_id, *exchange)
        redox = self._ir_redox_event(site_id)
        if redox is not None:
            self.ir_redox_buckets.add(site_id, *redox)

    def _discard_diffusion_source(self, source: int) -> None:
        for target in self.lattice.get_m_m_neighbors(source):
            self.diffusion_buckets.discard((source, int(target)))

    def _refresh_diffusion_source(self, source: int) -> None:
        self._discard_diffusion_source(source)
        if self.lattice.occupation[source] != Species.IR_ION:
            return
        for target in self.lattice.get_m_m_neighbors(source):
            target = int(target)
            event = self._diffusion_event(source, target)
            if event is not None:
                self.diffusion_buckets.add((source, target), *event)

    def _refresh_surface(self, site_id: int) -> None:
        self.surface_support.discard(site_id)
        if self.lattice.occupation[site_id] in (Species.CE, Species.O):
            if self._is_external_surface(site_id):
                self.surface_support.add(site_id)

    def _build_all_events(self) -> None:
        for site_id in range(self.lattice.nsites):
            self._refresh_ce(site_id)
            self._refresh_surface(site_id)
            if self.lattice.site_type[site_id] == SiteType.M:
                self._refresh_ir_site(site_id)
        for source in np.flatnonzero(self.lattice.occupation == Species.IR_ION):
            self._refresh_diffusion_source(int(source))

    def _affected_sites(
        self, changed_sites: set[int]
    ) -> tuple[set[int], set[int], set[int]]:
        ce_sites = set(changed_sites)
        base_m_sites: set[int] = set()
        for site_id in changed_sites:
            ce_neighbors = set(map(int, self.lattice.get_ce_o_neighbors(site_id)))
            ce_sites.update(ce_neighbors)
            if self.lattice.site_type[site_id] == SiteType.M:
                base_m_sites.add(site_id)
                base_m_sites.update(map(int, self.lattice.get_m_m_neighbors(site_id)))
            else:
                base_m_sites.update(ce_neighbors)
        m_sites = set(base_m_sites)
        for site_id in tuple(base_m_sites):
            m_sites.update(map(int, self.lattice.get_m_m_neighbors(site_id)))
        surface_sites = set(ce_sites)
        return ce_sites, m_sites, surface_sites

    def _before_change(
        self, ce_sites: set[int], m_sites: set[int], surface_sites: set[int]
    ) -> None:
        for site_id in ce_sites:
            self.ce_buckets.discard(site_id)
        for site_id in m_sites:
            self.ir_exchange_buckets.discard(site_id)
            self.ir_redox_buckets.discard(site_id)
            self._discard_diffusion_source(site_id)
        for site_id in surface_sites:
            self.surface_support.discard(site_id)

    def _after_change(
        self, ce_sites: set[int], m_sites: set[int], surface_sites: set[int]
    ) -> None:
        for site_id in ce_sites:
            self._refresh_ce(site_id)
        for site_id in m_sites:
            self._refresh_ir_site(site_id)
            self._refresh_diffusion_source(site_id)
        for site_id in surface_sites:
            self._refresh_surface(site_id)

    def _apply_site_changes(
        self,
        changes: dict[int, Species],
        solution_ce_delta: int = 0,
        solution_o_delta: int = 0,
        solution_ir_delta: int = 0,
    ) -> None:
        changed_sites = set(changes)
        ce_sites, m_sites, surface_sites = self._affected_sites(changed_sites)
        self._before_change(ce_sites, m_sites, surface_sites)
        for site_id, species in changes.items():
            self.lattice.occupation[site_id] = species
            if species == Species.EMPTY:
                neighbor_accessible = any(
                    self.accessible_empty[int(neighbor)]
                    for neighbor in self.lattice.get_ce_o_neighbors(site_id)
                )
                self.accessible_empty[site_id] = bool(
                    self.lattice.reservoir_boundary[site_id] or neighbor_accessible
                )
            else:
                self.accessible_empty[site_id] = False
        old_solution_counts = (
            self.state.solution_excess_ce_atoms,
            self.state.solution_excess_o_atoms,
        )
        self.state.solution_excess_ce_atoms = max(
            0,
            self.state.solution_excess_ce_atoms + solution_ce_delta,
        )
        self.state.solution_excess_o_atoms = max(
            0,
            self.state.solution_excess_o_atoms + solution_o_delta,
        )
        updated_ir_precursor = (
            self.state.solution_ir_precursor_atoms + solution_ir_delta
        )
        if not 0 <= updated_ir_precursor <= self.initial_ir_precursor_atoms:
            raise RuntimeError("finite Ir precursor inventory would be violated")
        self.state.solution_ir_precursor_atoms = updated_ir_precursor
        if old_solution_counts != (
            self.state.solution_excess_ce_atoms,
            self.state.solution_excess_o_atoms,
        ):
            self._refresh_ce_bucket_rates()
        self._after_change(ce_sites, m_sites, surface_sites)

    def _sonication_rate(self) -> float:
        if self.sonication_parameters is None or not self.surface_support.members:
            return 0.0
        # One KMC corrosion event exists for every current support-solution
        # interface center.  The family is stored as an n-fold-way aggregate;
        # _apply_sonication_event chooses the selected center uniformly.
        return (
            self.sonication_parameters.event_rate
            * len(self.surface_support.members)
        )

    def total_rate(self) -> float:
        return (
            self.ce_buckets.total_rate
            + self._ir_exchange_total_rate()
            + self.ir_redox_buckets.total_rate
            + self.diffusion_buckets.total_rate
            + self._sonication_rate()
        )

    def _draw_event(self) -> tuple[str, Hashable | None, object | None, float]:
        ir_exchange_rate = self._ir_exchange_total_rate()
        family_rates = (
            ("ce", self.ce_buckets.total_rate, self.ce_buckets),
            ("ir_exchange", ir_exchange_rate, self.ir_exchange_buckets),
            ("ir_redox", self.ir_redox_buckets.total_rate, self.ir_redox_buckets),
            ("diffusion", self.diffusion_buckets.total_rate, self.diffusion_buckets),
            ("sonication", self._sonication_rate(), None),
        )
        total_rate = sum(rate for _, rate, _ in family_rates)
        if not math.isfinite(total_rate) or total_rate <= 0.0:
            raise RuntimeError("no available KMC events")
        target = self.rng.random() * total_rate
        cumulative = 0.0
        for name, rate, buckets in family_rates:
            if target < cumulative + rate:
                if buckets is None:
                    return name, None, None, total_rate
                if name == "ir_exchange":
                    key, member = self._select_ir_exchange_event(
                        target - cumulative
                    )
                else:
                    key, member = buckets.select(target - cumulative, self.rng)
                return name, key, member, total_rate
            cumulative += rate
        raise RuntimeError("failed to select KMC event family")

    def _apply_drawn_event(self, family: str, key: Hashable | None, member: object) -> str:
        if family == "ce":
            kind = key[0]
            site_id = int(member)
            species = Species.EMPTY if kind in (EventType.CE_DESORPTION, EventType.O_DESORPTION) else (
                Species.CE if kind == EventType.CE_ADSORPTION else Species.O
            )
            ce_delta = (
                1 if kind == EventType.CE_DESORPTION
                else -1 if kind == EventType.CE_ADSORPTION
                else 0
            )
            o_delta = (
                1 if kind == EventType.O_DESORPTION
                else -1 if kind == EventType.O_ADSORPTION
                else 0
            )
            self._apply_site_changes(
                {site_id: species},
                solution_ce_delta=ce_delta,
                solution_o_delta=o_delta,
            )
            return kind.value
        if family == "ir_exchange":
            kind = key[0]
            site_id = int(member)
            species = Species.IR_ION if kind == IrEventType.IR_ION_ADSORPTION else Species.EMPTY
            solution_ir_delta = (
                -1 if kind == IrEventType.IR_ION_ADSORPTION else 1
            )
            self._apply_site_changes(
                {site_id: species}, solution_ir_delta=solution_ir_delta
            )
            return kind.value
        if family == "ir_redox":
            kind = key
            site_id = int(member)
            species = Species.IR if kind == IrEventType.IR_REDUCTION else Species.IR_ION
            self._apply_site_changes({site_id: species})
            return kind.value
        if family == "diffusion":
            source, target = member
            self._apply_site_changes({int(source): Species.EMPTY, int(target): Species.IR_ION})
            return IrEventType.IR_ION_DIFFUSION.value
        if family == "sonication":
            return self._apply_sonication_event()
        raise RuntimeError(f"unsupported event family {family}")

    def _apply_sonication_event(self) -> str:
        parameters = self.sonication_parameters
        if parameters is None:
            raise RuntimeError("sonication event selected without parameters")
        candidates = np.asarray(self.surface_support.members, dtype=np.int32)
        center_id = int(candidates[self.rng.integers(len(candidates))])
        offsets = self.lattice.positions_nm[candidates] - self.lattice.positions_nm[center_id]
        local_ids = candidates[
            np.einsum("ij,ij->i", offsets, offsets) <= parameters.radius_nm**2
        ]
        removed = local_ids[
            self.rng.random(len(local_ids)) < parameters.dissolution_probability
        ]
        if len(removed):
            occupations = self.lattice.occupation[removed].copy()
            removed_ce = int(
                np.count_nonzero(occupations == Species.CE)
            )
            removed_o = int(
                np.count_nonzero(occupations == Species.O)
            )
            self.state.sonication_removed_ce_atoms += removed_ce
            self.state.sonication_removed_o_atoms += removed_o
            self.state.sonication_removed_atoms += len(removed)
            self._apply_site_changes(
                {int(site_id): Species.EMPTY for site_id in removed},
                solution_ce_delta=removed_ce,
                solution_o_delta=removed_o,
            )
        return SonicationEventType.CORROSION.value

    def step_once(self, stop_time: float | None = None) -> bool:
        family, key, member, total_rate = self._draw_event()
        random_number = max(self.rng.random(), np.finfo(np.float64).tiny)
        delta_time = -math.log(random_number) / total_rate
        if stop_time is not None and self.state.kmc_time + delta_time > stop_time:
            self.state.kmc_time = stop_time
            return False
        event_name = self._apply_drawn_event(family, key, member)
        self.state.kmc_time += delta_time
        self.state.step += 1
        self.state.event_counts[event_name] += 1
        return True

    def advance_to_time(
        self,
        target_time: float,
        reconcile_every: int = 0,
        maximum_events: int | None = None,
    ) -> None:
        if target_time < self.state.kmc_time:
            raise ValueError("target_time must not go backwards")
        initial_step = self.state.step
        while self.state.kmc_time < target_time:
            if maximum_events is not None and self.state.step - initial_step >= maximum_events:
                raise RuntimeError("maximum event count reached before target time")
            applied = self.step_once(stop_time=target_time)
            if not applied:
                break
            if reconcile_every and self.state.step % reconcile_every == 0:
                self.reconcile_accessibility()

    def reconcile_accessibility(self) -> int:
        exact = find_accessible_empty_sites(self.lattice)
        difference = int(np.count_nonzero(exact != self.accessible_empty))
        if difference:
            self.accessible_empty = exact
            self.ce_buckets = DenseRateBuckets(self.lattice.nsites)
            self.ir_exchange_buckets = DenseRateBuckets(self.lattice.nsites)
            self.ir_redox_buckets = DenseRateBuckets(self.lattice.nsites)
            self.diffusion_buckets = SparseRateBuckets()
            self.surface_support = IndexedSet(self.lattice.nsites)
            self._build_all_events()
        return difference

    def metrics(self) -> dict:
        row = calculate_current(self.lattice, self.state)
        net_released_ce = (
            self.state.event_counts[EventType.CE_DESORPTION.value]
            + self.state.sonication_removed_ce_atoms
            - self.state.event_counts[EventType.CE_ADSORPTION.value]
        )
        net_released_o = (
            self.state.event_counts[EventType.O_DESORPTION.value]
            + self.state.sonication_removed_o_atoms
            - self.state.event_counts[EventType.O_ADSORPTION.value]
        )
        net_adsorbed_ir = (
            self.state.event_counts[IrEventType.IR_ION_ADSORPTION.value]
            - self.state.event_counts[IrEventType.IR_ION_DESORPTION.value]
        )
        row["net_released_Ce_atoms"] = int(net_released_ce)
        row["net_released_Ce_fraction"] = (
            max(0.0, net_released_ce / self.initial_ce_atoms)
            if self.initial_ce_atoms
            else 0.0
        )
        row["net_released_O_atoms"] = int(net_released_o)
        row["net_adsorbed_Ir_atoms"] = int(net_adsorbed_ir)
        row["initial_Ir_precursor_atoms"] = self.initial_ir_precursor_atoms
        row["solution_Ir_precursor_atoms"] = (
            self.state.solution_ir_precursor_atoms
        )
        row["Ir_precursor_fraction_remaining"] = self._ir_precursor_fraction()
        solid_ir_atoms = row["number_Ir_ion"] + row["number_Ir"]
        row["Ir_inventory_error_atoms"] = int(
            self.initial_ir_precursor_atoms
            - self.state.solution_ir_precursor_atoms
            - solid_ir_atoms
        )
        row["solution_excess_Ce_atoms"] = self.state.solution_excess_ce_atoms
        row["solution_excess_O_atoms"] = self.state.solution_excess_o_atoms
        row["effective_chemical_potential_Ce_ev"] = (
            self._solution_chemical_potential(Species.CE)
        )
        row["effective_chemical_potential_O_ev"] = (
            self._solution_chemical_potential(Species.O)
        )
        row["sonication_removed_Ce_atoms"] = self.state.sonication_removed_ce_atoms
        row["sonication_removed_O_atoms"] = self.state.sonication_removed_o_atoms
        row["interface_support_site_count"] = len(self.surface_support.members)
        row["sonication_total_propensity_per_s"] = self._sonication_rate()
        return row

    def write_snapshot(
        self,
        filename: Path,
        sample_time: float | None = None,
        supported_ir_only: bool = False,
    ) -> None:
        sample_time = self.state.kmc_time if sample_time is None else sample_time
        visibility_comment = ""
        if supported_ir_only:
            ir_mask = (
                (self.lattice.occupation == Species.IR_ION)
                | (self.lattice.occupation == Species.IR)
            )
            supported_ir = find_supported_ir_sites(self.lattice)
            total_ir = int(np.count_nonzero(ir_mask))
            visible_ir = int(np.count_nonzero(ir_mask & supported_ir))
            visibility_comment = (
                "ir_output=support_connected_only "
                f"visible_Ir={visible_ir} hidden_unattached_Ir={total_ir-visible_ir} "
            )
        write_xyz(
            filename,
            self.lattice,
            supported_ir_only=supported_ir_only,
            comment=(
                f"step={self.state.step} KMC_time={self.state.kmc_time:.10e} "
                f"sample_time={sample_time:.10e} "
                f"mu_Ce={self._solution_chemical_potential(Species.CE):.6f} "
                f"mu_O={self._solution_chemical_potential(Species.O):.6f} "
                f"Ir_precursor={self.state.solution_ir_precursor_atoms}/"
                f"{self.initial_ir_precursor_atoms} "
                f"{visibility_comment}"
                f"k_sonication_total={self._sonication_rate():.6e}"
            ),
        )

    def save_checkpoint(self, filename: Path) -> None:
        filename.parent.mkdir(parents=True, exist_ok=True)
        state_data = {
            "step": self.state.step,
            "kmc_time": self.state.kmc_time,
            "event_counts": dict(self.state.event_counts),
            "sonication_removed_atoms": self.state.sonication_removed_atoms,
            "sonication_removed_ce_atoms": self.state.sonication_removed_ce_atoms,
            "sonication_removed_o_atoms": self.state.sonication_removed_o_atoms,
            "solution_excess_ce_atoms": self.state.solution_excess_ce_atoms,
            "solution_excess_o_atoms": self.state.solution_excess_o_atoms,
            "solution_ir_precursor_atoms": self.state.solution_ir_precursor_atoms,
            "stopped_reason": self.state.stopped_reason,
        }
        bucket_data = {
            "model_version": 2,
            "ce": {
                "keys": [
                    [key[0].value, key[1], key[2]]
                    for key in self.ce_buckets.keys
                ],
                "rates": self.ce_buckets.rates,
                "members": self.ce_buckets.members,
                "total_rate": self.ce_buckets.total_rate,
            },
            "ir_exchange": {
                "keys": [
                    [key[0].value, key[1], key[2]]
                    for key in self.ir_exchange_buckets.keys
                ],
                "rates": self.ir_exchange_buckets.rates,
                "members": self.ir_exchange_buckets.members,
                "total_rate": self.ir_exchange_buckets.total_rate,
            },
            "ir_redox": {
                "keys": [key.value for key in self.ir_redox_buckets.keys],
                "rates": self.ir_redox_buckets.rates,
                "members": self.ir_redox_buckets.members,
                "total_rate": self.ir_redox_buckets.total_rate,
            },
            "diffusion": {
                "keys": [list(key) for key in self.diffusion_buckets.keys],
                "rates": self.diffusion_buckets.rates,
                "members": self.diffusion_buckets.members,
                "total_rate": self.diffusion_buckets.total_rate,
            },
            "surface_support": self.surface_support.members,
        }
        np.savez_compressed(
            filename,
            occupation=self.lattice.occupation,
            accessible_empty=self.accessible_empty,
            state=np.asarray(json.dumps(state_data)),
            rng_state=np.asarray(json.dumps(self.rng.bit_generator.state)),
            initial_ce_atoms=np.asarray(self.initial_ce_atoms, dtype=np.int64),
            initial_ir_precursor_atoms=np.asarray(
                self.initial_ir_precursor_atoms, dtype=np.int64
            ),
            bucket_state=np.asarray(json.dumps(bucket_data)),
        )

    def restore_bucket_state(self, data: dict) -> None:
        def restore_dense(bucket, saved, decoder):
            bucket.keys = [decoder(key) for key in saved["keys"]]
            bucket.rates = [float(rate) for rate in saved["rates"]]
            bucket.members = [
                [int(member) for member in members]
                for members in saved["members"]
            ]
            bucket.code_for_key = {
                key: code for code, key in enumerate(bucket.keys)
            }
            bucket.member_code.fill(-1)
            bucket.member_position.fill(-1)
            for code, members in enumerate(bucket.members):
                for position, member in enumerate(members):
                    bucket.member_code[member] = code
                    bucket.member_position[member] = position
            bucket.total_rate = float(saved["total_rate"])

        restore_dense(
            self.ce_buckets,
            data["ce"],
            lambda key: (EventType(key[0]), int(key[1]), int(key[2])),
        )
        restore_dense(
            self.ir_exchange_buckets,
            data["ir_exchange"],
            lambda key: (IrEventType(key[0]), int(key[1]), int(key[2])),
        )
        restore_dense(
            self.ir_redox_buckets,
            data["ir_redox"],
            lambda key: IrEventType(key),
        )

        saved_diffusion = data["diffusion"]
        self.diffusion_buckets.keys = [
            tuple(map(int, key)) for key in saved_diffusion["keys"]
        ]
        self.diffusion_buckets.rates = [
            float(rate) for rate in saved_diffusion["rates"]
        ]
        self.diffusion_buckets.members = [
            [tuple(map(int, member)) for member in members]
            for members in saved_diffusion["members"]
        ]
        self.diffusion_buckets.code_for_key = {
            key: code for code, key in enumerate(self.diffusion_buckets.keys)
        }
        self.diffusion_buckets.location = {}
        for code, members in enumerate(self.diffusion_buckets.members):
            for position, member in enumerate(members):
                self.diffusion_buckets.location[member] = (code, position)
        self.diffusion_buckets.total_rate = float(saved_diffusion["total_rate"])

        self.surface_support.members = [
            int(member) for member in data["surface_support"]
        ]
        self.surface_support.position.fill(-1)
        for position, member in enumerate(self.surface_support.members):
            self.surface_support.position[member] = position


def load_local_checkpoint(
    filename: Path,
    lattice: Lattice,
    ceox_parameters: CeOxParameters,
    ir_parameters: IrParameters,
    sonication_parameters: SonicationParameters | None = None,
) -> LocalKMC:
    with np.load(filename, allow_pickle=False) as checkpoint:
        lattice.occupation[:] = checkpoint["occupation"]
        state_data = json.loads(str(checkpoint["state"]))
        state = LocalKMCState(
            step=int(state_data["step"]),
            kmc_time=float(state_data["kmc_time"]),
            event_counts=Counter(state_data["event_counts"]),
            sonication_removed_atoms=int(state_data["sonication_removed_atoms"]),
            sonication_removed_ce_atoms=int(state_data["sonication_removed_ce_atoms"]),
            sonication_removed_o_atoms=int(state_data["sonication_removed_o_atoms"]),
            solution_excess_ce_atoms=int(
                state_data.get("solution_excess_ce_atoms", 0)
            ),
            solution_excess_o_atoms=int(
                state_data.get("solution_excess_o_atoms", 0)
            ),
            solution_ir_precursor_atoms=(
                int(state_data["solution_ir_precursor_atoms"])
                if "solution_ir_precursor_atoms" in state_data
                else None
            ),
            stopped_reason=str(state_data["stopped_reason"]),
        )
        engine = LocalKMC(
            lattice,
            ceox_parameters,
            ir_parameters,
            sonication_parameters=sonication_parameters,
            state=state,
            accessible_empty=checkpoint["accessible_empty"],
            rng_state=json.loads(str(checkpoint["rng_state"])),
            initial_ce_atoms=int(checkpoint["initial_ce_atoms"]),
            initial_ir_precursor_atoms=(
                int(checkpoint["initial_ir_precursor_atoms"])
                if "initial_ir_precursor_atoms" in checkpoint
                else None
            ),
        )
        if "bucket_state" in checkpoint:
            bucket_state = json.loads(str(checkpoint["bucket_state"]))
            # Ce/O bucket keys gained explicit Ir-O coordination.  Older
            # checkpoints retain valid occupations/state, but their cached
            # rates cannot be reused under the corrected Hamiltonian.
            if (
                bucket_state.get("model_version") == 2
                and all(len(key) == 3 for key in bucket_state["ce"]["keys"])
            ):
                engine.restore_bucket_state(bucket_state)
        return engine

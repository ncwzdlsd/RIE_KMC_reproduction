"""Local-update KMC engine for the Ce/O-only S34 reproduction.

This specialized engine omits Ir and sonication bookkeeping so the paper-scale
Ce/O-only case (roughly 0.6 million sites and 5 million events) remains compact.
Every active site belongs to one rate bucket defined by site type, occupation,
and local Ce-O coordination.  An event only updates the changed site and its
nearest Ce-O neighbors.

External-solution connectivity is calculated exactly at initialization.  It is
then maintained locally: adsorption removes one accessible empty site and
surface desorption adds one.  This is exact unless adsorption seals the last
one-site neck leading to a cavity; optional validation can detect that rare
topology change.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import numpy as np

from ceox_events import CeOxParameters, EventType, transition_rate
from constants import SiteType, Species
from generation import find_accessible_empty_sites, find_external_surface, write_xyz
from lattice_build import Lattice


EVENT_KINDS = (
    EventType.CE_ADSORPTION,
    EventType.CE_DESORPTION,
    EventType.O_ADSORPTION,
    EventType.O_DESORPTION,
)
MAX_COORDINATION = {
    EventType.CE_ADSORPTION: 8,
    EventType.CE_DESORPTION: 8,
    EventType.O_ADSORPTION: 4,
    EventType.O_DESORPTION: 4,
}


@dataclass
class FastKMCState:
    step: int = 0
    kmc_time: float = 0.0
    event_counts: Counter | None = None

    def __post_init__(self):
        if self.event_counts is None:
            self.event_counts = Counter()


class RateBuckets:
    """Dense O(1) add/remove/random-choice buckets with global site positions."""

    def __init__(self, number_of_sites: int, keys: list[tuple[EventType, int]]):
        self.keys = keys
        self.code_for_key = {key: code for code, key in enumerate(keys)}
        self.sites: list[list[int]] = [[] for _ in keys]
        self.site_code = np.full(number_of_sites, -1, dtype=np.int8)
        self.site_position = np.full(number_of_sites, -1, dtype=np.int32)

    def add(self, site_id: int, key: tuple[EventType, int]):
        code = self.code_for_key[key]
        if self.site_code[site_id] >= 0:
            raise RuntimeError(f"site {site_id} is already in a rate bucket")
        self.site_code[site_id] = code
        self.site_position[site_id] = len(self.sites[code])
        self.sites[code].append(site_id)

    def discard(self, site_id: int):
        code = int(self.site_code[site_id])
        if code < 0:
            return
        position = int(self.site_position[site_id])
        bucket = self.sites[code]
        last_site = bucket.pop()
        if position < len(bucket):
            bucket[position] = last_site
            self.site_position[last_site] = position
        self.site_code[site_id] = -1
        self.site_position[site_id] = -1

    def random_site(self, code: int, rng: np.random.Generator) -> int:
        bucket = self.sites[code]
        return bucket[int(rng.integers(len(bucket)))]


class FastCeOxKMC:
    def __init__(
        self,
        lattice: Lattice,
        parameters: CeOxParameters,
        random_seed: int = 2026,
        accessible_empty: np.ndarray | None = None,
        state: FastKMCState | None = None,
        rng_state: dict | None = None,
        require_growth_contact: bool = False,
    ):
        self.lattice = lattice
        self.parameters = parameters
        self.rng = np.random.default_rng(random_seed)
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state
        self.state = state or FastKMCState()
        self.require_growth_contact = require_growth_contact
        self.accessible_empty = (
            find_accessible_empty_sites(lattice)
            if accessible_empty is None
            else np.asarray(accessible_empty, dtype=bool)
        )

        keys = [
            (kind, coordination)
            for kind in EVENT_KINDS
            for coordination in range(MAX_COORDINATION[kind] + 1)
        ]
        self.buckets = RateBuckets(lattice.nsites, keys)
        self.rates = np.asarray([self._rate(*key) for key in keys], dtype=float)
        for site_id in range(lattice.nsites):
            key = self._site_key(site_id)
            if key is not None:
                self.buckets.add(site_id, key)

    def _coordination(self, site_id: int) -> int:
        neighbors = self.lattice.get_ce_o_neighbors(site_id)
        if self.lattice.site_type[site_id] == SiteType.M:
            return int(np.count_nonzero(self.lattice.occupation[neighbors] == Species.O))
        return int(np.count_nonzero(self.lattice.occupation[neighbors] == Species.CE))

    def _is_external_surface(self, site_id: int) -> bool:
        neighbors = self.lattice.get_ce_o_neighbors(site_id)
        return bool(np.any(self.accessible_empty[neighbors]))

    def _site_key(self, site_id: int) -> tuple[EventType, int] | None:
        occupation = self.lattice.occupation[site_id]
        site_type = self.lattice.site_type[site_id]
        coordination = self._coordination(site_id)
        if occupation == Species.EMPTY:
            if not self.accessible_empty[site_id]:
                return None
            if self.require_growth_contact and coordination == 0:
                return None
            kind = (
                EventType.CE_ADSORPTION
                if site_type == SiteType.M
                else EventType.O_ADSORPTION
            )
            return kind, coordination
        if not self._is_external_surface(site_id):
            return None
        if occupation == Species.CE:
            return EventType.CE_DESORPTION, coordination
        if occupation == Species.O:
            return EventType.O_DESORPTION, coordination
        return None

    def _rate(self, kind: EventType, coordination: int) -> float:
        binding = coordination * self.parameters.ce_o_binding_energy_ev
        if kind in (EventType.CE_ADSORPTION, EventType.O_ADSORPTION):
            chemical_potential = (
                self.parameters.chemical_potential_ce_ev
                if kind == EventType.CE_ADSORPTION
                else self.parameters.chemical_potential_o_ev
            )
            delta_omega = -chemical_potential - binding
            prefactor = self.parameters.adsorption_prefactor
        else:
            chemical_potential = (
                self.parameters.chemical_potential_ce_ev
                if kind == EventType.CE_DESORPTION
                else self.parameters.chemical_potential_o_ev
            )
            delta_omega = binding + chemical_potential
            prefactor = self.parameters.desorption_prefactor
        return transition_rate(delta_omega, prefactor, self.parameters)

    def total_rate(self) -> float:
        counts = np.fromiter(
            (len(bucket) for bucket in self.buckets.sites),
            dtype=np.int64,
            count=len(self.buckets.sites),
        )
        return float(counts @ self.rates)

    def step_once(self):
        weighted_rates = np.fromiter(
            (
                len(bucket) * rate
                for bucket, rate in zip(self.buckets.sites, self.rates)
            ),
            dtype=float,
            count=len(self.rates),
        )
        total_rate = float(weighted_rates.sum())
        if not math.isfinite(total_rate) or total_rate <= 0.0:
            raise RuntimeError("no available KMC events")
        target = self.rng.random() * total_rate
        code = int(np.searchsorted(np.cumsum(weighted_rates), target, side="right"))
        code = min(code, len(weighted_rates) - 1)
        kind, _ = self.buckets.keys[code]
        site_id = self.buckets.random_site(code, self.rng)

        impacted = [site_id]
        impacted.extend(map(int, self.lattice.get_ce_o_neighbors(site_id)))
        for impacted_site in impacted:
            self.buckets.discard(impacted_site)

        if kind == EventType.CE_ADSORPTION:
            self.lattice.occupation[site_id] = Species.CE
            self.accessible_empty[site_id] = False
        elif kind == EventType.O_ADSORPTION:
            self.lattice.occupation[site_id] = Species.O
            self.accessible_empty[site_id] = False
        elif kind in (EventType.CE_DESORPTION, EventType.O_DESORPTION):
            self.lattice.occupation[site_id] = Species.EMPTY
            self.accessible_empty[site_id] = True
        else:
            raise RuntimeError(f"unsupported event kind {kind}")

        for impacted_site in impacted:
            key = self._site_key(impacted_site)
            if key is not None:
                self.buckets.add(impacted_site, key)

        random_number = max(self.rng.random(), np.finfo(np.float64).tiny)
        self.state.kmc_time += -math.log(random_number) / total_rate
        self.state.step += 1
        self.state.event_counts[kind.value] += 1

    def reconcile_accessibility(self) -> int:
        """Recompute exact solution connectivity and rebuild affected buckets."""
        exact=find_accessible_empty_sites(self.lattice)
        difference_count=int(np.count_nonzero(exact!=self.accessible_empty))
        if difference_count:
            self.accessible_empty=exact
            keys=self.buckets.keys
            self.buckets=RateBuckets(self.lattice.nsites,keys)
            for site_id in range(self.lattice.nsites):
                key=self._site_key(site_id)
                if key is not None:
                    self.buckets.add(site_id,key)
        return difference_count

    def run_to(
        self,
        target_step: int,
        progress_every: int = 100_000,
        reconcile_every: int = 0,
    ):
        started = time.perf_counter()
        previous_step = self.state.step
        while self.state.step < target_step:
            self.step_once()
            if reconcile_every and self.state.step % reconcile_every == 0:
                corrected=self.reconcile_accessibility()
                if corrected:
                    print(
                        f"step={self.state.step:,} reconciled "
                        f"{corrected:,} connectivity sites",
                        flush=True,
                    )
            if progress_every and self.state.step % progress_every == 0:
                elapsed = time.perf_counter() - started
                completed = self.state.step - previous_step
                rate = completed / elapsed if elapsed else 0.0
                print(
                    f"step={self.state.step:,} rate={rate:,.0f} events/s "
                    f"KMC_time={self.state.kmc_time:.6e}",
                    flush=True,
                )

    def exact_connectivity_error_count(self) -> int:
        exact = find_accessible_empty_sites(self.lattice)
        return int(np.count_nonzero(exact != self.accessible_empty))

    def metrics(self) -> dict:
        occupation = self.lattice.occupation
        external_surface = find_external_surface(self.lattice)
        number_ce = int(np.count_nonzero(occupation == Species.CE))
        number_o = int(np.count_nonzero(occupation == Species.O))
        return {
            "step": self.state.step,
            "KMC_time": self.state.kmc_time,
            "number_Ce": number_ce,
            "number_O": number_o,
            "O_to_Ce": number_o / number_ce if number_ce else 0.0,
            "surface_Ce": int(
                np.count_nonzero(external_surface & (occupation == Species.CE))
            ),
            "surface_O": int(
                np.count_nonzero(external_surface & (occupation == Species.O))
            ),
            "connectivity_error_sites": self.exact_connectivity_error_count(),
            **{
                f"count_{kind.value}": self.state.event_counts[kind.value]
                for kind in EVENT_KINDS
            },
        }

    def write_snapshot(self, filename: Path):
        write_xyz(
            filename,
            self.lattice,
            comment=f"step={self.state.step} KMC_time={self.state.kmc_time:.10e}",
        )

    def save_checkpoint(self, filename: Path):
        filename.parent.mkdir(parents=True, exist_ok=True)
        bucket_lengths=np.asarray(
            [len(bucket) for bucket in self.buckets.sites],dtype=np.int64
        )
        bucket_offsets=np.concatenate(
            (np.asarray([0],dtype=np.int64),np.cumsum(bucket_lengths))
        )
        bucket_sites=np.fromiter(
            (site for bucket in self.buckets.sites for site in bucket),
            dtype=np.int32,
            count=int(bucket_offsets[-1]),
        )
        np.savez_compressed(
            filename,
            occupation=self.lattice.occupation,
            accessible_empty=self.accessible_empty,
            step=np.asarray(self.state.step, dtype=np.int64),
            kmc_time=np.asarray(self.state.kmc_time, dtype=np.float64),
            event_counts=np.asarray(json.dumps(dict(self.state.event_counts))),
            rng_state=np.asarray(json.dumps(self.rng.bit_generator.state)),
            bucket_offsets=bucket_offsets,
            bucket_sites=bucket_sites,
            require_growth_contact=np.asarray(self.require_growth_contact),
        )


def load_checkpoint(
    filename: Path,
    lattice: Lattice,
    parameters: CeOxParameters,
    require_growth_contact: bool = False,
):
    with np.load(filename, allow_pickle=False) as checkpoint:
        lattice.occupation[:] = checkpoint["occupation"]
        state = FastKMCState(
            step=int(checkpoint["step"]),
            kmc_time=float(checkpoint["kmc_time"]),
            event_counts=Counter(json.loads(str(checkpoint["event_counts"]))),
        )
        engine=FastCeOxKMC(
            lattice,
            parameters,
            accessible_empty=checkpoint["accessible_empty"],
            state=state,
            rng_state=json.loads(str(checkpoint["rng_state"])),
            require_growth_contact=require_growth_contact,
        )
        saved_contact_rule=(
            bool(checkpoint["require_growth_contact"])
            if "require_growth_contact" in checkpoint
            else False
        )
        if (
            "bucket_offsets" in checkpoint
            and "bucket_sites" in checkpoint
            and saved_contact_rule==require_growth_contact
        ):
            offsets=checkpoint["bucket_offsets"]
            flat_sites=checkpoint["bucket_sites"]
            engine.buckets.site_code.fill(-1)
            engine.buckets.site_position.fill(-1)
            engine.buckets.sites=[]
            for code in range(len(offsets)-1):
                bucket=flat_sites[offsets[code]:offsets[code+1]].astype(int).tolist()
                engine.buckets.sites.append(bucket)
                for position,site_id in enumerate(bucket):
                    engine.buckets.site_code[site_id]=code
                    engine.buckets.site_position[site_id]=position
        return engine

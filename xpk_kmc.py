"""Extended phenomenological kinetic (XPK) acceleration for ``LocalKMC``.

The implementation follows the two-stage construction used by Shen and Xu:

1. keep the chemical composition fixed and sample the fast diffusion-only
   lattice process;
2. average the propensities of all non-diffusive events over that ensemble and
   advance only the chemical-space clock.

Only ionic Ir diffusion is placed in the fast subspace because it is the only
diffusion process present in the five-reaction model of the target paper.
Ce/O exchange, Ir adsorption/desorption and redox, and acoustic corrosion stay
explicit.  In particular, no rate constant, chemical potential, barrier, or
event definition is rescaled by this module.

The original XPK test interpolates apparent rate constants between coverage
points.  This reproduction deliberately samples the current chemical state
instead: CeOx morphology and the location of metallic Ir are observables here,
so treating two different morphologies as the same coverage point would change
the target model.  The result is the zero-interpolation-error form of the same
diffusion-ensemble averaging procedure.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Hashable

import numpy as np

from constants import Species
from local_kmc import LocalKMC


# Numerical settings reported for the paper's 40x40 hydrogenation test.  They
# are recorded for provenance, not transplanted into this model: its chemical
# state contains an evolving three-dimensional CeOx/Ir morphology rather than
# a single adsorbate coverage coordinate.
XPK_PAPER_REFERENCE_DIFFUSION_STEPS_PER_POINT = 16_000_000
XPK_PAPER_REFERENCE_COVERAGE_INTERVAL = 0.04


@dataclass(frozen=True)
class XPKSamplingParameters:
    """Numerical controls for the diffusion-only ensemble.

    A sweep is one attempted accepted diffusion hop per currently mobile Ir
    ion.  These values control statistical convergence only; they do not alter
    physical rates or advance physical time.  Convergence must be checked by
    increasing all three controls for production calculations.
    """

    equilibration_sweeps: float = 1.0
    samples: int = 8
    decorrelation_sweeps: float = 0.25

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.equilibration_sweeps)
            or self.equilibration_sweeps < 0.0
        ):
            raise ValueError("equilibration_sweeps must be finite and non-negative")
        if self.samples <= 0:
            raise ValueError("samples must be positive")
        if (
            not math.isfinite(self.decorrelation_sweeps)
            or self.decorrelation_sweeps < 0.0
        ):
            raise ValueError("decorrelation_sweeps must be finite and non-negative")


@dataclass
class XPKStatistics:
    diffusion_sampling_steps: int = 0
    ensemble_evaluations: int = 0
    chemical_space_steps: int = 0
    zero_diffusion_ensembles: int = 0


class XPKLocalKMC(LocalKMC):
    """Local KMC with XPK diffusion averaging and a chemical-space clock."""

    def __init__(
        self,
        *args,
        xpk_sampling: XPKSamplingParameters | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.xpk_sampling = xpk_sampling or XPKSamplingParameters()
        self.xpk_statistics = XPKStatistics()

    def _slow_total_rate(self) -> float:
        """Total propensity of the chemical-space (non-diffusive) reactions."""
        # XPK sampling can perform and reverse many fast hops between chemical
        # events.  Recompute dense totals from their current populations to
        # prevent round-off drift from repeated incremental add/subtract pairs.
        ce_rate = math.fsum(
            rate * len(members)
            for rate, members in zip(self.ce_buckets.rates, self.ce_buckets.members)
        )
        redox_rate = math.fsum(
            rate * len(members)
            for rate, members in zip(
                self.ir_redox_buckets.rates, self.ir_redox_buckets.members
            )
        )
        self.ce_buckets.total_rate = ce_rate
        self.ir_redox_buckets.total_rate = redox_rate
        return ce_rate + self._ir_exchange_total_rate() + redox_rate

    def total_rate(self) -> float:
        """Return the current chemical-space rate, excluding sampled diffusion."""
        return self._slow_total_rate()

    def _draw_slow_reaction_event(
        self, total_rate: float
    ) -> tuple[str, Hashable | None, object | None]:
        ir_exchange_rate = self._ir_exchange_total_rate()
        family_rates = (
            ("ce", self.ce_buckets.total_rate, self.ce_buckets),
            ("ir_exchange", ir_exchange_rate, self.ir_exchange_buckets),
            ("ir_redox", self.ir_redox_buckets.total_rate, self.ir_redox_buckets),
        )
        if not math.isfinite(total_rate) or total_rate <= 0.0:
            raise RuntimeError("no available XPK chemical-space reactions")
        target = self.rng.random() * total_rate
        cumulative = 0.0
        for name, rate, buckets in family_rates:
            if target < cumulative + rate:
                if name == "ir_exchange":
                    key, member = self._select_ir_exchange_event(target - cumulative)
                else:
                    key, member = buckets.select(target - cumulative, self.rng)
                return name, key, member
            cumulative += rate
        raise RuntimeError("failed to select an XPK chemical-space reaction")

    def _diffusion_hop(self) -> tuple[int, int] | None:
        """Perform one diffusion-only sampling hop without advancing time."""
        total_rate = math.fsum(
            rate * len(members)
            for rate, members in zip(
                self.diffusion_buckets.rates, self.diffusion_buckets.members
            )
        )
        self.diffusion_buckets.total_rate = total_rate
        if not math.isfinite(total_rate) or total_rate <= 0.0:
            return None
        target_rate = self.rng.random() * total_rate
        _, member = self.diffusion_buckets.select(target_rate, self.rng)
        source, target = map(int, member)
        self._apply_site_changes({source: Species.EMPTY, target: Species.IR_ION})
        self.xpk_statistics.diffusion_sampling_steps += 1
        return source, target

    def _run_diffusion_steps(
        self, requested_steps: int, recorded_moves: list[tuple[int, int]]
    ) -> int:
        completed = 0
        for _ in range(requested_steps):
            move = self._diffusion_hop()
            if move is None:
                break
            recorded_moves.append(move)
            completed += 1
        return completed

    def _restore_sample(
        self, recorded_moves: list[tuple[int, int]], retained_move_count: int
    ) -> None:
        """Reverse the diffusion-only suffix to materialize a weighted sample."""
        for source, target in reversed(recorded_moves[retained_move_count:]):
            if self.lattice.occupation[target] != Species.IR_ION:
                raise RuntimeError(
                    "cannot reverse XPK diffusion sample: target is not Ir ion"
                )
            if self.lattice.occupation[source] != Species.EMPTY:
                raise RuntimeError(
                    "cannot reverse XPK diffusion sample: source is not empty"
                )
            self._apply_site_changes({target: Species.EMPTY, source: Species.IR_ION})

    def _diffusion_ensemble(
        self, condition_rate: float
    ) -> tuple[list[float], list[int], list[tuple[int, int]]]:
        mobile_ir = int(np.count_nonzero(self.lattice.occupation == Species.IR_ION))
        sweep_size = max(mobile_ir, 1)
        equilibration_steps = int(
            math.ceil(self.xpk_sampling.equilibration_sweeps * sweep_size)
        )
        decorrelation_steps = int(
            math.ceil(self.xpk_sampling.decorrelation_sweeps * sweep_size)
        )

        moves: list[tuple[int, int]] = []
        completed = self._run_diffusion_steps(equilibration_steps, moves)
        if equilibration_steps and completed == 0:
            self.xpk_statistics.zero_diffusion_ensembles += 1

        sample_rates: list[float] = []
        sample_move_counts: list[int] = []
        for sample_index in range(self.xpk_sampling.samples):
            if sample_index and decorrelation_steps:
                self._run_diffusion_steps(decorrelation_steps, moves)
            total_rate = self._slow_total_rate() + condition_rate
            if not math.isfinite(total_rate) or total_rate < 0.0:
                raise RuntimeError("invalid XPK ensemble-averaged propensity")
            sample_rates.append(total_rate)
            sample_move_counts.append(len(moves))

        self.xpk_statistics.ensemble_evaluations += 1
        return sample_rates, sample_move_counts, moves

    def step_once(self, stop_time: float | None = None) -> bool:
        """Advance one XPK chemical-space or independent condition event."""
        condition_rate = self._sonication_rate()
        sample_rates, sample_move_counts, moves = self._diffusion_ensemble(
            condition_rate
        )
        mean_total_rate = float(np.mean(sample_rates))
        if mean_total_rate <= 0.0:
            raise RuntimeError("no available XPK chemical-space or condition events")

        random_number = max(self.rng.random(), np.finfo(np.float64).tiny)
        delta_time = -math.log(random_number) / mean_total_rate
        if stop_time is not None and self.state.kmc_time + delta_time > stop_time:
            self.state.kmc_time = stop_time
            return False

        self.state.kmc_time += delta_time

        rate_sum = float(sum(sample_rates))
        sample_target = self.rng.random() * rate_sum
        cumulative = 0.0
        sample_index = len(sample_rates) - 1
        for index, rate in enumerate(sample_rates):
            cumulative += rate
            if sample_target < cumulative:
                sample_index = index
                break
        self._restore_sample(moves, sample_move_counts[sample_index])

        slow_rate = self._slow_total_rate()
        event_target = self.rng.random() * (slow_rate + condition_rate)
        if event_target >= slow_rate:
            event_name = self._apply_sonication_event()
            self.state.condition_event_counts[event_name] += 1
            return True

        family, key, member = self._draw_slow_reaction_event(slow_rate)
        event_name = self._apply_drawn_event(family, key, member)
        self.state.step += 1
        self.state.event_counts[event_name] += 1
        self.xpk_statistics.chemical_space_steps += 1
        return True

    def metrics(self) -> dict:
        row = super().metrics()
        row.update(
            {
                "xpk_diffusion_sampling_steps": (
                    self.xpk_statistics.diffusion_sampling_steps
                ),
                "xpk_ensemble_evaluations": self.xpk_statistics.ensemble_evaluations,
                "xpk_chemical_space_steps": self.xpk_statistics.chemical_space_steps,
                "xpk_zero_diffusion_ensembles": (
                    self.xpk_statistics.zero_diffusion_ensembles
                ),
            }
        )
        return row

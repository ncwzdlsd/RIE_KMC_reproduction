"""Calibration of physical KMC rates against supplementary Tables S3 and S5."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from kinetic_parameters import KineticParameterSet
from local_kmc import LocalKMC
from paper_parameters import (
    PAPER_CHEMICAL_POTENTIAL_CE_EV,
    PAPER_CHEMICAL_POTENTIAL_O_EV,
    PAPER_TARGET_TIMES_MIN,
)
from run_sonication_comparison import build_initial_lattice


# Particle diameters from supplementary Table S3, normalized by 7.9 nm so a
# compact calibration lattice can reproduce relative growth without pretending
# to represent the experimental particle count.
DIAMETER_RATIO_TARGETS = {
    "no_sonication": {
        0.0: 1.0,
        30.0: 8.0 / 7.9,
        60.0: 8.5 / 7.9,
        120.0: 9.0 / 7.9,
        180.0: 9.3 / 7.9,
    },
    "sonication": {
        0.0: 1.0,
        5.0: 8.1 / 7.9,
        30.0: 9.2 / 7.9,
        60.0: 10.4 / 7.9,
        180.0: 10.6 / 7.9,
    },
}

DISSOLVED_CE_UG_ML = {
    "no_sonication": {5.0: 1.70, 30.0: 1.78, 60.0: 2.35, 120.0: 2.57, 180.0: 2.70},
    "sonication": {5.0: 3.94, 30.0: 4.95, 60.0: 7.08, 120.0: 7.13, 180.0: 7.35},
}

# Table S5 uses 50 mg CeOx in 80 mL.  This converts a modeled net Ce fraction
# to the corresponding maximum-batch concentration under the single-particle
# representative approximation.
CE_ATOMIC_MASS = 140.116
O_ATOMIC_MASS = 15.999
CEO2_CE_MASS_FRACTION = CE_ATOMIC_MASS / (CE_ATOMIC_MASS + 2.0 * O_ATOMIC_MASS)
MAX_CE_UG_ML = 50_000.0 * CEO2_CE_MASS_FRACTION / 80.0


@dataclass(frozen=True)
class CalibrationConfig:
    box_nm: float = 4.8
    particle_diameter_nm: float = 4.0
    roughness_fraction: float = 0.05
    replicates: int = 2
    iterations: int = 12
    random_seed: int = 2026
    maximum_events_per_interval: int = 500_000
    acceptance_objective: float = 4.0


def simulate_metrics(
    parameters: KineticParameterSet,
    sonication: bool,
    seed: int,
    config: CalibrationConfig,
) -> dict[float, dict]:
    lattice = build_initial_lattice(
        seed,
        config.box_nm,
        config.particle_diameter_nm,
        config.roughness_fraction,
    )
    engine = LocalKMC(
        lattice,
        parameters.ceox_parameters(),
        parameters.ir_parameters(),
        sonication_parameters=(parameters.sonication_parameters() if sonication else None),
        random_seed=seed,
    )
    result = {}
    for time_min in PAPER_TARGET_TIMES_MIN:
        engine.advance_to_time(
            time_min * 60.0,
            maximum_events=config.maximum_events_per_interval,
        )
        result[time_min] = engine.metrics()
    return result


def calibration_objective(
    parameters: KineticParameterSet,
    config: CalibrationConfig,
) -> tuple[float, dict]:
    condition_rows = {"no_sonication": [], "sonication": []}
    for replicate in range(config.replicates):
        seed = config.random_seed + replicate
        condition_rows["no_sonication"].append(
            simulate_metrics(parameters, False, seed, config)
        )
        condition_rows["sonication"].append(
            simulate_metrics(parameters, True, seed, config)
        )

    residuals = []
    summary = {}
    for condition, replicate_rows in condition_rows.items():
        initial_diameters = np.asarray(
            [rows[0.0]["equivalent_diameter_nm"] for rows in replicate_rows]
        )
        condition_summary = {}
        for time_min in PAPER_TARGET_TIMES_MIN:
            diameter_ratios = np.asarray(
                [
                    rows[time_min]["equivalent_diameter_nm"] / initial
                    for rows, initial in zip(replicate_rows, initial_diameters)
                ]
            )
            released_fractions = np.asarray(
                [rows[time_min]["net_released_Ce_fraction"] for rows in replicate_rows]
            )
            mean_ratio = float(np.mean(diameter_ratios))
            mean_fraction = float(np.mean(released_fractions))
            condition_summary[str(time_min)] = {
                "diameter_ratio": mean_ratio,
                "dissolved_ce_ug_ml": mean_fraction * MAX_CE_UG_ML,
            }
            diameter_target = DIAMETER_RATIO_TARGETS[condition].get(time_min)
            if diameter_target is not None:
                residuals.append((mean_ratio - diameter_target) / 0.03)
            dissolution_target = DISSOLVED_CE_UG_ML[condition].get(time_min)
            if dissolution_target is not None:
                sigma = max(0.75, 0.20 * dissolution_target)
                residuals.append(
                    (mean_fraction * MAX_CE_UG_ML - dissolution_target) / sigma
                )
        summary[condition] = condition_summary
    objective = float(np.mean(np.square(residuals)))
    return objective, summary


def calibrate(
    initial: KineticParameterSet,
    config: CalibrationConfig,
) -> tuple[KineticParameterSet, list[dict], dict]:
    if config.iterations <= 0 or config.replicates <= 0:
        raise ValueError("iterations and replicates must be positive")
    initial = replace(
        initial,
        chemical_potential_ce_ev=PAPER_CHEMICAL_POTENTIAL_CE_EV,
        chemical_potential_o_ev=PAPER_CHEMICAL_POTENTIAL_O_EV,
    )
    rng = np.random.default_rng(config.random_seed)
    best_scales = np.asarray([1.0, 1.0], dtype=float)
    best_parameters = initial
    best_objective, best_summary = calibration_objective(best_parameters, config)
    history = [
        {
            "iteration": 0,
            "objective": best_objective,
            "ce_scale": 1.0,
            "sonication_scale": 1.0,
            "chemical_potential_ce_o_ev": initial.chemical_potential_ce_ev,
            "accepted": True,
        }
    ]

    for iteration in range(1, config.iterations + 1):
        cooling = max(0.15, 1.0 - iteration / (config.iterations + 1.0))
        proposed_scales = best_scales * np.exp(rng.normal(0.0, 0.9 * cooling, size=2))
        candidate = initial.scaled(
            ce_scale=float(proposed_scales[0]),
            sonication_scale=float(proposed_scales[1]),
        )
        objective, summary = calibration_objective(candidate, config)
        accepted = objective < best_objective
        if accepted:
            best_scales = proposed_scales
            best_parameters = candidate
            best_objective = objective
            best_summary = summary
        history.append(
            {
                "iteration": iteration,
                "objective": objective,
                "ce_scale": float(proposed_scales[0]),
                "sonication_scale": float(proposed_scales[1]),
                "chemical_potential_ce_o_ev": initial.chemical_potential_ce_ev,
                "accepted": accepted,
            }
        )

    calibrated = replace(
        best_parameters,
        calibrated=best_objective <= config.acceptance_objective,
        calibration_objective=best_objective,
        calibration_scope=(
            "Ce/O exchange time scale and per-interface-site sonication "
            "propensity fitted to Tables S3/S5 at fixed Ce/O chemical "
            "potential; "
            "Ir rates remain "
            "explicit initial estimates"
        ),
    )
    return calibrated, history, best_summary

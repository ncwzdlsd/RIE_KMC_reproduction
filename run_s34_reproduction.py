"""Reproduce the public-information portion of supplementary Fig. S34.

Paper-defined settings:
  * 20 x 20 x 20 nm^3 fluorite lattice box (approximately 0.5 million sites)
  * 5 nm rough CeO2 particle
  * T = 453 K, Ce/O chemical potentials = -0.60 eV
  * snapshots at 0, 1e6, 3e6, and 5e6 KMC events

The supplement does not quantify the random roughness.  It remains an explicit
command-line assumption and is recorded in run_metadata.json.
"""

from __future__ import annotations

import argparse
from builtins import str
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np

from ceox_events import CeOxParameters
from fast_ceox_kmc import FastCeOxKMC, load_checkpoint
from generation import initialize_sphere, roughen_surface
from lattice_build import build_fluorite_lattice
from paper_parameters import (
    DFT_CE_O_BINDING_ENERGY_EV,
    PAPER_BOX_NM,
    PAPER_CHEMICAL_POTENTIAL_CE_EV,
    PAPER_PARTICLE_DIAMETER_NM,
    PAPER_SNAPSHOT_STEPS,
    PAPER_TEMPERATURE_K,
)


PAPER_CHEMICAL_POTENTIAL_EV = PAPER_CHEMICAL_POTENTIAL_CE_EV
PAPER_CE_O_BINDING_EV = DFT_CE_O_BINDING_ENERGY_EV


def parse_steps(value: str) -> tuple[int, ...]:
    steps = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if any(step < 0 for step in steps):
        raise argparse.ArgumentTypeError("snapshot steps must be non-negative")
    return steps


def build_initial_lattice(
    random_seed: int,
    roughness_fraction: float,
):
    ncells = math.ceil(PAPER_BOX_NM / 0.541)
    lattice = build_fluorite_lattice(ncells=ncells)
    rng = np.random.default_rng(random_seed)
    initialize_sphere(
        lattice,
        diameter_nm=PAPER_PARTICLE_DIAMETER_NM,
        oxygen_x=2.0,
        rng=rng,
    )
    roughen_surface(lattice, fraction=roughness_fraction, rng=rng)
    return lattice


def newest_checkpoint(condition_directory: Path, target_step: int) -> Path | None:
    candidates = []
    for path in condition_directory.glob("checkpoint_*.npz"):
        try:
            step = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        if step <= target_step:
            candidates.append((step, path))
    return max(candidates, default=(None, None))[1]


def append_metric(filename: Path, row: dict):
    existing_rows = []
    if filename.exists():
        with filename.open("r", encoding="utf-8", newline="") as source:
            existing_rows = list(csv.DictReader(source))
    rows_by_step = {int(existing["step"]): existing for existing in existing_rows}
    rows_by_step[int(row["step"])] = row
    rows = [rows_by_step[step] for step in sorted(rows_by_step)]
    with filename.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(rows)


def write_ovito_readme(root: Path):
    (root / "OVITO_README.txt").write_text(
        "Open particle/snapshot_00000000.xyz as a file sequence.\n"
        "The trajectory contains one rough 5 nm CeO2 particle.\n"
        "Frames correspond to 0, 1e6, 3e6, and 5e6 KMC events.\n"
        "Use Particle Type colors (Ce/O); hide the simulation cell if desired.\n"
        "The kinetic prefactor ratio is fitted because the paper does not report it.\n",
        encoding="utf-8",
    )


def run_particle(args, parameters: CeOxParameters):
    condition_directory = args.output / "particle"
    condition_directory.mkdir(parents=True, exist_ok=True)
    lattice = build_initial_lattice(
        random_seed=args.seed,
        roughness_fraction=args.roughness_fraction,
    )
    checkpoint = newest_checkpoint(condition_directory, args.steps) if args.resume else None
    if checkpoint is None:
        engine = FastCeOxKMC(
            lattice,
            parameters,
            random_seed=args.seed,
            require_growth_contact=True,
        )
    else:
        print(f"resuming particle from {checkpoint.name}", flush=True)
        engine = load_checkpoint(
            checkpoint,
            lattice,
            parameters,
            require_growth_contact=True,
        )

    requested_snapshots = sorted(
        set(step for step in args.snapshot_steps if step <= args.steps) | {args.steps}
    )
    for target_step in requested_snapshots:
        if target_step < engine.state.step:
            continue
        while target_step > engine.state.step:
            chunk_target=min(target_step,engine.state.step+args.checkpoint_every)
            engine.run_to(
                chunk_target,
                progress_every=args.progress_every,
                reconcile_every=args.reconcile_every,
            )
            if chunk_target < target_step:
                engine.save_checkpoint(
                    condition_directory/f"checkpoint_{chunk_target:08d}.npz"
                )
        corrected=engine.reconcile_accessibility()
        if corrected:
            print(
                f"step={target_step:,} snapshot reconciliation corrected "
                f"{corrected:,} connectivity sites",
                flush=True,
            )
        snapshot = condition_directory / f"snapshot_{target_step:08d}.xyz"
        checkpoint = condition_directory / f"checkpoint_{target_step:08d}.npz"
        engine.write_snapshot(snapshot)
        engine.save_checkpoint(checkpoint)
        metric = {"condition": "particle", **engine.metrics()}
        append_metric(condition_directory / "metrics.csv", metric)
        print(
            f"saved particle step={target_step:,}; "
            f"Ce={metric['number_Ce']:,} O={metric['number_O']:,}; "
            f"connectivity_errors={metric['connectivity_error_sites']}",
            flush=True,
        )
    return engine.state.step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5_000_000)
    parser.add_argument(
        "--snapshot-steps",
        type=parse_steps,
        default=PAPER_SNAPSHOT_STEPS,
        help="comma-separated KMC steps (default: 0,1000000,3000000,5000000)",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--roughness-fraction", type=float, default=0.05)
    parser.add_argument(
        "--adsorption-prefactor",
        type=float,
        default=0.1,
        help="fitted relative rate; not reported by the paper",
    )
    parser.add_argument(
        "--desorption-prefactor",
        type=float,
        default=1.0,
        help="reference relative rate; not reported by the paper",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("kmc_output") / "s34_reproduction"
    )
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument(
        "--reconcile-every",
        type=int,
        default=250_000,
        help="recompute exact exterior connectivity at this step interval",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=250_000,
        help="save a resumable checkpoint at this step interval",
    )
    parser.add_argument(
        "--no-resume", action="store_false", dest="resume",
        help="ignore existing checkpoints",
    )
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be non-negative")
    if not 0.0 <= args.roughness_fraction <= 1.0:
        parser.error("--roughness-fraction must be between 0 and 1")
    if args.adsorption_prefactor <= 0.0 or args.desorption_prefactor <= 0.0:
        parser.error("kinetic prefactors must be positive")
    if args.reconcile_every < 0:
        parser.error("--reconcile-every must be non-negative")
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be positive")

    parameters = CeOxParameters(
        temperature_k=PAPER_TEMPERATURE_K,
        ce_o_binding_energy_ev=PAPER_CE_O_BINDING_EV,
        chemical_potential_ce_ev=PAPER_CHEMICAL_POTENTIAL_EV,
        chemical_potential_o_ev=PAPER_CHEMICAL_POTENTIAL_EV,
        adsorption_prefactor=args.adsorption_prefactor,
        desorption_prefactor=args.desorption_prefactor,
        exchange_barrier_ev=0.0,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "target": "Science 387 (2025), supplementary Fig. S34",
        "paper_defined": {
            "box_nm": [20.0, 20.0, 20.0],
            "particle_diameter_nm": PAPER_PARTICLE_DIAMETER_NM,
            "temperature_k": PAPER_TEMPERATURE_K,
            "chemical_potential_ce_ev": PAPER_CHEMICAL_POTENTIAL_EV,
            "chemical_potential_o_ev": PAPER_CHEMICAL_POTENTIAL_EV,
            "ce_o_binding_energy_ev": PAPER_CE_O_BINDING_EV,
            "snapshot_steps": list(PAPER_SNAPSHOT_STEPS),
        },
        "explicit_assumptions_not_reported_by_paper": {
            "roughness_fraction": args.roughness_fraction,
            "adsorption_prefactor": parameters.adsorption_prefactor,
            "desorption_prefactor": parameters.desorption_prefactor,
            "exchange_barrier_ev": parameters.exchange_barrier_ev,
            "prefactor_calibration_note": (
                "The adsorption/desorption ratio was calibrated so the 5 nm "
                "particle approaches the approximate S34 size range instead "
                "of crystallizing the full box."
            ),
        },
        "run": {
            "target_steps": args.steps,
            "snapshot_steps": list(args.snapshot_steps),
            "seed": args.seed,
            "geometry": "rough_spherical_particle",
            "resume": args.resume,
            "reconcile_every": args.reconcile_every,
            "checkpoint_every": args.checkpoint_every,
        },
        "parameters": asdict(parameters),
        "interpretation_limit": (
            "The paper states that undisclosed kinetic parameters were fitted. "
            "Therefore this run targets morphology trends and event-step panels, "
            "not an independently verified mapping to physical time."
        ),
        "model_choice": (
            "Ce/O adsorption requires at least one opposite-sublattice crystal "
            "neighbor. This suppresses unphysical homogeneous crystallization "
            "throughout the finite lattice box."
        ),
    }
    (args.output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    run_particle(args, parameters)
    write_ovito_readme(args.output)


if __name__ == "__main__":
    main()

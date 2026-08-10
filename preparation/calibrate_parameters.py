"""Fit physical KMC rate scales to supplementary Tables S3 and S5."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, "") and str(PROJECT_ROOT) not in sys.path:
    # Allow direct execution with: py -3 preparation/calibrate_parameters.py
    sys.path.insert(0, str(PROJECT_ROOT))

from preparation.calibration import CalibrationConfig, calibrate
from kinetic_parameters import KineticParameterSet


DEFAULT_OUTPUT = PROJECT_ROOT / "calibrated_parameters.json"
DEFAULT_ITERATIONS = 40
DEFAULT_REPLICATES = 5


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-parameters", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--box-nm", type=float, default=4.8)
    parser.add_argument("--particle-diameter-nm", type=float, default=4.0)
    parser.add_argument("--roughness-fraction", type=float, default=0.05)
    parser.add_argument("--maximum-events-per-interval", type=int, default=500_000)
    parser.add_argument("--acceptance-objective", type=float, default=4.0)
    args = parser.parse_args(argv)

    print("Starting one-time KMC parameter preparation.", flush=True)
    print(
        f"Calibration settings: iterations={args.iterations}, "
        f"replicates={args.replicates}, seed={args.seed}",
        flush=True,
    )
    print(f"Parameter output: {args.output.resolve()}", flush=True)

    initial = (
        KineticParameterSet.read(args.initial_parameters)
        if args.initial_parameters
        else KineticParameterSet()
    )
    config = CalibrationConfig(
        box_nm=args.box_nm,
        particle_diameter_nm=args.particle_diameter_nm,
        roughness_fraction=args.roughness_fraction,
        replicates=args.replicates,
        iterations=args.iterations,
        random_seed=args.seed,
        maximum_events_per_interval=args.maximum_events_per_interval,
        acceptance_objective=args.acceptance_objective,
    )
    calibrated, history, summary = calibrate(initial, config)
    calibrated.write(
        args.output,
        metadata={
            "calibration_config": asdict(config),
            "history": history,
            "best_simulated_targets": summary,
        },
    )
    report = {
        "parameter_file": str(args.output),
        "accepted": calibrated.calibrated,
        "objective": calibrated.calibration_objective,
        "scope": calibrated.calibration_scope,
        "accepted_iterations": [row for row in history if row["accepted"]],
    }
    report_path = args.output.with_name(f"{args.output.stem}_report.json")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Calibrated parameters: {args.output.resolve()}")
    print(f"Calibration report: {report_path.resolve()}")
    print(f"Objective: {calibrated.calibration_objective:.6g}")
    if not calibrated.calibrated:
        print(
            "Calibration did not meet the acceptance threshold; the output is "
            "marked uncalibrated and comparison runs will label it diagnostic."
        )
    else:
        print(
            "Preparation complete. Run run_sonication_comparison.py to start "
            "the standard 180 min comparison."
        )


if __name__ == "__main__":
    main()

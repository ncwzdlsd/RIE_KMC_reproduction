"""Time-resolved paired KMC simulation with and without sonication."""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, replace
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
from collections.abc import Sequence

import numpy as np

from generation import initialize_sphere, roughen_surface
from kinetic_parameters import KineticParameterSet
from lattice_build import build_fluorite_lattice
from local_kmc import LocalKMC
from paper_parameters import (
    PAPER_CHEMICAL_POTENTIAL_CE_EV,
    PAPER_CHEMICAL_POTENTIAL_O_EV,
    PAPER_TARGET_TIMES_MIN,
)


SEPARATION_NM = 8.0


def allocate_run_directory(output_hint: Path | None) -> Path:
    """Reserve a timestamped directory without ever reusing an old run."""
    if output_hint is None:
        parent = Path("kmc_output")
        prefix = "time_comparison"
    else:
        parent = output_hint.parent
        prefix = output_hint.name
    parent.mkdir(parents=True, exist_ok=True)
    for collision_index in range(10_000):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        collision_suffix = "" if collision_index == 0 else f"_{collision_index:04d}"
        candidate = parent / f"{prefix}_{stamp}{collision_suffix}"
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError("could not allocate a unique output directory")


def parse_times(value: str) -> tuple[float, ...]:
    try:
        times = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("times must be comma-separated numbers") from error
    if not times or any(not math.isfinite(item) or item < 0.0 for item in times):
        raise argparse.ArgumentTypeError("times must be finite and non-negative")
    if tuple(sorted(set(times))) != times:
        raise argparse.ArgumentTypeError("times must be unique and increasing")
    if times[0] != 0.0:
        raise argparse.ArgumentTypeError("the first target time must be 0")
    return times


def build_initial_lattice(
    random_seed: int,
    box_nm: float,
    particle_diameter_nm: float,
    roughness_fraction: float,
):
    rng = np.random.default_rng(random_seed)
    lattice = build_fluorite_lattice(ncells=math.ceil(box_nm / 0.541))
    initialize_sphere(
        lattice,
        diameter_nm=particle_diameter_nm,
        oxygen_x=2.0,
        rng=rng,
    )
    roughen_surface(lattice, fraction=roughness_fraction, rng=rng)
    return lattice


def read_snapshot_records(filename: Path) -> list[tuple]:
    records = []
    for line in filename.read_text(encoding="utf-8").splitlines()[2:]:
        fields = line.split()
        records.append(
            (
                fields[0],
                float(fields[1]),
                float(fields[2]),
                float(fields[3]),
                int(fields[4]),
                int(fields[5]),
                int(fields[6]),
                int(fields[7]),
            )
        )
    return records


def write_paired_frame(
    control_filename: Path,
    sonicated_filename: Path,
    output_filename: Path,
    sample_time_min: float,
) -> None:
    paired_records = []
    for condition, source, shift in (
        (0, control_filename, -0.5 * SEPARATION_NM),
        (1, sonicated_filename, 0.5 * SEPARATION_NM),
    ):
        for name, x, y, z, surface, ir_state, embedded, contacts in read_snapshot_records(source):
            paired_records.append(
                (
                    name,
                    x + shift,
                    y,
                    z,
                    surface,
                    ir_state,
                    embedded,
                    contacts,
                    condition,
                )
            )

    with output_filename.open("w", encoding="utf-8", newline="\n") as output:
        output.write(f"{len(paired_records)}\n")
        output.write(
            "Properties=species:S:1:pos:R:3:surface:I:1:ir_state:I:1:"
            "embedded:I:1:support_contacts:I:1:condition:I:1 "
            f"time_min={sample_time_min:g} condition=0:no_sonication "
            "condition=1:sonication ir_output=support_connected_only\n"
        )
        for record in paired_records:
            name, x, y, z, surface, ir_state, embedded, contacts, condition = record
            output.write(
                f"{name} {x:.6f} {y:.6f} {z:.6f} {surface} "
                f"{ir_state} {embedded} {contacts} {condition}\n"
            )


def write_trajectory(
    raw_root: Path,
    output_filename: Path,
    target_times_min: tuple[float, ...],
    snapshot_directory: Path,
) -> list[Path]:
    snapshot_directory.mkdir(parents=True, exist_ok=True)

    snapshot_files = []
    with output_filename.open("w", encoding="utf-8", newline="\n") as trajectory:
        for sample_index, time_min in enumerate(target_times_min):
            name = f"snapshot_{sample_index:04d}.xyz"
            time_label = f"{time_min:g}".replace(".", "p")
            frame_file = snapshot_directory / (
                f"snapshot_{sample_index:02d}_{time_label}min.xyz"
            )
            write_paired_frame(
                raw_root / "no_sonication" / name,
                raw_root / "sonication" / name,
                frame_file,
                time_min,
            )
            trajectory.write(frame_file.read_text(encoding="utf-8"))
            snapshot_files.append(frame_file)
    return snapshot_files


def write_metrics(
    filename: Path,
    control_metrics: list[dict],
    sonicated_metrics: list[dict],
) -> None:
    metric_names = (
        "KMC_time",
        "step",
        "number_Ce",
        "number_O",
        "number_Ir_ion",
        "number_Ir",
        "attached_Ir_ion",
        "attached_Ir",
        "attached_Ir_total",
        "attached_Ir_fraction",
        "unattached_Ir_ion",
        "unattached_Ir",
        "unattached_Ir_total",
        "embedded_Ir_total",
        "Ir_embedding_fraction",
        "mean_Ir_support_contacts",
        "Ir_cluster_count",
        "Ir_nanoparticle_count_ge_3",
        "largest_Ir_cluster_atoms",
        "mean_Ir_Ir_coordination",
        "largest_Ir_cluster_radius_gyration_nm",
        "largest_Ir_cluster_shape_anisotropy",
        "equivalent_diameter_nm",
        "net_released_Ce_atoms",
        "net_released_Ce_fraction",
        "net_released_O_atoms",
        "net_adsorbed_Ir_atoms",
        "initial_Ir_precursor_atoms",
        "solution_Ir_precursor_atoms",
        "Ir_precursor_fraction_remaining",
        "Ir_inventory_error_atoms",
        "effective_chemical_potential_Ce_ev",
        "effective_chemical_potential_O_ev",
        "sonication_removed_Ce_atoms",
        "sonication_removed_O_atoms",
        "interface_support_site_count",
        "sonication_total_propensity_per_s",
        "sonication_event_count",
        "sonication_removed_atoms",
    )
    rows = []
    for control, sonicated in zip(control_metrics, sonicated_metrics):
        row = {"sample_time_min": control["sample_time_min"]}
        for name in metric_names:
            control_value = control[name]
            sonicated_value = sonicated[name]
            row[f"no_sonication_{name}"] = control_value
            row[f"sonication_{name}"] = sonicated_value
            row[f"delta_{name}"] = sonicated_value - control_value
        rows.append(row)
    with filename.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_condition(
    engine: LocalKMC,
    target_times_min: tuple[float, ...],
    output_directory: Path,
    reconcile_every: int,
    maximum_events_per_interval: int | None,
    keep_checkpoint: bool,
) -> list[dict]:
    output_directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample_index, time_min in enumerate(target_times_min):
        time_seconds = time_min * 60.0
        engine.advance_to_time(
            time_seconds,
            reconcile_every=reconcile_every,
            maximum_events=maximum_events_per_interval,
        )
        row = engine.metrics()
        row["sample_time_min"] = time_min
        rows.append(row)
        engine.write_snapshot(
            output_directory / f"snapshot_{sample_index:04d}.xyz",
            sample_time=time_seconds,
            supported_ir_only=True,
        )
        if keep_checkpoint:
            engine.save_checkpoint(output_directory / "checkpoint_latest.npz")
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-times-min",
        type=parse_times,
        default=PAPER_TARGET_TIMES_MIN,
        help="Increasing comma-separated times; default: 0,5,30,60,120,180.",
    )
    parser.add_argument("--parameter-file", type=Path)
    parser.add_argument(
        "--allow-uncalibrated",
        action="store_true",
        help="Allow built-in initial guesses or an uncalibrated parameter file.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output naming hint. A timestamp is always appended so an "
            "existing run is never overwritten."
        ),
    )
    parser.add_argument("--box-nm", type=float, default=4.8)
    parser.add_argument("--particle-diameter-nm", type=float, default=4.0)
    parser.add_argument("--roughness-fraction", type=float, default=0.05)
    parser.add_argument(
        "--ir-precursor-atoms",
        type=int,
        help=(
            "Override the finite total Ir inventory. By default it is scaled "
            "from the initial Ce count using the Table S9 RIE Ir/Ce ratio."
        ),
    )
    parser.add_argument("--reconcile-every", type=int, default=100_000)
    parser.add_argument("--maximum-events-per-interval", type=int)
    parser.add_argument("--keep-checkpoints", action="store_true")
    args = parser.parse_args(argv)

    if args.box_nm <= args.particle_diameter_nm:
        parser.error("--box-nm must exceed --particle-diameter-nm")
    if not 0.0 <= args.roughness_fraction <= 1.0:
        parser.error("--roughness-fraction must be between 0 and 1")
    if args.reconcile_every < 0:
        parser.error("--reconcile-every must be non-negative")
    if args.maximum_events_per_interval is not None and args.maximum_events_per_interval <= 0:
        parser.error("--maximum-events-per-interval must be positive")
    if args.ir_precursor_atoms is not None and args.ir_precursor_atoms <= 0:
        parser.error("--ir-precursor-atoms must be positive")

    parameters = (
        KineticParameterSet.read(args.parameter_file)
        if args.parameter_file is not None
        else KineticParameterSet()
    )
    # The formal comparison is the fixed-high-chemical-potential case used in
    # supplementary Figs. S31/S34.  A legacy parameter file may contain the
    # former -0.69 -> -0.60 feedback settings, so force the selected bath here.
    parameters = replace(
        parameters,
        chemical_potential_ce_ev=PAPER_CHEMICAL_POTENTIAL_CE_EV,
        chemical_potential_o_ev=PAPER_CHEMICAL_POTENTIAL_O_EV,
    )
    if not parameters.calibrated and not args.allow_uncalibrated:
        parser.error(
            "physical-time simulation requires calibrated parameters; run "
            "calibrate_parameters.py or pass --allow-uncalibrated for diagnostics"
        )

    initial_lattice = build_initial_lattice(
        args.seed,
        args.box_nm,
        args.particle_diameter_nm,
        args.roughness_fraction,
    )
    control_lattice = copy.deepcopy(initial_lattice)
    sonicated_lattice = copy.deepcopy(initial_lattice)
    control_engine = LocalKMC(
        control_lattice,
        parameters.ceox_parameters(),
        parameters.ir_parameters(),
        random_seed=args.seed,
        initial_ir_precursor_atoms=args.ir_precursor_atoms,
    )
    sonicated_engine = LocalKMC(
        sonicated_lattice,
        parameters.ceox_parameters(),
        parameters.ir_parameters(),
        sonication_parameters=parameters.sonication_parameters(),
        random_seed=args.seed,
        initial_ir_precursor_atoms=args.ir_precursor_atoms,
    )

    root = allocate_run_directory(args.output)
    raw_root = root / "_raw"
    control_metrics = run_condition(
        control_engine,
        args.target_times_min,
        raw_root / "no_sonication",
        args.reconcile_every,
        args.maximum_events_per_interval,
        args.keep_checkpoints,
    )
    sonicated_metrics = run_condition(
        sonicated_engine,
        args.target_times_min,
        raw_root / "sonication",
        args.reconcile_every,
        args.maximum_events_per_interval,
        args.keep_checkpoints,
    )

    snapshot_files = write_trajectory(
        raw_root,
        root / "trajectory.xyz",
        args.target_times_min,
        root / "snapshots",
    )
    write_metrics(root / "metrics.csv", control_metrics, sonicated_metrics)
    metadata = {
        "target_times_min": args.target_times_min,
        "arguments": {
            **vars(args),
            "target_times_min": list(args.target_times_min),
            "parameter_file": str(args.parameter_file) if args.parameter_file else None,
            "output": str(args.output) if args.output else None,
        },
        "kinetic_parameters": asdict(parameters),
        "rate_unit": "s^-1",
        "time_unit": "s",
        "output_directory": str(root.resolve()),
        "output_policy": "unique_timestamped_directory_never_overwrite",
        "environment_model": {
            "solution": "implicit_well_mixed_solution",
            "lattice": "CeOx_support_plus_explicit_box_transport_Ir_on_M_sites",
            "ce_o_exchange": "solution_exposed_CeOx_growth_sites",
            "o_ir_interface": "O adsorption/desorption includes adjacent ionic and metallic Ir binding",
            "ce_o_inventory": "grand canonical; net exchange is derived from event counts only",
            "chemical_potential_response": "fixed; no feedback from desorption or cumulative sonication dissolution",
            "fixed_chemical_potential_Ce_ev": parameters.chemical_potential_ce_ev,
            "fixed_chemical_potential_O_ev": parameters.chemical_potential_o_ev,
            "sonication_event_catalog": "one_candidate_KMC_event_per_current_nanoparticle_solution_interface_site",
            "sonication_event_selection": "n_fold_way_total_propensity_then_uniform_interface_center",
            "sonication_rate_parameter_unit": "s^-1_per_interface_site",
            "ir_exchange": "adsorption_and_desorption_only_at_box_edge_M_sites",
            "ir_inventory": "finite_conserved_precursor_reservoir",
            "ir_adsorption_activity": "remaining_precursor_fraction",
            "initial_ir_precursor_atoms": control_engine.initial_ir_precursor_atoms,
            "initial_ce_atoms": control_engine.initial_ce_atoms,
            "ir_to_ce_atom_ratio": parameters.precursor_ir_to_ce_atom_ratio,
            "ir_capacity_basis": "Table_S9_RIE_Ir_to_Ce_atom_ratio_scaled_by_initial_Ce",
            "ir_diffusion": "nearest_neighbor_hops_over_all_solution_accessible_empty_M_sites",
            "ir_reduction": "only_at_support_or_existing_metallic_Ir_attachment_sites",
            "xyz_ir_visibility": "only_Ir_clusters_connected_to_CeOx; unattached_transport_Ir_remains_in_KMC_state",
            "ir_morphology": "diffusion_dominated_ionic_relaxation_before_reduction",
            "metallic_ir_surface_diffusion": False,
            "bulk_ir_reduction": False,
        },
        "condition_labels": {"0": "no_sonication", "1": "sonication"},
        "snapshot_files": [
            str(path.relative_to(root)) for path in snapshot_files
        ],
        "final_metrics": {
            "no_sonication": control_metrics[-1],
            "sonication": sonicated_metrics[-1],
        },
        "event_counts": {
            "no_sonication": dict(control_engine.state.event_counts),
            "sonication": dict(sonicated_engine.state.event_counts),
        },
    }
    (root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not args.keep_checkpoints:
        shutil.rmtree(raw_root)
    else:
        for path in raw_root.rglob("snapshot_*.xyz"):
            path.unlink()
        frame = raw_root / "paired_frame.xyz"
        if frame.exists():
            frame.unlink()

    print(f"Output directory: {root.resolve()}")
    print(f"Individual snapshots: {(root / 'snapshots').resolve()}")
    print(
        f"No sonication: {control_engine.state.step:,} events, "
        f"t={control_engine.state.kmc_time / 60.0:.1f} min"
    )
    print(
        f"Sonication: {sonicated_engine.state.step:,} events, "
        f"t={sonicated_engine.state.kmc_time / 60.0:.1f} min"
    )
    print(
        "Finite Ir precursor per condition: "
        f"{control_engine.initial_ir_precursor_atoms} atoms; remaining "
        f"no-sonication={control_engine.state.solution_ir_precursor_atoms}, "
        f"sonication={sonicated_engine.state.solution_ir_precursor_atoms}"
    )
    print(
        "Final supported/unattached Ir: "
        f"no-sonication={control_metrics[-1]['attached_Ir_total']}/"
        f"{control_metrics[-1]['unattached_Ir_total']}, "
        f"sonication={sonicated_metrics[-1]['attached_Ir_total']}/"
        f"{sonicated_metrics[-1]['unattached_Ir_total']}"
    )


if __name__ == "__main__":
    main()

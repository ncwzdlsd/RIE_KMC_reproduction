"""Compare supported Ir exposure/embedding with and without sonication."""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
from collections.abc import Sequence

from generation import build_initial_lattice
from kinetic_parameters import KineticParameterSet
from local_kmc import LocalKMC
from paper_parameters import (
    PAPER_CHEMICAL_POTENTIAL_CE_EV,
    PAPER_CHEMICAL_POTENTIAL_O_EV,
    PAPER_TARGET_TIMES_MIN,
)
from sonication_events import SonicationEventType
from xpk_kmc import (
    XPKLocalKMC,
    XPKSamplingParameters,
    XPK_PAPER_REFERENCE_COVERAGE_INTERVAL,
    XPK_PAPER_REFERENCE_DIFFUSION_STEPS_PER_POINT,
)


SEPARATION_NM = 8.0
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PARAMETER_FILE = PROJECT_ROOT / "calibrated_parameters.json"
DEFAULT_OUTPUT_HINT = PROJECT_ROOT / "kmc_output" / "comparison_180min"
DEFAULT_METHOD = "kmc"
XPK_METRIC_NAMES = (
    "xpk_diffusion_sampling_steps",
    "xpk_ensemble_evaluations",
    "xpk_chemical_space_steps",
    "xpk_zero_diffusion_ensembles",
)


def fixed_paper_setting_conflicts(
    parameters: KineticParameterSet,
) -> list[tuple[str, float, float]]:
    """Return physical settings that conflict with the fixed reproduction."""
    expected_settings = (
        (
            "chemical_potential_ce_ev",
            parameters.chemical_potential_ce_ev,
            PAPER_CHEMICAL_POTENTIAL_CE_EV,
        ),
        (
            "chemical_potential_o_ev",
            parameters.chemical_potential_o_ev,
            PAPER_CHEMICAL_POTENTIAL_O_EV,
        ),
        (
            "sonication_chemical_potential_shift_ev",
            parameters.sonication_chemical_potential_shift_ev,
            0.0,
        ),
    )
    return [
        (name, actual, expected)
        for name, actual, expected in expected_settings
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12)
    ]


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
    visualization_mode: str = "structure",
) -> None:
    paired_records = []
    for condition, source, shift in (
        (0, control_filename, -0.5 * SEPARATION_NM),
        (1, sonicated_filename, 0.5 * SEPARATION_NM),
    ):
        for name, x, y, z, surface, ir_state, embedded, contacts in read_snapshot_records(source):
            if visualization_mode == "ir_only" and name != "Ir":
                continue
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
        property_schema = (
            "Properties=species:S:1:pos:R:3:surface:I:1:ir_state:I:1:"
            "embedded:I:1:support_contacts:I:1:condition:I:1"
        )
        if visualization_mode in ("ir_emphasis", "ir_only"):
            property_schema += ":Radius:R:1:Transparency:R:1"
        output.write(
            property_schema + " "
            f"time_min={sample_time_min:g} condition=0:no_sonication "
            "condition=1:sonication output=main_CeOx_connected_only "
            f"visualization={visualization_mode}\n"
        )
        for record in paired_records:
            name, x, y, z, surface, ir_state, embedded, contacts, condition = record
            row = (
                f"{name} {x:.6f} {y:.6f} {z:.6f} {surface} "
                f"{ir_state} {embedded} {contacts} {condition}"
            )
            if visualization_mode in ("ir_emphasis", "ir_only"):
                radius = {"Ce": 0.085, "O": 0.045, "Ir": 0.19}[name]
                transparency = {"Ce": 0.60, "O": 0.82, "Ir": 0.0}[name]
                row += f" {radius:.3f} {transparency:.2f}"
            output.write(row + "\n")


def write_trajectory(
    raw_root: Path,
    output_filename: Path,
    target_times_min: tuple[float, ...],
    snapshot_directory: Path,
    visualization_mode: str = "structure",
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
                visualization_mode=visualization_mode,
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
        "main_particle_Ce",
        "main_particle_O",
        "detached_support_atoms",
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
        "support_shape_anisotropy",
        "support_axis_extent_ratio",
        "support_surface_radial_cv",
        "net_released_Ce_atoms",
        "net_released_Ce_fraction",
        "net_released_O_atoms",
        "net_adsorbed_Ir_atoms",
        "target_supported_Ir_atoms",
        "main_particle_Ir_target_fraction",
        "assumed_Ir_retention_fraction",
        "initial_Ir_precursor_atoms",
        "solution_Ir_precursor_atoms",
        "Ir_precursor_fraction_remaining",
        "Ir_adsorption_activity_relative_to_target",
        "Ir_inventory_error_atoms",
        "effective_chemical_potential_Ce_ev",
        "effective_chemical_potential_O_ev",
        "sonication_removed_Ce_atoms",
        "sonication_removed_O_atoms",
        "interface_support_site_count",
        "sonication_condition_frequency_per_s",
        "sonication_total_propensity_per_s",
        "sonication_event_count",
        "sonication_removed_atoms",
        *XPK_METRIC_NAMES,
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


def write_ir_embedding_comparison(
    filename: Path,
    control_metrics: list[dict],
    sonicated_metrics: list[dict],
) -> None:
    """Write the focused supported/surface/embedded Ir comparison."""
    rows = []
    for control, sonicated in zip(control_metrics, sonicated_metrics):
        control_surface = (
            control["attached_Ir_total"] - control["embedded_Ir_total"]
        )
        sonicated_surface = (
            sonicated["attached_Ir_total"] - sonicated["embedded_Ir_total"]
        )
        rows.append(
            {
                "sample_time_min": control["sample_time_min"],
                "no_sonication_supported_Ir": control["attached_Ir_total"],
                "no_sonication_surface_Ir": control_surface,
                "no_sonication_embedded_Ir": control["embedded_Ir_total"],
                "no_sonication_embedding_fraction": control[
                    "Ir_embedding_fraction"
                ],
                "sonication_supported_Ir": sonicated["attached_Ir_total"],
                "sonication_surface_Ir": sonicated_surface,
                "sonication_embedded_Ir": sonicated["embedded_Ir_total"],
                "sonication_embedding_fraction": sonicated[
                    "Ir_embedding_fraction"
                ],
                "delta_embedded_Ir": (
                    sonicated["embedded_Ir_total"]
                    - control["embedded_Ir_total"]
                ),
                "delta_embedding_fraction": (
                    sonicated["Ir_embedding_fraction"]
                    - control["Ir_embedding_fraction"]
                ),
            }
        )
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
        for metric_name in XPK_METRIC_NAMES:
            row.setdefault(metric_name, 0)
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
    parser.add_argument(
        "--parameter-file",
        type=Path,
        help=(
            "Kinetic parameter JSON. If omitted, calibrated_parameters.json "
            "beside this script is loaded when present; otherwise built-in "
            "estimates are used."
        ),
    )
    parser.add_argument(
        "--require-calibrated",
        action="store_true",
        help="Stop instead of running when the selected parameters are uncalibrated.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--method",
        choices=("xpk", "kmc"),
        default=DEFAULT_METHOD,
        help=(
            "Simulation method. xpk separates diffusion-only sampling from "
            "chemical-space evolution; kmc retains every physical diffusion hop."
        ),
    )
    parser.add_argument(
        "--xpk-equilibration-sweeps",
        type=float,
        default=1.0,
        help="Diffusion-only equilibration sweeps per XPK chemical state.",
    )
    parser.add_argument(
        "--xpk-samples",
        type=int,
        default=8,
        help="Diffusion-ensemble samples per XPK chemical state.",
    )
    parser.add_argument(
        "--xpk-decorrelation-sweeps",
        type=float,
        default=0.25,
        help="Diffusion-only sweeps between XPK ensemble samples.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_HINT,
        help=(
            "Output naming hint. A timestamp is always appended so an "
            "existing run is never overwritten."
        ),
    )
    parser.add_argument("--box-nm", type=float, default=20.0)
    parser.add_argument("--particle-diameter-nm", type=float, default=5.0)
    parser.add_argument("--roughness-fraction", type=float, default=0.05)
    parser.add_argument(
        "--ir-precursor-atoms",
        type=int,
        help=(
            "Override the finite total Ir dose. By default the Table S9 final "
            "supported-Ir/Ce target is divided by the assumed precursor "
            "retention fraction."
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
    if args.xpk_equilibration_sweeps < 0.0:
        parser.error("--xpk-equilibration-sweeps must be non-negative")
    if args.xpk_samples <= 0:
        parser.error("--xpk-samples must be positive")
    if args.xpk_decorrelation_sweeps < 0.0:
        parser.error("--xpk-decorrelation-sweeps must be non-negative")

    selected_parameter_file = args.parameter_file
    if selected_parameter_file is None and DEFAULT_PARAMETER_FILE.exists():
        candidate_parameters = KineticParameterSet.read(DEFAULT_PARAMETER_FILE)
        if candidate_parameters.calibrated:
            selected_parameter_file = DEFAULT_PARAMETER_FILE
        else:
            print(
                f"Ignoring failed calibration file {DEFAULT_PARAMETER_FILE}; "
                "using the current built-in diagnostic parameters.",
                flush=True,
            )
    parameters = (
        KineticParameterSet.read(selected_parameter_file)
        if selected_parameter_file is not None
        else KineticParameterSet()
    )
    args.parameter_file = selected_parameter_file
    # The reproduction uses one fixed bath for both conditions.  Reject an
    # incompatible file rather than silently modifying a physical parameter.
    conflicts = fixed_paper_setting_conflicts(parameters)
    if conflicts:
        details = ", ".join(
            f"{name}={actual} (expected {expected})"
            for name, actual, expected in conflicts
        )
        parser.error(
            f"parameter file conflicts with fixed paper settings: {details}; "
            "the program will not rewrite them implicitly"
        )
    if not parameters.calibrated and args.require_calibrated:
        parser.error(
            "the selected kinetic parameters are not calibrated"
        )
    if parameters.calibrated:
        print(f"Using calibrated parameters from {selected_parameter_file}", flush=True)
    else:
        source = selected_parameter_file or "built-in initial estimates"
        print(
            f"WARNING: using {source}; results are diagnostic because the "
            "kinetic parameters are not calibrated.",
            flush=True,
        )

    initial_lattice = build_initial_lattice(
        args.seed,
        args.box_nm,
        args.particle_diameter_nm,
        args.roughness_fraction,
    )
    control_lattice = copy.deepcopy(initial_lattice)
    sonicated_lattice = copy.deepcopy(initial_lattice)
    engine_type = XPKLocalKMC if args.method == "xpk" else LocalKMC
    xpk_sampling = XPKSamplingParameters(
        equilibration_sweeps=args.xpk_equilibration_sweeps,
        samples=args.xpk_samples,
        decorrelation_sweeps=args.xpk_decorrelation_sweeps,
    )
    xpk_arguments = {"xpk_sampling": xpk_sampling} if args.method == "xpk" else {}
    print(
        f"Simulation method: {args.method.upper()}"
        + (
            f" (equilibration={xpk_sampling.equilibration_sweeps} sweeps, "
            f"samples={xpk_sampling.samples}, "
            f"decorrelation={xpk_sampling.decorrelation_sweeps} sweeps)"
            if args.method == "xpk"
            else ""
        ),
        flush=True,
    )
    if args.method == "xpk":
        print(
            "WARNING: XPK production results require convergence against larger "
            "diffusion-ensemble settings and the explicit --method kmc baseline.",
            flush=True,
        )
    control_engine = engine_type(
        control_lattice,
        parameters.ceox_parameters(sonication=False),
        parameters.ir_parameters(),
        random_seed=args.seed,
        initial_ir_precursor_atoms=args.ir_precursor_atoms,
        **xpk_arguments,
    )
    sonicated_engine = engine_type(
        sonicated_lattice,
        parameters.ceox_parameters(sonication=True),
        parameters.ir_parameters(),
        sonication_parameters=parameters.sonication_parameters(),
        random_seed=args.seed,
        initial_ir_precursor_atoms=args.ir_precursor_atoms,
        **xpk_arguments,
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
    ir_emphasis_snapshot_files = write_trajectory(
        raw_root,
        root / "trajectory_ir_emphasis.xyz",
        args.target_times_min,
        root / "snapshots_ir_emphasis",
        visualization_mode="ir_emphasis",
    )
    ir_only_snapshot_files = write_trajectory(
        raw_root,
        root / "trajectory_ir_only.xyz",
        args.target_times_min,
        root / "snapshots_ir_only",
        visualization_mode="ir_only",
    )
    write_metrics(root / "metrics.csv", control_metrics, sonicated_metrics)
    write_ir_embedding_comparison(
        root / "ir_embedding_comparison.csv",
        control_metrics,
        sonicated_metrics,
    )
    final_control = control_metrics[-1]
    final_sonicated = sonicated_metrics[-1]
    first_post_initial_index = 1 if len(control_metrics) > 1 else 0
    early_control = control_metrics[first_post_initial_index]
    early_sonicated = sonicated_metrics[first_post_initial_index]
    ir_embedding_checks = {
        "sonication_increases_Ir_embedding_fraction": (
            final_sonicated["Ir_embedding_fraction"]
            > final_control["Ir_embedding_fraction"]
        ),
        "Ir_is_visible_after_first_nonzero_sample": (
            early_control["attached_Ir_total"] > 0
            or early_sonicated["attached_Ir_total"] > 0
        ),
        "Ir_inventory_is_conserved": (
            final_control["Ir_inventory_error_atoms"] == 0
            and final_sonicated["Ir_inventory_error_atoms"] == 0
        ),
    }
    ir_embedding_checks["all_pass"] = all(ir_embedding_checks.values())
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
        "simulation_method": args.method,
        "xpk_sampling": (
            asdict(xpk_sampling) if args.method == "xpk" else None
        ),
        "xpk_sampling_convergence_required": args.method == "xpk",
        "xpk_paper_reference_sampling": (
            {
                "diffusion_steps_per_interpolation_point": (
                    XPK_PAPER_REFERENCE_DIFFUSION_STEPS_PER_POINT
                ),
                "coverage_interval": XPK_PAPER_REFERENCE_COVERAGE_INTERVAL,
                "interpolation": "linear",
                "scope": "paper_hydrogenation_test_not_transplanted_as_a_physical_parameter",
            }
            if args.method == "xpk"
            else None
        ),
        "output_directory": str(root.resolve()),
        "output_policy": "unique_timestamped_directory_never_overwrite",
        "environment_model": {
            "solution": "implicit_well_mixed_solution",
            "lattice": "CeOx_support_plus_explicit_box_transport_Ir_on_M_sites",
            "ce_o_exchange": "solution_exposed_CeOx_growth_sites",
            "o_ir_interface": "O adsorption/desorption includes adjacent ionic and metallic Ir binding",
            "ce_o_inventory": "grand canonical; net exchange is derived from event counts only",
            "chemical_potential_response": "fixed_identical_minus_0.60_eV_Ce_O_bath_values; acoustic_removal_has_no_feedback",
            "fixed_chemical_potential_Ce_ev": parameters.chemical_potential_ce_ev,
            "fixed_chemical_potential_O_ev": parameters.chemical_potential_o_ev,
            "sonication_chemical_potential_shift_ev": parameters.sonication_chemical_potential_shift_ev,
            "sonication_event_catalog": "independent_acoustic_condition_clock_per_current_nanoparticle_solution_interface_site",
            "sonication_event_selection": "independent_Poisson_clock_then_uniform_interface_center; excluded_from_KMC_step_count",
            "sonication_rate_parameter_unit": "s^-1_per_interface_site",
            "sonication_frequency_metric": "sonication_condition_frequency_per_s",
            "ir_exchange": "adsorption_and_desorption_only_at_box_edge_M_sites",
            "ir_inventory": "finite_conserved_precursor_reservoir",
            "ir_adsorption_activity": "remaining_precursor_atoms_divided_by_Table_S9_supported_Ir_target_at_fixed_box_volume",
            "initial_ir_precursor_atoms": control_engine.initial_ir_precursor_atoms,
            "target_supported_ir_atoms": control_engine.target_supported_ir_atoms,
            "initial_ce_atoms": control_engine.initial_ce_atoms,
            "ir_to_ce_atom_ratio": parameters.precursor_ir_to_ce_atom_ratio,
            "assumed_ir_retention_fraction": parameters.precursor_retention_fraction,
            "ir_capacity_basis": "user_requested_approximately_600_atom_precursor_dose_for_standard_5nm_support_while_Table_S9_defines_the_separate_supported_Ir_target",
            "ir_diffusion": (
                "XPK_diffusion_only_ensemble_sampling_excluded_from_physical_time"
                if args.method == "xpk"
                else "nearest_neighbor_hops_over_all_solution_accessible_empty_M_sites"
            ),
            "xpk_fast_subspace": (
                "Ir_ion_diffusion_only" if args.method == "xpk" else None
            ),
            "xpk_slow_subspace": (
                "Ce_O_exchange_Ir_exchange_Ir_redox_and_independent_sonication"
                if args.method == "xpk"
                else None
            ),
            "xpk_interpolation": (
                "disabled_to_avoid_merging_distinct_CeOx_and_metallic_Ir_morphologies"
                if args.method == "xpk"
                else None
            ),
            "ir_reduction": "only_at_support_or_existing_metallic_Ir_attachment_sites",
            "xyz_visibility": "largest_CeOx_component_and_only_Ir_clusters_connected_to_that_main_particle; detached_fragments_and_transport_Ir_remain_in_KMC_state",
            "ir_morphology": "diffusion_dominated_ionic_relaxation_before_reduction",
            "metallic_ir_surface_diffusion": False,
            "bulk_ir_reduction": False,
        },
        "condition_labels": {"0": "no_sonication", "1": "sonication"},
        "snapshot_files": [
            str(path.relative_to(root)) for path in snapshot_files
        ],
        "ir_emphasis_snapshot_files": [
            str(path.relative_to(root)) for path in ir_emphasis_snapshot_files
        ],
        "ir_only_snapshot_files": [
            str(path.relative_to(root)) for path in ir_only_snapshot_files
        ],
        "ir_embedding_checks": ir_embedding_checks,
        "final_metrics": {
            "no_sonication": final_control,
            "sonication": final_sonicated,
        },
        "event_counts": {
            "no_sonication": dict(control_engine.state.event_counts),
            "sonication": dict(sonicated_engine.state.event_counts),
        },
        "condition_event_counts": {
            "no_sonication": dict(control_engine.state.condition_event_counts),
            "sonication": dict(sonicated_engine.state.condition_event_counts),
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
        "Independent sonication-condition events: "
        f"{sonicated_engine.state.condition_event_counts[SonicationEventType.CORROSION.value]:,}"
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
    print(
        "Final surface/embedded supported Ir: "
        f"no-sonication="
        f"{final_control['attached_Ir_total'] - final_control['embedded_Ir_total']}/"
        f"{final_control['embedded_Ir_total']} "
        f"({final_control['Ir_embedding_fraction']:.3f} embedded), "
        f"sonication="
        f"{final_sonicated['attached_Ir_total'] - final_sonicated['embedded_Ir_total']}/"
        f"{final_sonicated['embedded_Ir_total']} "
        f"({final_sonicated['Ir_embedding_fraction']:.3f} embedded)"
    )
    print(
        "Ir-embedding checks: "
        + ("PASS" if ir_embedding_checks["all_pass"] else "REVIEW")
        + " "
        + json.dumps(ir_embedding_checks, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()

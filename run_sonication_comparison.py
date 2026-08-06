"""Generate paired OVITO trajectories with and without sonication-assisted ripening.

The 1 nm corrosion radius and 10% dissolution probability come from the paper.
The sonication event rate is a visualization parameter because its fitted value
was not reported in the supplementary information.  A compact mean-field growth
burst represents Ce/O supplied by unresolved sacrificial particles so that the
embedding mechanism remains visible in a short trajectory.
"""

import argparse
import copy
import csv
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from ceox_events import CeOxParameters
from generation import (
    initialize_sphere,
    roughen_surface,
    seed_ir_nanoparticle,
)
from kmc_engine import KMCRunConfig, run_KMC
from lattice_build import build_fluorite_lattice
from paper_parameters import (
    DFT_CE_O_BINDING_ENERGY_EV,
    PAPER_DISSOLUTION_PROBABILITY,
    PAPER_SONICATION_RADIUS_NM,
    PAPER_TEMPERATURE_K,
)
from sonication_events import SonicationParameters


SEPARATION_NM = 8.0


def build_initial_lattice(random_seed: int):
    rng = np.random.default_rng(random_seed)
    lattice = build_fluorite_lattice(ncells=11)
    # A compact 4 nm support leaves enough room for pre-nucleated Ir particles
    # and for the support to grow around them in a short visualization run.
    initialize_sphere(lattice, diameter_nm=4.0, oxygen_x=2.0, rng=rng)
    roughen_surface(lattice, fraction=0.05, rng=rng)
    center = lattice.center_nm
    for direction in (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ):
        seed_ir_nanoparticle(
            lattice,
            center_nm=center + 2.25 * direction,
            diameter_nm=0.9,
        )
    return lattice


def read_snapshot_records(filename: Path):
    lines = filename.read_text(encoding="utf-8").splitlines()
    records = []
    for line in lines[2:]:
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


def filter_ir_environment(records, radius_nm: float):
    ir_positions = np.asarray(
        [(record[1], record[2], record[3]) for record in records if record[0] == "Ir"],
        dtype=float,
    )
    if len(ir_positions) == 0:
        return []
    kept = []
    for record in records:
        position = np.asarray(record[1:4], dtype=float)
        if np.any(np.linalg.norm(ir_positions - position, axis=1) <= radius_nm):
            kept.append(record)
    return kept


def write_paired_snapshot(
    control_filename: Path,
    sonicated_filename: Path,
    output_filename: Path,
    environment_radius_nm: float | None = None,
):
    paired_records = []
    for condition, source, shift in (
        (0, control_filename, -0.5 * SEPARATION_NM),
        (1, sonicated_filename, 0.5 * SEPARATION_NM),
    ):
        records = read_snapshot_records(source)
        if environment_radius_nm is not None:
            records = filter_ir_environment(records, environment_radius_nm)
        for name, x, y, z, surface, ir_state, embedded, contacts in records:
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

    output_filename.parent.mkdir(parents=True, exist_ok=True)
    with output_filename.open("w", encoding="utf-8", newline="\n") as output:
        output.write(f"{len(paired_records)}\n")
        output.write(
            "Properties=species:S:1:pos:R:3:surface:I:1:ir_state:I:1:"
            "embedded:I:1:support_contacts:I:1:condition:I:1 "
            "condition=0:no_sonication condition=1:sonication\n"
        )
        for record in paired_records:
            name, x, y, z, surface, ir_state, embedded, contacts, condition = record
            output.write(
                f"{name} {x:.6f} {y:.6f} {z:.6f} {surface} "
                f"{ir_state} {embedded} {contacts} {condition}\n"
            )


def write_paired_process_trajectories(root: Path):
    control_snapshots = {
        path.name: path for path in (root / "no_sonication").glob("snapshot_*.xyz")
    }
    sonicated_snapshots = {
        path.name: path for path in (root / "sonication").glob("snapshot_*.xyz")
    }
    common_names = sorted(control_snapshots.keys() & sonicated_snapshots.keys())
    for name in common_names:
        write_paired_snapshot(
            control_snapshots[name],
            sonicated_snapshots[name],
            root / "paired_process" / name,
        )
        write_paired_snapshot(
            control_snapshots[name],
            sonicated_snapshots[name],
            root / "paired_ir_environment" / name,
            environment_radius_nm=1.0,
        )
    return common_names


def write_process_metrics(filename: Path, control_metrics, sonicated_metrics):
    control_by_step = {int(row["step"]): row for row in control_metrics}
    sonicated_by_step = {int(row["step"]): row for row in sonicated_metrics}
    metric_names = (
        "KMC_time",
        "number_Ce",
        "number_O",
        "number_Ir",
        "embedded_Ir_total",
        "Ir_embedding_fraction",
        "mean_Ir_support_contacts",
        "highly_covered_Ir_fraction",
        "equivalent_diameter_nm",
        "sonication_event_count",
        "sonication_removed_atoms",
        "sonication_redeposited_atoms",
        "solution_chemical_potential_boost_ev",
    )
    rows = []
    for step in sorted(control_by_step.keys() & sonicated_by_step.keys()):
        row = {"step": step}
        for metric_name in metric_names:
            control_value = control_by_step[step][metric_name]
            sonicated_value = sonicated_by_step[step][metric_name]
            row[f"no_sonication_{metric_name}"] = control_value
            row[f"sonication_{metric_name}"] = sonicated_value
            if isinstance(control_value, (int, float)) and isinstance(
                sonicated_value, (int, float)
            ):
                row[f"delta_{metric_name}"] = sonicated_value - control_value
        rows.append(row)
    with filename.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--snapshot-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--sonication-rate",
        type=float,
        default=20.0,
        help="Visualization-only total event rate; not reported by the paper.",
    )
    args = parser.parse_args()

    initial_lattice = build_initial_lattice(args.seed)
    control_lattice = copy.deepcopy(initial_lattice)
    sonicated_lattice = copy.deepcopy(initial_lattice)

    ceox_parameters = CeOxParameters(
        temperature_k=PAPER_TEMPERATURE_K,
        ce_o_binding_energy_ev=DFT_CE_O_BINDING_ENERGY_EV,
        chemical_potential_ce_ev=-0.69,
        chemical_potential_o_ev=-0.69,
        adsorption_prefactor=1.0,
        desorption_prefactor=1.0,
        exchange_barrier_ev=0.0,
    )
    sonication_parameters = SonicationParameters(
        event_rate=args.sonication_rate,
        radius_nm=PAPER_SONICATION_RADIUS_NM,
        dissolution_probability=PAPER_DISSOLUTION_PROBABILITY,
        maximum_chemical_potential_boost_ev=0.09,
        events_for_maximum_boost=10,
        mean_field_growth_atoms_per_event=24,
        growth_capture_radius_nm=1.0,
    )

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path("kmc_output") / f"sonication_comparison_{run_stamp}"
    common_config = dict(
        number_of_steps=args.steps,
        snapshot_every=args.snapshot_every,
        metrics_every=args.snapshot_every,
        random_seed=args.seed,
    )

    control_state, control_metrics = run_KMC(
        control_lattice,
        ceox_parameters,
        KMCRunConfig(
            **common_config,
            output_directory=str(root / "no_sonication"),
        ),
    )
    sonicated_state, sonicated_metrics = run_KMC(
        sonicated_lattice,
        ceox_parameters,
        KMCRunConfig(
            **common_config,
            output_directory=str(root / "sonication"),
        ),
        sonication_parameters=sonication_parameters,
    )

    paired_snapshot_names = write_paired_process_trajectories(root)
    write_process_metrics(
        root / "comparison_process_metrics.csv",
        control_metrics,
        sonicated_metrics,
    )
    final_snapshot_name = f"snapshot_{args.steps:08d}.xyz"
    write_paired_snapshot(
        root / "no_sonication" / final_snapshot_name,
        root / "sonication" / final_snapshot_name,
        root / "comparison_final.xyz",
    )
    write_paired_snapshot(
        root / "no_sonication" / final_snapshot_name,
        root / "sonication" / final_snapshot_name,
        root / "comparison_ir_environment.xyz",
        environment_radius_nm=1.0,
    )
    summary_rows = []
    for condition, state, metrics in (
        ("no_sonication", control_state, control_metrics[-1]),
        ("sonication", sonicated_state, sonicated_metrics[-1]),
    ):
        summary_rows.append(
            {
                "condition": condition,
                "step": state.step,
                "KMC_time": metrics["KMC_time"],
                "number_Ce": metrics["number_Ce"],
                "number_O": metrics["number_O"],
                "number_Ir_ion": metrics["number_Ir_ion"],
                "number_Ir": metrics["number_Ir"],
                "embedded_Ir_total": metrics["embedded_Ir_total"],
                "Ir_embedding_fraction": metrics["Ir_embedding_fraction"],
                "mean_Ir_support_contacts": metrics["mean_Ir_support_contacts"],
                "highly_covered_Ir_fraction": metrics[
                    "highly_covered_Ir_fraction"
                ],
                "equivalent_diameter_nm": metrics["equivalent_diameter_nm"],
                "sonication_event_count": metrics["sonication_event_count"],
                "sonication_removed_atoms": metrics["sonication_removed_atoms"],
                "sonication_redeposited_atoms": metrics[
                    "sonication_redeposited_atoms"
                ],
                "solution_chemical_potential_boost_ev": metrics[
                    "solution_chemical_potential_boost_ev"
                ],
            }
        )
    with (root / "comparison_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    metadata = {
        "purpose": "short paired OVITO comparison of catalyst-support formation",
        "arguments": vars(args),
        "ceox_parameters": asdict(ceox_parameters),
        "sonication_parameters": asdict(sonication_parameters),
        "paired_snapshot_names": paired_snapshot_names,
        "condition_labels": {"0": "no_sonication", "1": "sonication"},
        "model_note": (
            "Mean-field growth bursts are a visualization accelerator for "
            "unresolved donor particles and are not a fitted paper parameter."
        ),
    }
    (root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    (root / "OVITO_README.txt").write_text(
        "Open comparison_final.xyz for the final side-by-side view.\n"
        "Open comparison_ir_environment.xyz for the clearest local comparison;\n"
        "it contains only Ir and atoms within 1 nm of Ir.\n"
        "Left: condition=0 (no sonication). Right: condition=1 (sonication).\n"
        "For a paired full-particle animation, open\n"
        "paired_process/snapshot_00000000.xyz as a file sequence.\n"
        "For a paired local Ir-environment animation, open\n"
        "paired_ir_environment/snapshot_00000000.xyz as a file sequence.\n"
        "Each frame uses the same step for both conditions. KMC_time differs\n"
        "because the two event catalogs have different total rates.\n"
        "comparison_process_metrics.csv contains the values at every frame.\n"
        "Select Particle Type Ir, then color by embedded or ir_state.\n"
        "embedded=1 means an Ir site is no longer on the external surface.\n"
        "For early embedding, color Ir by support_contacts; larger values mean\n"
        "more neighboring Ce/O atoms and stronger support coverage.\n"
        "The initial metallic Ir nanoparticles are identical in both conditions.\n"
        "Sonication raises the mean-field Ce/O solution chemical potential from\n"
        "-0.69 eV toward -0.60 eV after interfacial corrosion events.\n"
        "Use a Slice modifier to inspect how CeOx grows around Ir.\n",
        encoding="utf-8",
    )

    print(f"Output directory: {root.resolve()}")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()

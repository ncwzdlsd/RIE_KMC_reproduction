"""Minimal morphology reproduction of Fig. 2I-L from Shi et al. (2025).

This first-stage model isolates the visual mechanism shown in the paper:

1. start from one rough, nearly spherical CeOx particle;
2. nucleate metallic Ir nanoparticles on its surface;
3. grow the CeOx support around the early Ir nanoparticles;
4. retain a mixture of embedded and newly nucleated surface Ir.

The paper does not publish the kinetic parameters or a physical-time mapping
for panels I-L.  Consequently, this script is a stochastic lattice morphology
reconstruction, not a quantitative KMC clock.  Paper-reported dimensions and
explicit reconstruction assumptions are kept separate in run_metadata.json.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np

from constants import SiteType, Species
from generation import find_external_surface, seed_ir_nanoparticle
from lattice_build import Lattice, build_fluorite_lattice


LATTICE_CONSTANT_NM = 0.541
PAPER_INITIAL_DIAMETER_NM = 7.9
PAPER_GROWN_DIAMETER_NM = 10.4
PAPER_INITIAL_DIAMETER_STD_NM = 1.3
PAPER_GROWN_DIAMETER_STD_NM = 1.1


@dataclass(frozen=True)
class Figure2Config:
    box_nm: float = 14.0
    initial_diameter_nm: float = PAPER_INITIAL_DIAMETER_NM
    grown_diameter_nm: float = PAPER_GROWN_DIAMETER_NM
    surface_roughness_nm: float = 0.18
    ir_diameter_nm: float = 1.5
    seed: int = 2026

    def validate(self) -> None:
        if self.initial_diameter_nm <= 0.0:
            raise ValueError("initial diameter must be positive")
        if self.grown_diameter_nm <= self.initial_diameter_nm:
            raise ValueError("grown diameter must exceed initial diameter")
        if self.surface_roughness_nm < 0.0:
            raise ValueError("surface roughness must be non-negative")
        if self.ir_diameter_nm <= 0.0:
            raise ValueError("Ir diameter must be positive")
        required_box_nm = self.grown_diameter_nm + 2.0 * self.ir_diameter_nm
        if self.box_nm < required_box_nm:
            raise ValueError(
                f"box must be at least {required_box_nm:.2f} nm for this geometry"
            )


def normalized(vector: tuple[float, float, float]) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    return value / np.linalg.norm(value)


# The first two particles nucleate before support growth and become embedded.
# Later particles remain partially exposed, as in Fig. 2K-L.
EARLY_IR_DIRECTIONS = (
    normalized((1.0, 0.3, 0.8)),
    normalized((-0.7, 0.8, 0.5)),
)
LATE_IR_DIRECTIONS = (
    normalized((0.4, 1.0, -0.7)),
    normalized((-0.9, -0.3, 0.6)),
    normalized((0.8, -0.7, -0.2)),
    normalized((-0.2, 0.5, -1.0)),
)


def grow_spherical_ceox(
    lattice: Lattice,
    diameter_nm: float,
    roughness_nm: float,
    rng: np.random.Generator,
) -> int:
    """Add a compact fluorite CeOx region with a stochastic outer surface.

    Sites well inside the requested radius are filled deterministically.  In a
    thin outer band, occupancy falls linearly with radius.  Existing Ir sites
    are never overwritten, so increasing the support radius naturally embeds
    Ir nanoparticles that nucleated at an earlier stage.
    """

    radius_nm = 0.5 * diameter_nm
    distances = np.linalg.norm(
        lattice.positions_nm - lattice.center_nm,
        axis=1,
    )
    if roughness_nm == 0.0:
        selected = distances <= radius_nm
    else:
        inner_radius = radius_nm - roughness_nm
        outer_radius = radius_nm + roughness_nm
        probability = np.clip(
            (outer_radius - distances) / (2.0 * roughness_nm),
            0.0,
            1.0,
        )
        selected = rng.random(lattice.nsites) < probability
        selected[distances <= inner_radius] = True

    empty_selected = selected & (lattice.occupation == Species.EMPTY)
    ce_ids = np.flatnonzero(empty_selected & (lattice.site_type == SiteType.M))
    o_ids = np.flatnonzero(empty_selected & (lattice.site_type == SiteType.O))
    lattice.occupation[ce_ids] = Species.CE
    lattice.occupation[o_ids] = Species.O
    return int(len(ce_ids) + len(o_ids))


def add_ir_batch(
    lattice: Lattice,
    support_diameter_nm: float,
    ir_diameter_nm: float,
    directions: tuple[np.ndarray, ...],
) -> int:
    support_radius_nm = 0.5 * support_diameter_nm
    # Put the NP center slightly outside the support surface.  The part that
    # overlaps occupied Ce sites is rejected by seed_ir_nanoparticle().
    # Keep a newly nucleated NP outside the support.  Subsequent radial growth
    # of 1.25 nm then buries roughly half of a 1.5 nm Ir particle, matching the
    # cross-sectional relationship shown in Fig. 2H/L.
    center_radius_nm = support_radius_nm + 0.65 * ir_diameter_nm
    number_added = 0
    for direction in directions:
        ids = seed_ir_nanoparticle(
            lattice,
            center_nm=lattice.center_nm + center_radius_nm * direction,
            diameter_nm=ir_diameter_nm,
        )
        number_added += len(ids)
    return number_added


def calculate_metrics(
    lattice: Lattice,
    stage: str,
    time_min: int,
    nominal_support_diameter_nm: float,
) -> dict[str, float | int | str]:
    occupation = lattice.occupation
    number_ce = int(np.count_nonzero(occupation == Species.CE))
    number_o = int(np.count_nonzero(occupation == Species.O))
    ir_ids = np.flatnonzero(occupation == Species.IR)
    volume_per_ce_nm3 = lattice.lattice_constant_nm**3 / 4.0
    equivalent_diameter_nm = 2.0 * (
        3.0 * number_ce * volume_per_ce_nm3 / (4.0 * math.pi)
    ) ** (1.0 / 3.0)

    if len(ir_ids):
        ir_radii = np.linalg.norm(
            lattice.positions_nm[ir_ids] - lattice.center_nm,
            axis=1,
        )
        burial_depths = np.maximum(
            0.0,
            0.5 * nominal_support_diameter_nm - ir_radii,
        )
        embedding_threshold_nm = 0.25 * lattice.lattice_constant_nm
        embedded_ir = int(
            np.count_nonzero(burial_depths >= embedding_threshold_nm)
        )
        mean_burial_depth_nm = float(np.mean(burial_depths))
        maximum_burial_depth_nm = float(np.max(burial_depths))
    else:
        embedded_ir = 0
        mean_burial_depth_nm = 0.0
        maximum_burial_depth_nm = 0.0

    return {
        "stage": stage,
        "time_min": time_min,
        "nominal_support_diameter_nm": nominal_support_diameter_nm,
        "equivalent_support_diameter_nm": equivalent_diameter_nm,
        "number_Ce": number_ce,
        "number_O": number_o,
        "O_to_Ce": number_o / number_ce,
        "number_Ir": int(len(ir_ids)),
        "embedded_Ir": embedded_ir,
        "Ir_embedding_fraction": embedded_ir / len(ir_ids) if len(ir_ids) else 0.0,
        "mean_radial_burial_depth_nm": mean_burial_depth_nm,
        "maximum_radial_burial_depth_nm": maximum_burial_depth_nm,
    }


def write_figure2_xyz(
    filename: Path,
    lattice: Lattice,
    stage: str,
    time_min: int,
    nominal_support_diameter_nm: float,
) -> None:
    occupied_ids = np.flatnonzero(lattice.occupation != Species.EMPTY)
    surface = find_external_surface(lattice)
    support_radius_nm = 0.5 * nominal_support_diameter_nm
    embedding_threshold_nm = 0.25 * lattice.lattice_constant_nm
    species_names = {
        Species.CE: "Ce",
        Species.O: "O",
        Species.IR: "Ir",
    }

    with filename.open("w", encoding="utf-8", newline="\n") as output:
        output.write(f"{len(occupied_ids)}\n")
        output.write(
            "Properties=species:S:1:pos:R:3:surface:I:1:embedded:I:1:"
            "burial_depth:R:1:support_contacts:I:1 "
            f"figure=2{stage} time_min={time_min}\n"
        )
        for site_id in occupied_ids:
            site_id = int(site_id)
            species = Species(lattice.occupation[site_id])
            position = lattice.positions_nm[site_id]
            burial_depth_nm = 0.0
            embedded = 0
            support_contacts = 0
            if species == Species.IR:
                radius_nm = float(np.linalg.norm(position - lattice.center_nm))
                burial_depth_nm = max(0.0, support_radius_nm - radius_nm)
                embedded = int(burial_depth_nm >= embedding_threshold_nm)
                o_neighbors = lattice.get_ce_o_neighbors(site_id)
                m_neighbors = lattice.get_m_m_neighbors(site_id)
                support_contacts = (
                    int(np.count_nonzero(lattice.occupation[o_neighbors] == Species.O))
                    + int(np.count_nonzero(lattice.occupation[m_neighbors] == Species.CE))
                )
            output.write(
                f"{species_names[species]} "
                f"{position[0]:.6f} {position[1]:.6f} {position[2]:.6f} "
                f"{int(surface[site_id])} {embedded} {burial_depth_nm:.6f} "
                f"{support_contacts}\n"
            )


def save_stage(
    lattice: Lattice,
    output: Path,
    stage: str,
    time_min: int,
    nominal_support_diameter_nm: float,
) -> tuple[Path, dict[str, float | int | str]]:
    filename = output / f"snapshot_{time_min:03d}min.xyz"
    write_figure2_xyz(
        filename,
        lattice,
        stage,
        time_min,
        nominal_support_diameter_nm,
    )
    metrics = calculate_metrics(
        lattice,
        stage,
        time_min,
        nominal_support_diameter_nm,
    )
    return filename, metrics


def write_trajectory(snapshot_paths: list[Path], filename: Path) -> None:
    with filename.open("w", encoding="utf-8", newline="\n") as trajectory:
        for snapshot in snapshot_paths:
            trajectory.write(snapshot.read_text(encoding="utf-8"))


def run(config: Figure2Config, output: Path) -> list[dict[str, float | int | str]]:
    config.validate()
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    ncells = math.ceil(config.box_nm / LATTICE_CONSTANT_NM)
    lattice = build_fluorite_lattice(ncells=ncells)

    snapshots: list[Path] = []
    metrics: list[dict[str, float | int | str]] = []

    grow_spherical_ceox(
        lattice,
        config.initial_diameter_nm,
        config.surface_roughness_nm,
        rng,
    )
    path, row = save_stage(
        lattice, output, "I", 0, config.initial_diameter_nm
    )
    snapshots.append(path)
    metrics.append(row)

    add_ir_batch(
        lattice,
        config.initial_diameter_nm,
        config.ir_diameter_nm,
        EARLY_IR_DIRECTIONS,
    )
    path, row = save_stage(
        lattice, output, "J", 5, config.initial_diameter_nm
    )
    snapshots.append(path)
    metrics.append(row)

    grow_spherical_ceox(
        lattice,
        config.grown_diameter_nm,
        config.surface_roughness_nm,
        rng,
    )
    add_ir_batch(
        lattice,
        config.grown_diameter_nm,
        config.ir_diameter_nm,
        LATE_IR_DIRECTIONS[:2],
    )
    path, row = save_stage(
        lattice, output, "K", 60, config.grown_diameter_nm
    )
    snapshots.append(path)
    metrics.append(row)

    add_ir_batch(
        lattice,
        config.grown_diameter_nm,
        config.ir_diameter_nm,
        LATE_IR_DIRECTIONS[2:],
    )
    path, row = save_stage(
        lattice, output, "L", 180, config.grown_diameter_nm
    )
    snapshots.append(path)
    metrics.append(row)

    with (output / "metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)

    metadata = {
        "target": "Shi et al., Science 387 (2025), Fig. 2I-L",
        "paper_reported": {
            "initial_ceox_diameter_nm": PAPER_INITIAL_DIAMETER_NM,
            "initial_ceox_diameter_std_nm": PAPER_INITIAL_DIAMETER_STD_NM,
            "ceox_diameter_at_60_min_nm": PAPER_GROWN_DIAMETER_NM,
            "ceox_diameter_at_60_min_std_nm": PAPER_GROWN_DIAMETER_STD_NM,
            "panel_times_min": [0, 5, 60, 180],
        },
        "reconstruction_assumptions": {
            **asdict(config),
            "early_ir_particles": len(EARLY_IR_DIRECTIONS),
            "late_ir_particles": len(LATE_IR_DIRECTIONS),
            "growth_model": (
                "stochastic radial filling of fluorite Ce/O lattice sites; "
                "not a calibrated physical-time KMC model"
            ),
        },
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_trajectory(snapshots, output / "figure2_trajectory.xyz")
    (output / "OVITO_README.txt").write_text(
        "Open figure2_trajectory.xyz in OVITO.\n"
        "Color by Particle Type: Ce, O, Ir.\n"
        "The four frames correspond to Fig. 2I-L at 0, 5, 60, and 180 min.\n"
        "Use a Slice modifier in the last frame to inspect embedded Ir.\n"
        "Color Ir by embedded or burial_depth for a quantitative view.\n"
        "The minute labels identify experimental panels; they are not a fitted KMC clock.\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("kmc_output/figure2"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--box-nm", type=float, default=14.0)
    parser.add_argument(
        "--initial-diameter-nm", type=float, default=PAPER_INITIAL_DIAMETER_NM
    )
    parser.add_argument(
        "--grown-diameter-nm", type=float, default=PAPER_GROWN_DIAMETER_NM
    )
    parser.add_argument("--surface-roughness-nm", type=float, default=0.18)
    parser.add_argument("--ir-diameter-nm", type=float, default=1.5)
    args = parser.parse_args()

    config = Figure2Config(
        box_nm=args.box_nm,
        initial_diameter_nm=args.initial_diameter_nm,
        grown_diameter_nm=args.grown_diameter_nm,
        surface_roughness_nm=args.surface_roughness_nm,
        ir_diameter_nm=args.ir_diameter_nm,
        seed=args.seed,
    )
    metrics = run(config, args.output)
    print(f"Output: {args.output.resolve()}")
    for row in metrics:
        print(
            f"Fig. 2{row['stage']}: diameter="
            f"{row['equivalent_support_diameter_nm']:.2f} nm, "
            f"Ir={row['number_Ir']}, "
            f"embedded={row['Ir_embedding_fraction']:.1%}, "
            f"max_depth={row['maximum_radial_burial_depth_nm']:.2f} nm"
        )


if __name__ == "__main__":
    main()

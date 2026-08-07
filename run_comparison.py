"""Run the standard 0-180 min sonication/control KMC comparison.

Run this file directly from an IDE, or use::

    py -3 run_comparison.py

Advanced overrides remain available in ``run_sonication_comparison.py``.
"""

from __future__ import annotations

from pathlib import Path

from kinetic_parameters import KineticParameterSet
from run_sonication_comparison import main


# Standard paper-comparison settings. Edit these values only when a different
# geometry, random realization, or set of observation times is required.
TARGET_TIMES_MIN = "0,5,30,60,120,180"
BOX_NM = 20.0
PARTICLE_DIAMETER_NM = 5.0
RANDOM_SEED = 2025
PROJECT_ROOT = Path(__file__).resolve().parent
PARAMETER_FILE = PROJECT_ROOT / "calibrated_parameters.json"
OUTPUT_DIRECTORY = PROJECT_ROOT / "kmc_output" / "comparison_180min"


def build_arguments() -> list[str]:
    arguments = [
        "--target-times-min",
        TARGET_TIMES_MIN,
        "--box-nm",
        str(BOX_NM),
        "--particle-diameter-nm",
        str(PARTICLE_DIAMETER_NM),
        "--seed",
        str(RANDOM_SEED),
        "--output",
        str(OUTPUT_DIRECTORY),
    ]

    if PARAMETER_FILE.exists():
        arguments.extend(("--parameter-file", str(PARAMETER_FILE)))
        parameters = KineticParameterSet.read(PARAMETER_FILE)
        if not parameters.calibrated:
            print(
                "WARNING: calibrated_parameters.json is not marked as calibrated; "
                "the result is diagnostic, not a quantitative paper result."
            )
            arguments.append("--allow-uncalibrated")
    else:
        print(
            "WARNING: calibrated_parameters.json was not found. Built-in initial "
            "rate estimates will be used; the result is diagnostic, not a "
            "quantitative paper result."
        )
        arguments.append("--allow-uncalibrated")

    return arguments


if __name__ == "__main__":
    main(build_arguments())

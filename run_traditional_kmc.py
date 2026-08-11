"""Direct entry point for the conventional, explicit-hop KMC reproduction.

Run this file directly from an IDE or with ``py -3 run_traditional_kmc.py``.
No command-line options are required.
"""

from pathlib import Path

from run_sonication_comparison import main as run_comparison


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_HINT = PROJECT_ROOT / "kmc_output" / "traditional_kmc_180min"


def main() -> None:
    run_comparison(
        [
            "--method",
            "kmc",
            "--output",
            str(OUTPUT_HINT),
        ]
    )


if __name__ == "__main__":
    main()

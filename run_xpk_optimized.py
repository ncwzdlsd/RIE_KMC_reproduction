"""Direct entry point for the XPK-accelerated KMC reproduction.

Run this file directly from an IDE or with ``py -3 run_xpk_optimized.py``.
No command-line options are required.  XPK sampling controls remain the
documented defaults in ``XPKSamplingParameters`` and are recorded in the run
metadata so convergence can be audited.
"""

from pathlib import Path

from run_sonication_comparison import main as run_comparison


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_HINT = PROJECT_ROOT / "kmc_output" / "xpk_180min"


def main() -> None:
    run_comparison(
        [
            "--method",
            "xpk",
            "--output",
            str(OUTPUT_HINT),
        ]
    )


if __name__ == "__main__":
    main()

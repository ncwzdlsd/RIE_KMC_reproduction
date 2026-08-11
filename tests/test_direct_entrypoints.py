import unittest
from unittest.mock import patch

import run_traditional_kmc
import run_xpk_optimized


class DirectEntrypointTests(unittest.TestCase):
    def test_traditional_entrypoint_selects_kmc_without_user_arguments(self):
        with patch.object(run_traditional_kmc, "run_comparison") as run:
            run_traditional_kmc.main()

        arguments = run.call_args.args[0]
        self.assertEqual(arguments[arguments.index("--method") + 1], "kmc")
        self.assertTrue(
            any("traditional_kmc_180min" in argument for argument in arguments)
        )

    def test_xpk_entrypoint_selects_xpk_without_user_arguments(self):
        with patch.object(run_xpk_optimized, "run_comparison") as run:
            run_xpk_optimized.main()

        arguments = run.call_args.args[0]
        self.assertEqual(arguments[arguments.index("--method") + 1], "xpk")
        self.assertTrue(any("xpk_180min" in argument for argument in arguments))


if __name__ == "__main__":
    unittest.main()

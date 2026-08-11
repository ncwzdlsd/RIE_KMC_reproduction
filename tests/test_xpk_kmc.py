import unittest
from dataclasses import replace

import numpy as np

from constants import SiteType, Species
from generation import build_initial_lattice, find_accessible_empty_sites
from kinetic_parameters import KineticParameterSet
from run_sonication_comparison import fixed_paper_setting_conflicts
from xpk_kmc import XPKLocalKMC, XPKSamplingParameters


class XPKLocalKMCTests(unittest.TestCase):
    def make_mobile_ir_engine(self) -> XPKLocalKMC:
        lattice = build_initial_lattice(
            random_seed=17,
            box_nm=2.2,
            particle_diameter_nm=1.2,
            roughness_fraction=0.0,
        )
        accessible = find_accessible_empty_sites(lattice)
        candidates = np.flatnonzero(
            accessible
            & lattice.reservoir_boundary
            & (lattice.site_type == SiteType.M)
        )
        mobile_site = None
        for candidate in candidates:
            neighbors = lattice.get_m_m_neighbors(int(candidate))
            if np.any(accessible[neighbors]):
                mobile_site = int(candidate)
                break
        self.assertIsNotNone(mobile_site)
        lattice.occupation[mobile_site] = Species.IR_ION

        parameters = KineticParameterSet()
        return XPKLocalKMC(
            lattice,
            parameters.ceox_parameters(),
            parameters.ir_parameters(),
            random_seed=23,
            initial_ir_precursor_atoms=1,
            xpk_sampling=XPKSamplingParameters(
                equilibration_sweeps=1.0,
                samples=4,
                decorrelation_sweeps=1.0,
            ),
        )

    def test_diffusion_ensemble_preserves_composition_and_physical_time(self):
        engine = self.make_mobile_ir_engine()
        initial_time = engine.state.kmc_time
        initial_step = engine.state.step
        initial_ir = int(
            np.count_nonzero(engine.lattice.occupation == Species.IR_ION)
        )

        rates, sample_moves, moves = engine._diffusion_ensemble(0.0)

        self.assertEqual(len(rates), 4)
        self.assertEqual(len(sample_moves), 4)
        self.assertGreater(len(moves), 0)
        self.assertEqual(engine.state.kmc_time, initial_time)
        self.assertEqual(engine.state.step, initial_step)
        self.assertEqual(
            int(np.count_nonzero(engine.lattice.occupation == Species.IR_ION)),
            initial_ir,
        )
        self.assertNotIn("Ir_ion_diffusion", engine.state.event_counts)

        engine._restore_sample(moves, sample_moves[0])
        self.assertEqual(engine.reconcile_accessibility(), 0)
        self.assertEqual(engine.metrics()["Ir_inventory_error_atoms"], 0)

    def test_xpk_step_counts_only_chemical_reactions(self):
        engine = self.make_mobile_ir_engine()

        applied = engine.step_once()

        self.assertTrue(applied)
        self.assertEqual(engine.state.step, 1)
        self.assertGreater(engine.state.kmc_time, 0.0)
        self.assertEqual(engine.xpk_statistics.chemical_space_steps, 1)
        self.assertGreater(engine.xpk_statistics.diffusion_sampling_steps, 0)
        self.assertNotIn("Ir_ion_diffusion", engine.state.event_counts)
        self.assertEqual(engine.metrics()["Ir_inventory_error_atoms"], 0)

    def test_repeated_sampling_does_not_accumulate_bucket_rate_drift(self):
        engine = self.make_mobile_ir_engine()

        for _ in range(100):
            _, sample_moves, moves = engine._diffusion_ensemble(0.0)
            engine._restore_sample(moves, sample_moves[0])

        engine._diffusion_hop()
        exact_diffusion_rate = sum(
            rate * len(members)
            for rate, members in zip(
                engine.diffusion_buckets.rates,
                engine.diffusion_buckets.members,
            )
        )
        self.assertAlmostEqual(
            engine.diffusion_buckets.total_rate,
            exact_diffusion_rate,
            places=10,
        )
        self.assertEqual(engine.reconcile_accessibility(), 0)
        self.assertEqual(engine.metrics()["Ir_inventory_error_atoms"], 0)

    def test_xpk_does_not_modify_fixed_chemical_potentials(self):
        engine = self.make_mobile_ir_engine()
        parameters = KineticParameterSet()

        engine.step_once()

        self.assertEqual(
            engine.ceox_parameters.chemical_potential_ce_ev,
            parameters.chemical_potential_ce_ev,
        )
        self.assertEqual(
            engine.ceox_parameters.chemical_potential_o_ev,
            parameters.chemical_potential_o_ev,
        )

    def test_nonpaper_chemical_potential_is_rejected_not_rewritten(self):
        parameters = replace(
            KineticParameterSet(), chemical_potential_ce_ev=-0.55
        )

        conflicts = fixed_paper_setting_conflicts(parameters)

        self.assertEqual(
            conflicts,
            [("chemical_potential_ce_ev", -0.55, -0.6)],
        )


if __name__ == "__main__":
    unittest.main()

import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ceox_events import CeOxParameters, build_CeOx_event_catalog
from fast_ceox_kmc import FastCeOxKMC, load_checkpoint
from generation import initialize_sphere
from lattice_build import build_fluorite_lattice
from constants import SiteType


class FastCeOxKMCTests(unittest.TestCase):
    def setUp(self):
        self.parameters = CeOxParameters(
            temperature_k=453.0,
            ce_o_binding_energy_ev=0.30,
            chemical_potential_ce_ev=-0.60,
            chemical_potential_o_ev=-0.60,
        )
        self.lattice = build_fluorite_lattice(ncells=5)
        initialize_sphere(
            self.lattice,
            diameter_nm=2.0,
            oxygen_x=2.0,
            rng=np.random.default_rng(7),
        )

    def assert_catalog_matches(self, engine):
        events = build_CeOx_event_catalog(engine.lattice, self.parameters)
        expected = {}
        for event in events:
            key = (event.event_type, event.coordination)
            expected[key] = expected.get(key, 0) + 1
        actual = {
            key: len(engine.buckets.sites[code])
            for code, key in enumerate(engine.buckets.keys)
            if engine.buckets.sites[code]
        }
        self.assertEqual(expected, actual)
        self.assertTrue(
            math.isclose(
                sum(event.rate for event in events),
                engine.total_rate(),
                rel_tol=1e-12,
            )
        )

    def test_vectorized_lattice_neighbor_counts(self):
        metal = self.lattice.site_type == SiteType.M
        oxygen = self.lattice.site_type == SiteType.O
        self.assertEqual(int(self.lattice.ce_o_neighbor_count[metal].max()), 8)
        self.assertEqual(int(self.lattice.ce_o_neighbor_count[oxygen].max()), 4)
        self.assertEqual(int(self.lattice.m_m_neighbor_count[metal].max()), 12)

    def test_local_catalog_matches_full_catalog(self):
        engine = FastCeOxKMC(self.lattice, self.parameters, random_seed=11)
        self.assert_catalog_matches(engine)
        engine.run_to(100, progress_every=0)
        self.assertEqual(engine.exact_connectivity_error_count(), 0)
        self.assert_catalog_matches(engine)

    def test_checkpoint_round_trip(self):
        engine = FastCeOxKMC(self.lattice, self.parameters, random_seed=13)
        engine.run_to(25, progress_every=0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.npz"
            engine.save_checkpoint(checkpoint)
            restored_lattice = build_fluorite_lattice(ncells=5)
            restored = load_checkpoint(checkpoint, restored_lattice, self.parameters)
            self.assertEqual(restored.state.step, engine.state.step)
            self.assertEqual(restored.state.event_counts, engine.state.event_counts)
            np.testing.assert_array_equal(
                restored.lattice.occupation, engine.lattice.occupation
            )
            engine.run_to(50, progress_every=0)
            restored.run_to(50, progress_every=0)
            np.testing.assert_array_equal(
                restored.lattice.occupation, engine.lattice.occupation
            )
            self.assertEqual(restored.state.event_counts, engine.state.event_counts)


if __name__ == "__main__":
    unittest.main()

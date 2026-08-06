import unittest

import numpy as np

from constants import Species
from generation import find_external_surface, initialize_sphere
from lattice_build import build_fluorite_lattice
from sonication_events import (
    SonicationParameters,
    apply_mean_field_ripening_growth,
    apply_sonication_event,
    build_sonication_event,
)


class SonicationEventTests(unittest.TestCase):
    def test_corrosion_only_removes_surface_support_inside_paper_radius(self):
        lattice = build_fluorite_lattice(ncells=7)
        initialize_sphere(
            lattice,
            diameter_nm=3.0,
            oxygen_x=2.0,
            rng=np.random.default_rng(1),
        )
        surface = find_external_surface(lattice)
        parameters = SonicationParameters(
            event_rate=1.0,
            radius_nm=1.0,
            dissolution_probability=1.0,
        )
        event = build_sonication_event(lattice, parameters, surface)
        self.assertIsNotNone(event)
        original_occupation = lattice.occupation.copy()

        removed = apply_sonication_event(
            lattice,
            event,
            parameters,
            np.random.default_rng(2),
        )

        self.assertGreater(len(removed), 0)
        self.assertTrue(np.all(surface[removed]))
        self.assertTrue(
            np.all(
                (original_occupation[removed] == Species.CE)
                | (original_occupation[removed] == Species.O)
            )
        )
        self.assertTrue(np.all(lattice.occupation[removed] == Species.EMPTY))

    def test_mean_field_growth_adds_only_ceox_near_ir(self):
        lattice = build_fluorite_lattice(ncells=7)
        initialize_sphere(
            lattice,
            diameter_nm=2.5,
            oxygen_x=2.0,
            rng=np.random.default_rng(3),
        )
        empty_m_ids = np.flatnonzero(
            (lattice.occupation == Species.EMPTY) & (lattice.site_type == 0)
        )
        support_center = lattice.center_nm
        ir_id = int(
            empty_m_ids[
                np.argmin(
                    np.linalg.norm(
                        lattice.positions_nm[empty_m_ids]
                        - (support_center + np.array([1.4, 0.0, 0.0])),
                        axis=1,
                    )
                )
            ]
        )
        lattice.occupation[ir_id] = Species.IR
        parameters = SonicationParameters(
            event_rate=1.0,
            mean_field_growth_atoms_per_event=12,
            growth_capture_radius_nm=1.0,
        )
        before = lattice.occupation.copy()

        added = apply_mean_field_ripening_growth(
            lattice, parameters, np.random.default_rng(4)
        )

        self.assertGreater(len(added), 0)
        self.assertTrue(np.all(before[added] == Species.EMPTY))
        self.assertTrue(
            np.all(
                (lattice.occupation[added] == Species.CE)
                | (lattice.occupation[added] == Species.O)
            )
        )
        distances = np.linalg.norm(
            lattice.positions_nm[added] - lattice.positions_nm[ir_id], axis=1
        )
        self.assertTrue(np.all(distances <= 1.0))


if __name__ == "__main__":
    unittest.main()

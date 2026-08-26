from __future__ import annotations

from pathlib import Path
import sys
import unittest

import jax.numpy as jp
import numpy as np


MJX_DIR = Path(__file__).resolve().parents[1]
if str(MJX_DIR) not in sys.path:
    sys.path.insert(0, str(MJX_DIR))

from teacher_distillation import (  # noqa: E402
    XY_ACTION_MASK,
    legacy_teacher_observation,
)


class TeacherDistillationTest(unittest.TestCase):
    def test_command5_observation_projects_to_exact_legacy_layout(self) -> None:
        student = jp.arange(2 * 146, dtype=jp.float32).reshape((2, 146))
        teacher = np.asarray(legacy_teacher_observation(student))
        expected = np.concatenate(
            (np.asarray(student[:, :2]), np.asarray(student[:, 5:145])), axis=-1
        )
        self.assertEqual(teacher.shape, (2, 142))
        np.testing.assert_array_equal(teacher, expected)

    def test_v2_teacher_mask_excludes_every_z_action(self) -> None:
        mask = np.asarray(XY_ACTION_MASK).reshape((6, 3))
        np.testing.assert_array_equal(mask[:, :2], np.ones((6, 2)))
        np.testing.assert_array_equal(mask[:, 2], np.zeros(6))

    def test_adapter_rejects_nonstudent_observation(self) -> None:
        with self.assertRaisesRegex(ValueError, "146-D"):
            legacy_teacher_observation(jp.zeros(142))


if __name__ == "__main__":
    unittest.main()

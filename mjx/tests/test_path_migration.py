"""User-run weight layout checks; no simulator or PPO."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from adaptive_path_migration import input_indices


class PathMigrationTest(unittest.TestCase):
    def test_candidate_insertion_and_tail(self):
        ids = input_indices(64)
        self.assertEqual(len(ids), 4498)
        self.assertEqual(len(set(ids)), len(ids))
        np.testing.assert_array_equal(ids[:234], np.arange(234))
        self.assertEqual(ids[234+28], 234+34)
        self.assertEqual(ids[-64], 5334)

    def test_zero_padded_dense_preserves_output(self):
        rng = np.random.default_rng(4)
        for tail in (64, 379):
            ids = input_indices(tail)
            old_x = rng.normal(size=(4434+tail,))
            old_w = rng.normal(size=(4434+tail, 4))
            new_x = rng.normal(size=(5334+tail,))
            new_x[ids] = old_x
            new_w = np.zeros((5334+tail, 4))
            new_w[ids] = old_w
            np.testing.assert_allclose(new_x @ new_w, old_x @ old_w, atol=1e-10)


if __name__ == '__main__':
    unittest.main()

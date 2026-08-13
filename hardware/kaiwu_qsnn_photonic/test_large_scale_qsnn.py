import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from large_scale_qsnn import (
    LAYOUT,
    build_ising,
    decode_solutions,
    encode_inputs,
    make_circle_dataset,
    make_reservoir,
    reservoir_forward,
    train_readout,
)


class LargeScaleQSNTests(unittest.TestCase):
    def test_layout_fills_qboson_1000(self):
        self.assertEqual(LAYOUT.model_spins, 999)
        self.assertEqual(LAYOUT.bias_spin, 999)

    def test_builds_finite_symmetric_1000_matrix(self):
        xy, labels = make_circle_dataset(40, 9)
        encoded = encode_inputs(xy)
        projections = make_reservoir(9)
        features = reservoir_forward(encoded, projections)
        readout = train_readout(np.concatenate((encoded, features), axis=1), labels)
        matrix = build_ising(encoded[0], projections, readout)
        self.assertEqual(matrix.shape, (1000, 1000))
        self.assertTrue(np.isfinite(matrix).all())
        self.assertTrue(np.allclose(matrix, matrix.T))
        self.assertEqual(np.count_nonzero(np.diag(matrix)), 0)

    def test_decode_is_invariant_to_global_spin_flip(self):
        solution = np.ones((1, 1000))
        decoded = decode_solutions(solution)
        flipped = decode_solutions(-solution)
        self.assertEqual(decoded["pred"], flipped["pred"])
        self.assertEqual(decoded["class1_fraction"], flipped["class1_fraction"])


if __name__ == "__main__":
    unittest.main()

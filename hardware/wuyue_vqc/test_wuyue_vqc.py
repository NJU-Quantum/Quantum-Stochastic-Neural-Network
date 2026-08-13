import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wuyue_vqc import (
    CONFIG,
    build_vqc_circuit,
    decode_counts,
    fast_probability,
    make_circle_dataset,
    wuyue_probability,
)


class WuYueVQCTests(unittest.TestCase):
    def setUp(self):
        self.params = np.array((6.0, -0.72, 0.1, 0.0, 0.0, 0.05, 0.0, 0.0))

    def test_dataset_is_deterministic_and_balanced(self):
        xy1, labels1, split1 = make_circle_dataset()
        xy2, labels2, split2 = make_circle_dataset()
        np.testing.assert_allclose(xy1, xy2)
        np.testing.assert_array_equal(labels1, labels2)
        np.testing.assert_array_equal(split1, split2)
        self.assertEqual(xy1.shape, (CONFIG.train_samples + CONFIG.test_samples, 2))
        self.assertEqual(int(labels1.sum()), labels1.size // 2)

    def test_fast_simulator_matches_wuyue_backend(self):
        sample = (0.2, -0.4)
        fast = fast_probability(sample, self.params)
        sdk = wuyue_probability(sample, self.params)
        self.assertAlmostEqual(fast, sdk, places=11)

    def test_circuit_is_native_qasm_and_uses_expected_gates(self):
        qasm = build_vqc_circuit((0.2, -0.4), self.params).QASM()
        self.assertIn("OPENQASM 2.0", qasm)
        self.assertIn("ry(", qasm)
        self.assertIn("rz(", qasm)
        self.assertIn("cx q[2],q[3]", qasm)
        self.assertIn("cx q[3],q[4]", qasm)

    def test_decode_counts_uses_readout_qubit(self):
        result = decode_counts({"000010000": 7, "000000000": 3})
        self.assertAlmostEqual(result["p1"], 0.7)
        self.assertEqual(result["pred"], 1)


if __name__ == "__main__":
    unittest.main()

import unittest

import torch

from qgan.metrics import (
    density_fidelity,
    hellinger_distance,
    physicality_diagnostics,
    purity,
    trace_distance,
    total_variation_distance,
)


class QuantumMetricTests(unittest.TestCase):
    def setUp(self):
        self.zero = torch.diag(torch.tensor([1.0, 0.0], dtype=torch.complex64))
        self.one = torch.diag(torch.tensor([0.0, 1.0], dtype=torch.complex64))
        self.mixed = 0.5 * torch.eye(2, dtype=torch.complex64)

    def test_identical_and_orthogonal_states(self):
        self.assertAlmostEqual(float(density_fidelity(self.zero, self.zero)), 1.0, places=6)
        self.assertAlmostEqual(float(density_fidelity(self.zero, self.one)), 0.0, places=6)
        self.assertAlmostEqual(float(trace_distance(self.zero, self.zero)), 0.0, places=6)
        self.assertAlmostEqual(float(trace_distance(self.zero, self.one)), 1.0, places=6)
        self.assertAlmostEqual(float(hellinger_distance(self.zero, self.one)), 1.0, places=6)
        self.assertAlmostEqual(float(total_variation_distance(self.zero, self.one)), 1.0, places=6)

    def test_purity_and_physicality(self):
        self.assertAlmostEqual(float(purity(self.zero)), 1.0, places=6)
        self.assertAlmostEqual(float(purity(self.mixed)), 0.5, places=6)
        diagnostics = physicality_diagnostics(self.mixed, include_min_eigenvalue=True)
        self.assertLess(float(diagnostics["trace_drift_max"]), 1e-7)
        self.assertLess(float(diagnostics["hermiticity_drift_max"]), 1e-7)
        self.assertAlmostEqual(float(diagnostics["min_eigenvalue"]), 0.5, places=6)


if __name__ == "__main__":
    unittest.main()

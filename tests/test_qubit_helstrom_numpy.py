import math
import unittest

import numpy as np

from tasks.quantum_state_discrimination.numpy_reference import (
    NumpyTrainConfig,
    effective_povm,
    helstrom,
    initial_qsnn_parameters,
    make_ensemble,
    povm_diagnostics,
    qsnn_forward,
    train,
)


class NumpyQubitHelstromTests(unittest.TestCase):
    def test_helstrom_pure_state_formula(self):
        angle = 60.0
        ensemble = make_ensemble(angle, phase_degrees=37.0)
        overlap = math.cos(math.radians(angle) / 2.0)
        expected = 0.5 * (1.0 + math.sqrt(1.0 - overlap**2))
        self.assertAlmostEqual(float(helstrom(ensemble)["success"]), expected, places=10)

    def test_effective_povm_matches_forward(self):
        ensemble = make_ensemble(55.0, phase_degrees=37.0)
        parameters = initial_qsnn_parameters(np.random.default_rng(3), 1.0, 0.99)
        forward = lambda values, states: qsnn_forward(values, states, 1.0, 1.0)
        outputs = forward(parameters, ensemble.states)
        effects = effective_povm(parameters, forward)
        reconstructed = np.asarray(
            [[np.trace(effect @ rho).real for effect in effects] for rho in ensemble.states]
        )
        np.testing.assert_allclose(reconstructed, outputs["probabilities"], atol=1e-10)
        self.assertGreaterEqual(povm_diagnostics(effects)["minimum_effect_eigenvalue"], -1e-10)

    def test_short_training_approaches_helstrom(self):
        ensemble = make_ensemble(60.0, phase_degrees=37.0)
        initial = initial_qsnn_parameters(np.random.default_rng(7), 1.0, 0.995)
        forward = lambda values, states: qsnn_forward(values, states, 1.0, 1.0)
        metrics = train(
            initial,
            ensemble,
            forward,
            NumpyTrainConfig(epochs=180, learning_rate=0.05, log_every=90),
            regularize_qsnn=True,
        )
        self.assertLess(metrics["helstrom_gap"], 0.02)
        self.assertLess(metrics["weighted_leakage"], 0.01)


if __name__ == "__main__":
    unittest.main()

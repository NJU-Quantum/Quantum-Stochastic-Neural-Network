import math
import unittest

import torch

from tasks.quantum_state_discrimination.bounds import (
    helstrom_measurement,
    measurement_success,
)
from tasks.quantum_state_discrimination.experiment import (
    TrainConfig,
    train_discriminator,
)
from tasks.quantum_state_discrimination.models import (
    QubitHelstromQSNN,
    effective_povm,
    povm_diagnostics,
)
from tasks.quantum_state_discrimination.states import (
    make_nonorthogonal_qubit_ensemble,
)


class QubitHelstromTests(unittest.TestCase):
    def test_pure_state_helstrom_formula(self):
        angle = 60.0
        ensemble = make_nonorthogonal_qubit_ensemble(angle, phase_degrees=31.0)
        result = helstrom_measurement(ensemble)
        overlap = math.cos(math.radians(angle) / 2.0)
        expected = 0.5 * (1.0 + math.sqrt(1.0 - overlap * overlap))
        self.assertAlmostEqual(result.success, expected, places=10)
        measured = measurement_success(
            ensemble.rho0,
            ensemble.rho1,
            result.effect0,
            result.effect1,
            ensemble.priors,
        )
        self.assertAlmostEqual(measured, result.success, places=10)

    def test_identical_and_orthogonal_limits(self):
        identical = make_nonorthogonal_qubit_ensemble(0.0)
        orthogonal = make_nonorthogonal_qubit_ensemble(180.0)
        self.assertAlmostEqual(helstrom_measurement(identical).success, 0.5, places=10)
        self.assertAlmostEqual(helstrom_measurement(orthogonal).success, 1.0, places=10)

    def test_qsnn_outputs_are_physical_and_effective_povm_matches(self):
        torch.manual_seed(4)
        ensemble = make_nonorthogonal_qubit_ensemble(
            55.0, phase_degrees=37.0, noise_model="depolarizing", noise_strength=0.1
        )
        model = QubitHelstromQSNN(target_initial_output_mass=0.99)
        outputs = model(ensemble.states)
        self.assertTrue(torch.isfinite(outputs["rho_out"]).all())
        self.assertTrue(torch.all(outputs["probabilities"] >= -1e-10))
        self.assertTrue(torch.all(outputs["output_mass"] <= 1.0 + 1e-10))
        traces = torch.diagonal(outputs["rho_out"], dim1=-2, dim2=-1).sum(dim=-1).real
        self.assertTrue(torch.allclose(traces, torch.ones_like(traces), atol=1e-10))

        effects = effective_povm(model)
        predicted = torch.stack(
            [torch.stack([torch.trace(effect @ rho).real for effect in effects]) for rho in ensemble.states]
        )
        self.assertTrue(torch.allclose(predicted, outputs["probabilities"], atol=1e-9))
        diagnostics = povm_diagnostics(effects)
        self.assertGreaterEqual(diagnostics["effect0_min_eigenvalue"], -1e-9)
        self.assertGreaterEqual(diagnostics["effect1_min_eigenvalue"], -1e-9)
        self.assertGreaterEqual(diagnostics["leakage_effect_min_eigenvalue"], -1e-9)

    def test_short_training_improves_success(self):
        torch.manual_seed(7)
        ensemble = make_nonorthogonal_qubit_ensemble(60.0, phase_degrees=37.0)
        model = QubitHelstromQSNN(target_initial_output_mass=0.995)
        initial = model(ensemble.states)["probabilities"]
        initial_success = 0.5 * float(initial[0, 0] + initial[1, 1])
        metrics = train_discriminator(
            model,
            ensemble,
            TrainConfig(epochs=160, learning_rate=0.05, log_every=80),
        )
        self.assertGreater(metrics["success"], initial_success + 0.02)
        self.assertLess(metrics["helstrom_gap"], 0.02)
        self.assertLess(metrics["weighted_leakage"], 0.01)


if __name__ == "__main__":
    unittest.main()

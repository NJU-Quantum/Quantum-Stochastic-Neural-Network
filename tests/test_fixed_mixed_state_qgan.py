import unittest

import torch

from qgan.generators import PurifiedPQCGenerator
from qgan.metrics import physicality_diagnostics, purity, trainable_parameter_count
from qgan.mixed_state_discriminators import AncillaVQCDiscriminator, LayeredQSNNDiscriminator
from qgan.mixed_states import bell_population, negativity, pauli_correlations, werner_state


class MixedStateGeneratorTests(unittest.TestCase):
    def test_purified_generator_is_physical_and_differentiable(self):
        torch.manual_seed(8)
        generator = PurifiedPQCGenerator(n_layers=2)
        rho = generator()
        diagnostics = physicality_diagnostics(rho, include_min_eigenvalue=True)
        self.assertEqual(rho.shape, (4, 4))
        self.assertLess(float(diagnostics["trace_drift_max"].detach()), 1e-6)
        self.assertLess(float(diagnostics["hermiticity_drift_max"].detach()), 1e-6)
        self.assertGreaterEqual(float(diagnostics["min_eigenvalue"].detach()), -1e-6)
        purity(rho).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in generator.parameters()))

    def test_werner_reference_properties(self):
        rho = werner_state(0.6)
        correlations = pauli_correlations(rho)
        self.assertAlmostEqual(float(purity(rho)), 0.52, places=5)
        self.assertAlmostEqual(float(bell_population(rho)), 0.7, places=5)
        self.assertAlmostEqual(float(negativity(rho)), 0.2, places=5)
        self.assertAlmostEqual(float(correlations["xx"]), 0.6, places=5)
        self.assertAlmostEqual(float(correlations["yy"]), 0.6, places=5)
        self.assertAlmostEqual(float(correlations["zz"]), -0.6, places=5)


class MixedStateDiscriminatorTests(unittest.TestCase):
    def test_parameter_matched_discriminators_and_gradients(self):
        torch.manual_seed(9)
        rho = werner_state(0.6)
        fake = PurifiedPQCGenerator(n_layers=2)()
        qsnn = LayeredQSNNDiscriminator(chebyshev_order=32)
        vqc = AncillaVQCDiscriminator(n_layers=8)
        self.assertEqual(trainable_parameter_count(qsnn), 46)
        self.assertEqual(trainable_parameter_count(vqc), 48)
        for discriminator in (qsnn, vqc):
            real_output = discriminator(rho)
            fake_output = discriminator(fake.detach())
            loss = -(real_output["z_expectation"] - fake_output["z_expectation"])
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in discriminator.parameters()))
            self.assertGreaterEqual(float(real_output["output_mass"].detach()), -1e-6)
            self.assertLessEqual(float(real_output["output_mass"].detach()), 1.0 + 1e-5)


if __name__ == "__main__":
    unittest.main()

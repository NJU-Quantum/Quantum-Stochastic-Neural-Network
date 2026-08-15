import unittest

import torch

from qgan.generators import ConditionalPurifiedPQCGenerator
from qgan.metrics import physicality_diagnostics, trainable_parameter_count
from qgan.mixed_state_discriminators import (
    ConditionalAncillaVQCDiscriminator,
    ConditionalLayeredQSNNDiscriminator,
)
from qgan.mixed_states import werner_state


class ConditionalWernerQGANTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        self.conditions = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])

    def test_batched_werner_states(self):
        states = werner_state(self.conditions)
        self.assertEqual(states.shape, (5, 4, 4))
        traces = torch.diagonal(states, dim1=-2, dim2=-1).sum(-1).real
        self.assertTrue(torch.allclose(traces, torch.ones_like(traces)))

    def test_conditional_generator_is_physical_and_condition_sensitive(self):
        generator = ConditionalPurifiedPQCGenerator(n_layers=2)
        states = generator(self.conditions)
        diagnostics = physicality_diagnostics(states, include_min_eigenvalue=True)
        self.assertEqual(states.shape, (5, 4, 4))
        self.assertLess(float(diagnostics["trace_drift_max"].detach()), 1e-6)
        self.assertGreaterEqual(float(diagnostics["min_eigenvalue"].detach()), -1e-6)
        self.assertGreater(float((states[0] - states[-1]).abs().sum().detach()), 1e-5)
        (states.real.square().sum()).backward()
        self.assertTrue(all(parameter.grad is not None for parameter in generator.parameters()))
        self.assertEqual(trainable_parameter_count(generator), 32)

    def test_parameter_matched_conditional_discriminators(self):
        states = werner_state(self.conditions)
        fake = ConditionalPurifiedPQCGenerator(n_layers=2)(self.conditions).detach()
        qsnn = ConditionalLayeredQSNNDiscriminator(chebyshev_order=24)
        vqc = ConditionalAncillaVQCDiscriminator(n_layers=8)
        self.assertEqual(trainable_parameter_count(qsnn), 92)
        self.assertEqual(trainable_parameter_count(vqc), 96)
        for discriminator in (qsnn, vqc):
            real_output = discriminator(states, self.conditions)
            fake_output = discriminator(fake, self.conditions)
            loss = -(real_output["z_expectation"] - fake_output["z_expectation"]).mean()
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(parameter.grad is not None for parameter in discriminator.parameters()))
            self.assertTrue(torch.all(real_output["output_mass"].detach() <= 1.0 + 1e-5))


if __name__ == "__main__":
    unittest.main()

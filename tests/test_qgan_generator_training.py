import unittest

import torch

from qgan.encoding import probability_amplitude_encode
from qgan.generators import PQCGenerator
from qgan.qsnn_discriminator import QSNNDiscriminator
from qgan.trainer import QGANTrainer


class PQCGeneratorTests(unittest.TestCase):
    def test_generator_is_physical_and_differentiable(self):
        torch.manual_seed(3)
        generator = PQCGenerator(n_qubits=3, n_layers=1)
        noise = torch.randn(4, 3)
        rho = generator(noise)
        trace = torch.diagonal(rho, dim1=-2, dim2=-1).real.sum(-1)
        self.assertEqual(rho.shape, (4, 8, 8))
        self.assertTrue(torch.allclose(rho, rho.mH, atol=1e-6))
        self.assertTrue(torch.allclose(trace, torch.ones_like(trace), atol=1e-6))
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(rho).min().detach()), -1e-6)

        loss = torch.diagonal(rho, dim1=-2, dim2=-1).real[:, 0].mean()
        loss.backward()
        gradients = [parameter.grad for parameter in generator.parameters()]
        self.assertTrue(any(gradient is not None and gradient.abs().sum() > 0 for gradient in gradients))


class TinyAlternatingTrainingTests(unittest.TestCase):
    def test_discriminator_and_generator_steps_update_only_their_side(self):
        torch.manual_seed(17)
        generator = PQCGenerator(n_qubits=3, n_layers=1)
        discriminator = QSNNDiscriminator(
            input_dim=8,
            coherent_time=0.1,
            dissipative_time=0.2,
            backend="cheby_suzuki",
            stage2_steps=2,
            chebyshev_order=24,
        )
        trainer = QGANTrainer(
            generator,
            discriminator,
            torch.optim.Adam(generator.parameters(), lr=1e-2),
            torch.optim.Adam(discriminator.parameters(), lr=1e-2),
        )
        pixels = torch.rand(4, 8)
        _, _, real_rho = probability_amplitude_encode(pixels)
        noise = torch.randn(4, 3)

        generator_before = [parameter.detach().clone() for parameter in generator.parameters()]
        discriminator_before = [parameter.detach().clone() for parameter in discriminator.parameters()]
        d_result = trainer.discriminator_step(real_rho, noise)
        self.assertTrue(torch.isfinite(d_result["loss_d"]))
        self.assertGreater(float(d_result["grad_norm_d"]), 0.0)
        self.assertTrue(
            all(torch.equal(before, after) for before, after in zip(generator_before, generator.parameters()))
        )
        self.assertTrue(
            any(not torch.equal(before, after) for before, after in zip(discriminator_before, discriminator.parameters()))
        )

        discriminator_after_d = [parameter.detach().clone() for parameter in discriminator.parameters()]
        generator_before_g = [parameter.detach().clone() for parameter in generator.parameters()]
        g_result = trainer.generator_step(noise)
        self.assertTrue(torch.isfinite(g_result["loss_g"]))
        self.assertGreater(float(g_result["grad_norm_g"]), 0.0)
        self.assertTrue(
            all(torch.equal(before, after) for before, after in zip(discriminator_after_d, discriminator.parameters()))
        )
        self.assertTrue(
            any(not torch.equal(before, after) for before, after in zip(generator_before_g, generator.parameters()))
        )


if __name__ == "__main__":
    unittest.main()

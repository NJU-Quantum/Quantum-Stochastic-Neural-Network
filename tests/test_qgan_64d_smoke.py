import unittest

import torch

from qgan.encoding import probability_amplitude_encode
from qgan.generators import PQCGenerator
from qgan.qsnn_discriminator import QSNNDiscriminator
from qgan.trainer import QGANTrainer


class QGAN64DSmokeTests(unittest.TestCase):
    def _run_alternating_step(self, device, backend):
        torch.manual_seed(41)
        _, _, real_rho = probability_amplitude_encode(torch.rand(2, 64, device=device))
        generator = PQCGenerator(6, n_layers=1).to(device)
        discriminator = QSNNDiscriminator(
            64,
            coherent_time=0.1,
            dissipative_time=0.2,
            backend=backend,
            stage2_steps=2,
            chebyshev_order=32,
            suzuki_steps=2,
            init_gamma=0.2,
        ).to(device)
        trainer = QGANTrainer(
            generator,
            discriminator,
            torch.optim.Adam(generator.parameters(), lr=1e-3),
            torch.optim.Adam(discriminator.parameters(), lr=1e-3),
            grad_clip=5.0,
        )

        discriminator_result = trainer.discriminator_step(real_rho, torch.randn(2, 6, device=device))
        generator_result = trainer.generator_step(torch.randn(2, 6, device=device))
        rho_out = generator_result["fake_output"]["rho_out"]
        trace = torch.diagonal(rho_out, dim1=-2, dim2=-1).real.sum(-1)

        for value in (
            discriminator_result["loss_d"],
            discriminator_result["grad_norm_d"],
            generator_result["loss_g"],
            generator_result["grad_norm_g"],
        ):
            self.assertTrue(torch.isfinite(value))
        self.assertGreater(float(discriminator_result["grad_norm_d"]), 0.0)
        self.assertGreater(float(generator_result["grad_norm_g"]), 0.0)
        self.assertTrue(torch.allclose(trace, torch.ones_like(trace), atol=3e-5))
        self.assertTrue(torch.allclose(rho_out, rho_out.mH, atol=3e-5))

    def test_cpu_both_composite_backends(self):
        for backend in ("cheby_suzuki", "suzuki_global"):
            with self.subTest(backend=backend):
                self._run_alternating_step(torch.device("cpu"), backend)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_both_composite_backends(self):
        for backend in ("cheby_suzuki", "suzuki_global"):
            with self.subTest(backend=backend):
                self._run_alternating_step(torch.device("cuda"), backend)


if __name__ == "__main__":
    unittest.main()

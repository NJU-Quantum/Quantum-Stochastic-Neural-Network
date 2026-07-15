import tempfile
import unittest
from pathlib import Path

import torch

from qgan.checkpoint import load_checkpoint, save_checkpoint
from qgan.encoding import probability_amplitude_encode
from qgan.generators import PQCGenerator
from qgan.qsnn_discriminator import QSNNDiscriminator
from qgan.vqc_discriminator import VQCDiscriminator


class VQCDiscriminatorTests(unittest.TestCase):
    def test_unitary_baseline_is_physical_and_differentiable(self):
        torch.manual_seed(29)
        _, _, rho = probability_amplitude_encode(torch.rand(3, 8))
        rho.requires_grad_(True)
        model = VQCDiscriminator(
            input_dim=8,
            evolution_time=0.2,
            backend="chebyshev",
            chebyshev_order=32,
        )
        output = model(rho)
        rho_out = output["rho_out"]
        trace = torch.diagonal(rho_out, dim1=-2, dim2=-1).real.sum(-1)
        self.assertTrue(torch.allclose(model.hamiltonian(), model.hamiltonian().mH, atol=1e-7))
        self.assertTrue(torch.allclose(rho_out, rho_out.mH, atol=2e-6))
        self.assertTrue(torch.allclose(trace, torch.ones_like(trace), atol=2e-6))
        self.assertTrue(torch.allclose(output["output_mass"], torch.ones(3), atol=2e-6))
        self.assertTrue(torch.allclose(output["leakage"], torch.zeros(3), atol=2e-6))

        gradients = torch.autograd.grad(-output["z_expectation"].mean(), (rho, model.H_raw))
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(torch.linalg.vector_norm(gradient)), 0.0)

    def test_chebyshev_baseline_matches_exact(self):
        torch.manual_seed(31)
        _, _, rho = probability_amplitude_encode(torch.rand(2, 4))
        exact = VQCDiscriminator(4, evolution_time=0.3, backend="exact")
        approximate = VQCDiscriminator(
            4,
            evolution_time=0.3,
            backend="chebyshev",
            chebyshev_order=40,
            chebyshev_tol=1e-11,
        )
        approximate.load_state_dict(exact.state_dict())
        self.assertTrue(
            torch.allclose(
                exact(rho)["rho_out"],
                approximate(rho)["rho_out"],
                atol=3e-6,
                rtol=3e-6,
            )
        )


class CheckpointTests(unittest.TestCase):
    def test_restore_reproduces_outputs_optimizer_and_next_rng(self):
        torch.manual_seed(37)
        generator = PQCGenerator(3, n_layers=1)
        discriminator = QSNNDiscriminator(
            8,
            coherent_time=0.1,
            dissipative_time=0.2,
            stage2_steps=2,
            chebyshev_order=24,
        )
        optimizer_g = torch.optim.Adam(generator.parameters(), lr=1e-3)
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=2e-3)
        noise = torch.randn(2, 3)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_checkpoint(
                path,
                generator,
                discriminator,
                optimizer_g,
                optimizer_d,
                epoch=2,
                step=17,
                config={"name": "checkpoint_test"},
            )
            expected_output = generator(noise).detach().clone()
            expected_random = torch.randn(5)

            with torch.no_grad():
                for parameter in generator.parameters():
                    parameter.add_(1.0)
                for parameter in discriminator.parameters():
                    parameter.add_(1.0)
            torch.manual_seed(999)

            payload = load_checkpoint(
                path,
                generator,
                discriminator,
                optimizer_g,
                optimizer_d,
            )
            actual_output = generator(noise).detach()
            actual_random = torch.randn(5)

        self.assertEqual(payload["epoch"], 2)
        self.assertEqual(payload["step"], 17)
        self.assertEqual(payload["config"]["name"], "checkpoint_test")
        self.assertTrue(torch.equal(actual_output, expected_output))
        self.assertTrue(torch.equal(actual_random, expected_random))


if __name__ == "__main__":
    unittest.main()

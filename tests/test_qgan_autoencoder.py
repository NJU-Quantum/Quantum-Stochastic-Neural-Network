import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from qgan.autoencoder import (
    ProbabilityAutoencoder,
    load_autoencoder_artifact,
    save_autoencoder_artifact,
)
from qgan.generators import PQCGenerator
from scripts.train_autoencoder import save_reconstruction_grid


class ProbabilityAutoencoderTests(unittest.TestCase):
    def test_single_validation_sample_grid(self):
        model = ProbabilityAutoencoder(latent_dim=8, base_channels=2).eval()
        dataset = TensorDataset(torch.rand(1, 784), torch.zeros(1, dtype=torch.long))
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            save_reconstruction_grid(model, dataset, output_dir, torch.device("cpu"))
            self.assertTrue((output_dir / "reconstruction_grid.png").exists())
            self.assertTrue((output_dir / "reconstruction_samples.pt").exists())

    def test_probability_latent_and_reconstruction_shapes(self):
        torch.manual_seed(5)
        model = ProbabilityAutoencoder(latent_dim=64, base_channels=4)
        images = torch.rand(3, 784)
        reconstruction, latent = model(images)

        self.assertEqual(reconstruction.shape, (3, 1, 28, 28))
        self.assertEqual(latent.shape, (3, 64))
        self.assertTrue(torch.all(latent >= 0))
        self.assertTrue(torch.allclose(latent.sum(dim=-1), torch.ones(3), atol=1e-6))
        self.assertTrue(torch.all((reconstruction >= 0) & (reconstruction <= 1)))

    def test_artifact_round_trip_is_exact(self):
        torch.manual_seed(7)
        model = ProbabilityAutoencoder(latent_dim=64, base_channels=4).eval()
        images = torch.rand(2, 1, 28, 28)
        expected, expected_latent = model(images)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "autoencoder.pt"
            save_autoencoder_artifact(path, model, metadata={"split": "train-only"})
            restored, payload = load_autoencoder_artifact(path, map_location="cpu")
            actual, actual_latent = restored(images)

        self.assertEqual(payload["metadata"]["split"], "train-only")
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(expected_latent, actual_latent))
        self.assertTrue(all(not parameter.requires_grad for parameter in restored.parameters()))


class EnhancedGeneratorTests(unittest.TestCase):
    def test_reuploading_increases_capacity_and_remains_differentiable(self):
        torch.manual_seed(11)
        baseline = PQCGenerator(6, n_layers=2, noise_dim=12)
        enhanced = PQCGenerator(
            6,
            n_layers=4,
            noise_dim=12,
            noise_reuploading=True,
            alternating_entanglement=True,
            canonicalize_output=True,
        )
        baseline_parameters = sum(parameter.numel() for parameter in baseline.parameters())
        enhanced_parameters = sum(parameter.numel() for parameter in enhanced.parameters())
        self.assertGreater(enhanced_parameters, baseline_parameters)

        rho = enhanced(torch.randn(3, 12))
        self.assertEqual(rho.shape, (3, 64, 64))
        trace = torch.diagonal(rho, dim1=-2, dim2=-1).real.sum(dim=-1)
        self.assertTrue(torch.allclose(trace, torch.ones_like(trace), atol=1e-6))
        self.assertEqual(float(rho.imag.abs().max().detach()), 0.0)
        self.assertTrue(torch.all(rho.real >= 0))
        loss = torch.diagonal(rho, dim1=-2, dim2=-1).real[:, :8].sum(dim=-1).mean()
        loss.backward()
        projection_gradients = [
            parameter.grad
            for projection in enhanced.layer_noise_projections
            for parameter in projection.parameters()
        ]
        self.assertTrue(all(gradient is not None for gradient in projection_gradients))
        self.assertTrue(any(float(gradient.norm()) > 0 for gradient in projection_gradients))


if __name__ == "__main__":
    unittest.main()

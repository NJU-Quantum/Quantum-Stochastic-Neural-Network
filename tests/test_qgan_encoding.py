import unittest

import torch

from qgan.encoding import (
    area_downsample,
    embed_binary_label_density,
    pad_density_dimension,
    padding_mass,
    probability_amplitude_encode,
    probability_amplitude_state,
    probabilities_from_density,
)


class ProbabilityEncodingTests(unittest.TestCase):
    def test_statevector_encoding_matches_density_encoding(self):
        pixels = torch.rand(3, 16)
        probabilities, state = probability_amplitude_state(pixels)
        expected_probabilities, expected_state, rho = probability_amplitude_encode(pixels)
        self.assertTrue(torch.equal(probabilities, expected_probabilities))
        self.assertTrue(torch.equal(state, expected_state))
        self.assertTrue(torch.allclose(state @ state.mH, rho))

    def test_probability_amplitude_encoding_is_physical(self):
        pixels = torch.tensor([[0.0, 1.0, 3.0], [2.0, 0.0, 2.0]])
        probabilities, psi, rho = probability_amplitude_encode(pixels)

        self.assertTrue(torch.all(probabilities >= 0))
        self.assertTrue(torch.allclose(probabilities.sum(-1), torch.ones(2), atol=1e-6))
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(psi, dim=1).squeeze(-1), torch.ones(2)))
        self.assertTrue(torch.allclose(rho, rho.mH, atol=1e-6))
        self.assertTrue(
            torch.allclose(
                torch.diagonal(rho, dim1=-2, dim2=-1).real.sum(-1),
                torch.ones(2),
                atol=1e-6,
            )
        )
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(rho).min()), -1e-6)
        self.assertTrue(torch.allclose(probabilities_from_density(rho), probabilities, atol=1e-6))

    def test_area_downsample_shape_and_range(self):
        images = torch.rand(4, 28, 28)
        resized = area_downsample(images, 8)
        self.assertEqual(resized.shape, (4, 8, 8))
        self.assertGreaterEqual(float(resized.min()), 0.0)
        self.assertLessEqual(float(resized.max()), 1.0)

    def test_binary_label_tensor_product_embedding(self):
        pixels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        _, _, rho = probability_amplitude_encode(pixels)
        embedded = embed_binary_label_density(rho, torch.tensor([0, 1]))
        self.assertEqual(embedded.shape, (2, 4, 4))
        self.assertTrue(torch.allclose(embedded[0, :2, :2], rho[0]))
        self.assertTrue(torch.allclose(embedded[1, 2:, 2:], rho[1]))
        self.assertEqual(float(embedded[0, 2:, 2:].abs().max()), 0.0)
        self.assertEqual(float(embedded[1, :2, :2].abs().max()), 0.0)

    def test_lossless_padding_and_padding_mass(self):
        pixels = torch.tensor([[1.0, 2.0, 3.0]])
        probabilities, _, rho = probability_amplitude_encode(pixels)
        padded = pad_density_dimension(rho, 8)
        recovered = probabilities_from_density(padded, feature_dim=3)
        self.assertTrue(torch.allclose(recovered, probabilities, atol=1e-6))
        self.assertTrue(torch.allclose(padding_mass(padded, 3), torch.zeros(1)))

    def test_full_resolution_784_to_1024_is_lossless(self):
        torch.manual_seed(23)
        pixels = torch.rand(2, 784)
        probabilities, _, rho = probability_amplitude_encode(pixels, eps=0.0)
        padded = pad_density_dimension(rho, 1024)
        recovered = probabilities_from_density(padded)
        self.assertTrue(torch.allclose(recovered[:, :784], probabilities, atol=2e-7))
        self.assertEqual(float(recovered[:, 784:].abs().max()), 0.0)
        self.assertEqual(float(padding_mass(padded, 784).max()), 0.0)

    def test_generated_padding_mass_has_a_finite_gradient(self):
        torch.manual_seed(27)
        logits = torch.randn(2, 1024, requires_grad=True)
        probabilities = torch.softmax(logits, dim=-1)
        psi = torch.sqrt(probabilities).to(torch.complex64).unsqueeze(-1)
        rho = psi @ psi.mH
        mass = padding_mass(rho, 784).mean()
        gradient = torch.autograd.grad(mass, logits)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.norm()), 0.0)


if __name__ == "__main__":
    unittest.main()

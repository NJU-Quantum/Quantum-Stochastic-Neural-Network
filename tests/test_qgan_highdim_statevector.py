import unittest

import torch

from qgan.encoding import probability_amplitude_state
from qgan.generators import PQCGenerator
from qgan.metrics import (
    density_fidelity,
    empirical_pure_state_metrics,
    hellinger_distance,
    purity,
    total_variation_distance,
    trace_distance,
)
from qgan.qsnn_discriminator import QSNNDiscriminator
from qgan.trainer import QGANTrainer
from qgan.vqc_discriminator import VQCDiscriminator


class StatevectorPathTests(unittest.TestCase):
    def test_generator_statevector_matches_density_output(self):
        torch.manual_seed(101)
        generator = PQCGenerator(7, n_layers=2, noise_dim=9, noise_reuploading=True)
        noise = torch.randn(2, 9)
        state = generator.statevector(noise)
        rho = generator(noise)
        self.assertEqual(state.shape, (2, 128))
        self.assertTrue(torch.allclose(state.unsqueeze(-1) @ state.unsqueeze(-1).mH, rho))

    def test_qsnn_state_path_matches_density_path(self):
        torch.manual_seed(103)
        _, state = probability_amplitude_state(torch.rand(2, 4))
        state = state[..., 0]
        rho = state.unsqueeze(-1) @ state.unsqueeze(-1).mH
        discriminator = QSNNDiscriminator(
            4,
            coherent_time=0.2,
            dissipative_time=0.3,
            backend="exact_split",
            stage2_steps=3,
            init_h=0.01,
            target_output_mass=0.6,
        )
        state_output = discriminator.forward_state(state)
        density_output = discriminator(rho)
        for key in ("p_real", "p_fake", "output_mass", "leakage", "z_expectation"):
            self.assertTrue(
                torch.allclose(state_output[key], density_output[key], atol=2e-5, rtol=2e-5),
                key,
            )
        self.assertTrue(torch.allclose(state_output["state_trace"], torch.ones(2), atol=2e-5))

    def test_vqc_state_path_matches_density_path(self):
        torch.manual_seed(107)
        _, state = probability_amplitude_state(torch.rand(2, 8))
        state = state[..., 0]
        rho = state.unsqueeze(-1) @ state.unsqueeze(-1).mH
        discriminator = VQCDiscriminator(8, backend="exact", evolution_time=0.2)
        state_output = discriminator.forward_state(state)
        density_output = discriminator(rho)
        for key in ("p_real", "p_fake", "output_mass", "z_expectation"):
            self.assertTrue(torch.allclose(state_output[key], density_output[key], atol=2e-5), key)

    def test_statevector_training_penalizes_padding_and_has_gradients(self):
        torch.manual_seed(109)
        generator = PQCGenerator(3, n_layers=1, noise_dim=4, canonicalize_output=True)
        discriminator = VQCDiscriminator(8, backend="exact", evolution_time=0.1)
        trainer = QGANTrainer(
            generator,
            discriminator,
            torch.optim.Adam(generator.parameters(), lr=1e-3),
            torch.optim.Adam(discriminator.parameters(), lr=1e-3),
            valid_feature_dim=6,
            padding_mass_penalty=1.0,
        )
        _, real_state = probability_amplitude_state(torch.rand(2, 8))
        d_result = trainer.discriminator_step_state(real_state[..., 0], torch.randn(2, 4))
        g_result = trainer.generator_step_state(torch.randn(2, 4))
        self.assertTrue(torch.isfinite(d_result["loss_d"]))
        self.assertTrue(torch.isfinite(g_result["loss_g"]))
        self.assertGreaterEqual(float(g_result["padding_mass_g"]), 0.0)
        self.assertGreater(float(g_result["grad_norm_g"]), 0.0)


class LowRankMetricTests(unittest.TestCase):
    def test_low_rank_metrics_match_dense_reference(self):
        torch.manual_seed(113)
        real = torch.randn(3, 8, dtype=torch.complex128)
        fake = torch.randn(3, 8, dtype=torch.complex128)
        real = real / torch.linalg.vector_norm(real, dim=-1, keepdim=True)
        fake = fake / torch.linalg.vector_norm(fake, dim=-1, keepdim=True)
        rho_real = torch.einsum("bi,bj->bij", real, real.conj()).mean(dim=0)
        rho_fake = torch.einsum("bi,bj->bij", fake, fake.conj()).mean(dim=0)
        actual = empirical_pure_state_metrics(real, fake)
        expected = {
            "fidelity_mean_states": density_fidelity(rho_real, rho_fake),
            "trace_distance_mean_states": trace_distance(rho_real, rho_fake),
            "hellinger_mean_states": hellinger_distance(rho_real, rho_fake),
            "total_variation_mean_states": total_variation_distance(rho_real, rho_fake),
            "purity_real_mean_state": purity(rho_real),
            "purity_fake_mean_state": purity(rho_fake),
        }
        for key, value in expected.items():
            self.assertTrue(torch.allclose(actual[key], value, atol=1e-8, rtol=1e-7), key)


if __name__ == "__main__":
    unittest.main()

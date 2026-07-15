import unittest

import torch

from qgan.objectives import (
    direct_success_value,
    discriminator_loss,
    generator_loss,
    output_statistics,
    partition_output_statistics,
    trace_z_value,
)


def diagonal_density(values):
    return torch.diag_embed(torch.tensor(values, dtype=torch.complex64))


class QGANObjectiveTests(unittest.TestCase):
    def test_trace_z_matches_direct_success_for_complete_output(self):
        real = diagonal_density([[0.0, 0.0, 0.8, 0.2]])
        fake = diagonal_density([[0.0, 0.0, 0.1, 0.9]])
        trace_value = trace_z_value(real, fake, 2, 3)
        direct_value = direct_success_value(real, fake, 2, 3)
        self.assertAlmostEqual(float(trace_value), 0.85, places=6)
        self.assertTrue(torch.allclose(trace_value, direct_value, atol=1e-7))
        self.assertTrue(torch.allclose(discriminator_loss(real, fake, 2, 3), -trace_value))

    def test_leakage_explains_objective_difference(self):
        real = diagonal_density([[0.4, 0.0, 0.5, 0.1]])
        fake = diagonal_density([[0.3, 0.0, 0.1, 0.6]])
        trace_value = trace_z_value(real, fake, 2, 3)
        direct_value = direct_success_value(real, fake, 2, 3)
        real_stats = output_statistics(real, 2, 3)
        fake_stats = output_statistics(fake, 2, 3)
        expected_gap = 0.25 * (
            real_stats["leakage"].mean() + fake_stats["leakage"].mean()
        )
        self.assertTrue(torch.allclose(trace_value - direct_value, expected_gap, atol=1e-7))

    def test_generator_loss_pushes_toward_real(self):
        fake = diagonal_density([[0.0, 0.0, 0.2, 0.8]])
        loss = generator_loss(fake, 2, 3)
        self.assertAlmostEqual(float(loss), 0.6, places=6)

    def test_partition_measurement_is_complete(self):
        rho = diagonal_density(
            [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]
        )
        stats = partition_output_statistics(rho, split_index=2)
        self.assertTrue(torch.allclose(stats["p_real"], torch.tensor([0.3, 0.7])))
        self.assertTrue(torch.allclose(stats["p_fake"], torch.tensor([0.7, 0.3])))
        self.assertTrue(torch.allclose(stats["output_mass"], torch.ones(2)))
        self.assertTrue(torch.allclose(stats["leakage"], torch.zeros(2), atol=1e-7))


if __name__ == "__main__":
    unittest.main()

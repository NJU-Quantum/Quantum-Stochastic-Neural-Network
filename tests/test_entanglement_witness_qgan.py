import unittest

import torch

from qgan.entanglement_witness import (
    SeparableMixtureGenerator,
    calibrated_witness,
    certified_separable_score_bound,
    effective_observable,
    observable_score,
    werner_psi_plus_witness,
)
from qgan.metrics import physicality_diagnostics
from qgan.mixed_state_discriminators import AncillaVQCDiscriminator, LayeredQSNNDiscriminator
from qgan.mixed_states import negativity, werner_state


class EntanglementWitnessTests(unittest.TestCase):
    def test_separable_generator_is_physical_and_ppt(self):
        torch.manual_seed(31)
        generator = SeparableMixtureGenerator(components=8)
        rho = generator()
        diagnostics = physicality_diagnostics(rho, include_min_eigenvalue=True)
        self.assertLess(float(diagnostics["trace_drift_max"].detach()), 1e-10)
        self.assertGreaterEqual(float(diagnostics["min_eigenvalue"].detach()), -1e-10)
        self.assertLess(float(negativity(rho).detach()), 1e-10)
        rho.real.square().sum().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in generator.parameters()))

    def test_werner_witness_and_certified_product_bound(self):
        witness = werner_psi_plus_witness()
        projector = 0.5 * torch.eye(4, dtype=witness.dtype) - witness
        bound = certified_separable_score_bound(
            projector, tolerance=2e-3, max_cells=40_000
        )
        self.assertLessEqual(bound.lower, 0.5 + 1e-10)
        self.assertGreaterEqual(bound.upper, 0.5 - 1e-10)
        self.assertLess(bound.upper - 0.5, 2.1e-3)
        calibrated = calibrated_witness(projector, bound.upper)
        self.assertLess(float(observable_score(calibrated, werner_state(0.6, dtype=witness.dtype))), -0.19)
        self.assertGreaterEqual(
            float(observable_score(calibrated, werner_state(0.2, dtype=witness.dtype))),
            -1e-10,
        )

    def test_effective_observable_reconstructs_linear_discriminator_scores(self):
        torch.manual_seed(32)
        states = torch.stack([werner_state(0.2), werner_state(0.6), werner_state(1.0)])
        for discriminator in (
            LayeredQSNNDiscriminator(chebyshev_order=24),
            AncillaVQCDiscriminator(n_layers=8),
        ):
            observable = effective_observable(discriminator)
            predicted = observable_score(observable, states.to(observable.dtype))
            direct = torch.stack(
                [discriminator(state)["z_expectation"] for state in states]
            )
            self.assertTrue(torch.allclose(predicted, direct, atol=2e-5, rtol=2e-5))


if __name__ == "__main__":
    unittest.main()

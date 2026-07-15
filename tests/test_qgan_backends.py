import unittest

import torch

import qsw
from qgan.encoding import probability_amplitude_encode
from qgan.qsnn_discriminator import QSNNDiscriminator


class CompositeBackendTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)

    def test_density_chebyshev_matches_exact_unitary(self):
        raw = torch.randn(2, 5, 2)
        psi = torch.view_as_complex(raw.contiguous()).unsqueeze(-1)
        psi = psi / torch.linalg.vector_norm(psi, dim=1, keepdim=True)
        rho = psi @ psi.mH
        h_raw = torch.randn(5, 5)
        H = (0.5 * (h_raw + h_raw.T)).to(torch.complex64)
        exact = qsw.evolve_unitary(rho, H, 0.2)
        chebyshev = qsw.evolve_density_chebyshev(rho, H, 0.2, max_order=48, tol=1e-11)
        self.assertLess(float((chebyshev - exact).abs().max()), 2e-6)

    def test_composite_backends_match_exact_split_at_small_dimension(self):
        pixels = torch.rand(2, 4)
        _, _, rho_input = probability_amplitude_encode(pixels)
        rho = torch.nn.functional.pad(rho_input, (0, 2, 0, 2))
        h_raw = torch.randn(4, 4) * 0.1
        H = torch.nn.functional.pad((0.5 * (h_raw + h_raw.T)).to(torch.complex64), (0, 2, 0, 2))
        gamma = torch.rand(2, 4).to(torch.complex64) * 0.2
        reference = qsw.evolve_qsnn2d_stage2_split(rho, H, gamma, 0.2, 4, steps=4)
        chebyshev = qsw.evolve_qsnn2d_cheby_suzuki(
            rho, H, gamma, 0.2, 4, steps=4, chebyshev_order=48, chebyshev_tol=1e-11
        )
        suzuki = qsw.evolve_qsnn2d_suzuki_global(
            rho, H, gamma, 0.2, 4, steps=4, coherent_steps=4, coherent_order=2
        )
        self.assertLess(float((chebyshev - reference).abs().max()), 3e-6)
        self.assertLess(float((suzuki - reference).abs().max()), 2e-4)


class QSNNDiscriminatorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        pixels = torch.rand(3, 4)
        _, _, self.rho = probability_amplitude_encode(pixels)

    def test_physical_parameters_and_forward_outputs(self):
        model = QSNNDiscriminator(
            input_dim=4,
            backend="cheby_suzuki",
            coherent_time=0.2,
            dissipative_time=0.3,
            stage2_steps=3,
            chebyshev_order=32,
        )
        H = model.hamiltonian()
        self.assertTrue(torch.allclose(H, H.mH, atol=1e-7))
        self.assertTrue(torch.all(model.jump_amplitudes().real > 0))

        out = model(self.rho)
        trace = torch.diagonal(out["rho_out"], dim1=-2, dim2=-1).real.sum(-1)
        self.assertTrue(torch.allclose(trace, torch.ones_like(trace), atol=2e-5))
        self.assertTrue(torch.all(out["output_mass"] >= 0))
        self.assertTrue(torch.all(out["output_mass"] <= 1.0 + 2e-5))
        self.assertTrue(torch.all(out["z_expectation"].abs() <= 1.0 + 2e-5))

    def test_gradient_reaches_input_and_parameters_for_both_composites(self):
        for backend in ("cheby_suzuki", "suzuki_global"):
            rho = self.rho.detach().clone().requires_grad_(True)
            model = QSNNDiscriminator(
                input_dim=4,
                backend=backend,
                coherent_time=0.15,
                dissipative_time=0.2,
                stage2_steps=2,
                chebyshev_order=24,
                suzuki_steps=4,
            )
            out = model(rho)
            loss = -out["z_expectation"].mean()
            gradients = torch.autograd.grad(
                loss,
                (rho, model.H_raw, model.gamma_raw, model.total_rate_raw),
            )
            for gradient in gradients:
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(float(torch.linalg.vector_norm(gradient)), 0.0)

    def test_rate_semantics_maps_to_effective_rate(self):
        model = QSNNDiscriminator(
            input_dim=4,
            gamma_semantics="rate",
            init_gamma=0.2,
            target_output_mass=None,
        )
        expected_total_rate = torch.full((4,), 0.4 + model.min_positive)
        self.assertTrue(torch.allclose(model.total_rates(), expected_total_rate, atol=1e-7))
        self.assertTrue(
            torch.allclose(model.effective_rates().sum(dim=0), model.total_rates(), atol=1e-7)
        )

    def test_target_output_mass_controls_initial_leakage(self):
        target = 0.8
        model = QSNNDiscriminator(
            input_dim=4,
            coherent_time=0.0,
            dissipative_time=1.0,
            backend="cheby_suzuki",
            stage2_steps=2,
            target_output_mass=target,
        )
        output = model(self.rho)
        self.assertTrue(
            torch.allclose(
                output["output_mass"],
                torch.full_like(output["output_mass"], target),
                atol=3e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                model.branch_probabilities().sum(dim=0),
                torch.ones(4),
                atol=1e-7,
            )
        )

    def test_batch_matches_single_sample_execution(self):
        model = QSNNDiscriminator(
            input_dim=4,
            backend="cheby_suzuki",
            coherent_time=0.15,
            dissipative_time=0.2,
            stage2_steps=2,
            chebyshev_order=32,
        )
        batched = model(self.rho)["rho_out"]
        singles = torch.stack([model(sample)["rho_out"] for sample in self.rho])
        self.assertTrue(torch.allclose(batched, singles, atol=2e-6, rtol=2e-6))

    def test_autograd_matches_central_finite_difference(self):
        model = QSNNDiscriminator(
            input_dim=4,
            backend="cheby_suzuki",
            coherent_time=0.12,
            dissipative_time=0.18,
            stage2_steps=2,
            chebyshev_order=40,
            real_dtype=torch.float64,
        )
        rho = self.rho[:1].to(torch.complex128)

        def objective():
            return model(rho)["z_expectation"].mean()

        analytical = torch.autograd.grad(objective(), model.gamma_raw)[0][0, 0]
        epsilon = 1e-5
        with torch.no_grad():
            model.gamma_raw[0, 0] += epsilon
        plus = objective().detach()
        with torch.no_grad():
            model.gamma_raw[0, 0] -= 2 * epsilon
        minus = objective().detach()
        with torch.no_grad():
            model.gamma_raw[0, 0] += epsilon
        numerical = (plus - minus) / (2 * epsilon)
        self.assertTrue(torch.allclose(analytical, numerical, atol=2e-7, rtol=2e-3))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cpu_and_cuda_agree_at_small_dimension(self):
        cpu_model = QSNNDiscriminator(
            input_dim=4,
            backend="suzuki_global",
            coherent_time=0.15,
            dissipative_time=0.2,
            stage2_steps=2,
            suzuki_steps=4,
        )
        cuda_model = QSNNDiscriminator(
            input_dim=4,
            backend="suzuki_global",
            coherent_time=0.15,
            dissipative_time=0.2,
            stage2_steps=2,
            suzuki_steps=4,
        ).cuda()
        cuda_model.load_state_dict(cpu_model.state_dict())
        cpu_output = cpu_model(self.rho)["rho_out"]
        cuda_output = cuda_model(self.rho.cuda())["rho_out"].cpu()
        self.assertTrue(torch.allclose(cpu_output, cuda_output, atol=3e-6, rtol=3e-6))


if __name__ == "__main__":
    unittest.main()

import torch

from qgan.generators import ConditionalPurifiedPQCGenerator
from qgan.metrics import physicality_diagnostics, trainable_parameter_count
from qgan.mixed_state_discriminators import (
    ConditionalAncillaVQCDiscriminator,
    ConditionalLayeredQSNNDiscriminator,
)
from qgan.mixed_states import negativity, werner_state
from qgan.rotations import (
    condition_grid,
    local_bloch_vectors,
    normalize_quaternions,
    pauli_tensor,
    quaternion_to_su2,
    random_quaternions,
    rotated_werner_state,
)


DTYPE = torch.float64
CDTYPE = torch.complex128


def test_quaternion_parameterization_is_unitary_and_sign_invariant():
    quaternions = random_quaternions(7, seed=17, dtype=DTYPE)
    unitary = quaternion_to_su2(quaternions)
    identity = torch.eye(2, dtype=CDTYPE).expand(7, -1, -1)
    assert torch.allclose(unitary @ unitary.mH, identity, atol=1e-12, rtol=1e-12)
    assert torch.allclose(
        quaternion_to_su2(quaternions),
        quaternion_to_su2(-quaternions),
        atol=1e-12,
        rtol=1e-12,
    )
    assert bool((normalize_quaternions(quaternions)[:, 0] >= 0).all())


def test_rotated_werner_states_preserve_spectrum_and_entanglement():
    p = torch.tensor([0.2, 0.6, 1.0], dtype=DTYPE)
    quaternions = random_quaternions(3, seed=23, dtype=DTYPE)
    target = rotated_werner_state(p, quaternions, dtype=CDTYPE)
    reference = werner_state(p, dtype=CDTYPE)
    assert torch.allclose(
        torch.linalg.eigvalsh(target), torch.linalg.eigvalsh(reference), atol=1e-12
    )
    assert torch.allclose(negativity(target), negativity(reference), atol=1e-12)
    assert torch.allclose(local_bloch_vectors(target), torch.zeros(3, 6, dtype=DTYPE), atol=1e-12)
    assert pauli_tensor(target).shape == (3, 3, 3)


def test_identity_rotation_and_condition_grid():
    identity_q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=DTYPE)
    p = torch.tensor([0.4, 0.8], dtype=DTYPE)
    assert torch.allclose(
        rotated_werner_state(p, identity_q.expand(2, -1), dtype=CDTYPE),
        werner_state(p, dtype=CDTYPE),
        atol=1e-12,
    )
    grid = condition_grid(p, random_quaternions(3, seed=31, dtype=DTYPE))
    assert grid.shape == (6, 5)
    assert torch.allclose(grid[:3, 0], torch.full((3,), 0.4, dtype=DTYPE))


def test_vector_conditional_generator_is_physical_and_differentiable():
    generator = ConditionalPurifiedPQCGenerator(
        system_qubits=2,
        environment_qubits=2,
        n_layers=3,
        condition_dim=5,
        condition_feature_map="equivariant",
        real_dtype=DTYPE,
    )
    conditions = condition_grid(
        torch.tensor([0.4, 0.8], dtype=DTYPE),
        random_quaternions(2, seed=41, dtype=DTYPE),
    )
    states = generator(conditions)
    diagnostics = physicality_diagnostics(states, include_min_eigenvalue=True)
    assert states.shape == (4, 4, 4)
    assert generator.condition_feature_dim == 1
    assert diagnostics["trace_drift_max"] < 1e-12
    assert diagnostics["hermiticity_drift_max"] < 1e-12
    assert diagnostics["min_eigenvalue"] > -1e-12
    states.real.mean().backward()
    assert generator.ry_condition.grad is not None
    assert torch.isfinite(generator.ry_condition.grad).all()


def test_equivariant_generator_obeys_local_rotation_covariance():
    generator = ConditionalPurifiedPQCGenerator(
        system_qubits=2,
        environment_qubits=2,
        n_layers=3,
        condition_dim=5,
        condition_feature_map="equivariant",
        real_dtype=DTYPE,
    )
    quaternions = random_quaternions(4, seed=47, dtype=DTYPE)
    conditions = condition_grid(torch.tensor([0.7], dtype=DTYPE), quaternions)
    identity = torch.tensor([[0.7, 1.0, 0.0, 0.0, 0.0]], dtype=DTYPE)
    from qgan.rotations import rotate_second_qubit

    expected = rotate_second_qubit(generator(identity).expand(4, -1, -1), quaternions)
    assert torch.allclose(generator(conditions), expected, atol=1e-11, rtol=1e-11)


def test_vector_conditional_discriminators_are_parameter_matched_and_differentiable():
    conditions = condition_grid(
        torch.tensor([0.45], dtype=DTYPE),
        random_quaternions(2, seed=51, dtype=DTYPE),
    )
    states = rotated_werner_state(conditions[:, 0], conditions[:, 1:], dtype=CDTYPE)
    qsnn = ConditionalLayeredQSNNDiscriminator(
        input_dim=4,
        hidden_dim=4,
        condition_dim=5,
        target_layer_mass=0.9,
        real_dtype=DTYPE,
    )
    vqc = ConditionalAncillaVQCDiscriminator(
        system_qubits=2, n_layers=8, condition_dim=5, real_dtype=DTYPE
    )
    assert trainable_parameter_count(qsnn) == 276
    assert trainable_parameter_count(vqc) == 288
    assert abs(trainable_parameter_count(qsnn) - trainable_parameter_count(vqc)) / 276 < 0.05
    for discriminator in (qsnn, vqc):
        output = discriminator(states, conditions)
        assert output["z_expectation"].shape == (2,)
        output["z_expectation"].mean().backward()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in discriminator.parameters()
        )

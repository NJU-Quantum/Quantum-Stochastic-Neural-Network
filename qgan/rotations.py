"""Local SU(2) rotations and observables for rotated Werner-state benchmarks."""

from __future__ import annotations

import math

import torch

from .mixed_states import werner_state


def normalize_quaternions(quaternions: torch.Tensor) -> torch.Tensor:
    """Normalize unit quaternions and choose the canonical representative q0 >= 0."""
    values = torch.as_tensor(quaternions)
    if values.shape[-1] != 4:
        raise ValueError("quaternions must end in dimension 4")
    norms = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    if bool((norms <= 1e-12).any()):
        raise ValueError("zero quaternion is not a valid rotation")
    normalized = values / norms
    signs = torch.where(
        normalized[..., :1] < 0,
        -torch.ones_like(normalized[..., :1]),
        torch.ones_like(normalized[..., :1]),
    )
    return normalized * signs


def random_quaternions(
    count: int,
    seed: int,
    *,
    dtype: torch.dtype = torch.float64,
    device=None,
) -> torch.Tensor:
    """Draw deterministic Haar-uniform SU(2) rotations via normalized Gaussians."""
    if count <= 0:
        raise ValueError("count must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    samples = torch.randn(count, 4, generator=generator, dtype=dtype)
    return normalize_quaternions(samples).to(device=device)


def stress_quaternions(
    *, dtype: torch.dtype = torch.float64, device=None
) -> torch.Tensor:
    """Identity, half turns, and quarter turns about canonical/diagonal axes."""
    root_half = math.sqrt(0.5)
    values = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [root_half, root_half, 0.0, 0.0],
            [root_half, 0.0, root_half, 0.0],
            [root_half, 0.0, 0.0, root_half],
            [root_half, -root_half, 0.0, 0.0],
            [root_half, 0.0, -root_half, 0.0],
            [root_half, 0.0, 0.0, -root_half],
            [0.5, 0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5, -0.5],
        ],
        dtype=dtype,
        device=device,
    )
    return normalize_quaternions(values)


def quaternion_to_su2(quaternions: torch.Tensor) -> torch.Tensor:
    """Map q=(w,x,y,z) to U=wI-i(xX+yY+zZ)."""
    q = normalize_quaternions(quaternions)
    complex_dtype = torch.complex128 if q.dtype == torch.float64 else torch.complex64
    w, x, y, z = (component.to(complex_dtype) for component in q.unbind(dim=-1))
    row0 = torch.stack([w - 1j * z, -y - 1j * x], dim=-1)
    row1 = torch.stack([y - 1j * x, w + 1j * z], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def local_rotation_operator(quaternions: torch.Tensor) -> torch.Tensor:
    """Return I tensor U(q), with leading quaternion batch dimensions retained."""
    unitary = quaternion_to_su2(quaternions)
    identity = torch.eye(2, dtype=unitary.dtype, device=unitary.device)
    return torch.einsum("ij,...kl->...ikjl", identity, unitary).reshape(
        *unitary.shape[:-2], 4, 4
    )


def rotate_second_qubit(rho: torch.Tensor, quaternions: torch.Tensor) -> torch.Tensor:
    """Apply I tensor U(q) to a two-qubit density matrix or batch."""
    if rho.shape[-2:] != (4, 4):
        raise ValueError("rho must be a two-qubit density matrix")
    operator = local_rotation_operator(quaternions).to(device=rho.device, dtype=rho.dtype)
    return operator @ rho @ operator.mH


def rotated_werner_state(
    p: float | torch.Tensor,
    quaternions: torch.Tensor,
    *,
    dtype: torch.dtype = torch.complex64,
    device=None,
) -> torch.Tensor:
    """Construct (I tensor U) rho_W(p) (I tensor U dagger)."""
    q = torch.as_tensor(quaternions, device=device)
    base = werner_state(p, dtype=dtype, device=device)
    return rotate_second_qubit(base, q)


def condition_grid(p_values: torch.Tensor, quaternions: torch.Tensor) -> torch.Tensor:
    """Cartesian product of p values and quaternion rotations as five features."""
    ps = torch.as_tensor(p_values, dtype=quaternions.dtype, device=quaternions.device).reshape(-1)
    qs = normalize_quaternions(quaternions)
    p_column = ps[:, None, None].expand(-1, qs.shape[0], 1)
    q_rows = qs[None, :, :].expand(ps.shape[0], -1, -1)
    return torch.cat([p_column, q_rows], dim=-1).reshape(-1, 5)


def pauli_tensor(rho: torch.Tensor) -> torch.Tensor:
    """Return the full 3x3 correlation tensor Tr[rho sigma_i tensor sigma_j]."""
    real_dtype = torch.float64 if rho.dtype == torch.complex128 else torch.float32
    zero = torch.tensor(0.0, dtype=real_dtype, device=rho.device)
    one = torch.tensor(1.0, dtype=real_dtype, device=rho.device)
    x = torch.stack([torch.stack([zero, one]), torch.stack([one, zero])]).to(rho.dtype)
    y = torch.stack(
        [
            torch.stack([zero.to(rho.dtype), -1j * one.to(rho.dtype)]),
            torch.stack([1j * one.to(rho.dtype), zero.to(rho.dtype)]),
        ]
    )
    z = torch.diag(torch.stack([one, -one])).to(rho.dtype)
    paulis = (x, y, z)
    rows = []
    for left in paulis:
        entries = []
        for right in paulis:
            observable = torch.kron(left.contiguous(), right.contiguous())
            entries.append(torch.einsum("...ij,ji->...", rho, observable).real)
        rows.append(torch.stack(entries, dim=-1))
    return torch.stack(rows, dim=-2)


def local_bloch_vectors(rho: torch.Tensor) -> torch.Tensor:
    """Return the concatenated three-component Bloch vectors of both qubits."""
    real_dtype = torch.float64 if rho.dtype == torch.complex128 else torch.float32
    zero = torch.tensor(0.0, dtype=real_dtype, device=rho.device)
    one = torch.tensor(1.0, dtype=real_dtype, device=rho.device)
    x = torch.stack([torch.stack([zero, one]), torch.stack([one, zero])]).to(rho.dtype)
    y = torch.stack(
        [
            torch.stack([zero.to(rho.dtype), -1j * one.to(rho.dtype)]),
            torch.stack([1j * one.to(rho.dtype), zero.to(rho.dtype)]),
        ]
    )
    z = torch.diag(torch.stack([one, -one])).to(rho.dtype)
    identity = torch.eye(2, dtype=rho.dtype, device=rho.device)
    values = []
    for pauli in (x, y, z):
        observable = torch.kron(pauli.contiguous(), identity.contiguous())
        values.append(torch.einsum("...ij,ji->...", rho, observable).real)
    for pauli in (x, y, z):
        observable = torch.kron(identity.contiguous(), pauli.contiguous())
        values.append(torch.einsum("...ij,ji->...", rho, observable).real)
    return torch.stack(values, dim=-1)

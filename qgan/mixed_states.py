"""Reference states and observables for small mixed-state QGAN benchmarks."""

import torch


def bell_psi_plus(dtype: torch.dtype = torch.complex64, device=None) -> torch.Tensor:
    state = torch.zeros(4, dtype=dtype, device=device)
    state[1] = 2.0**-0.5
    state[2] = 2.0**-0.5
    return state


def werner_state(
    p: float | torch.Tensor,
    dtype: torch.dtype = torch.complex64,
    device=None,
) -> torch.Tensor:
    """Two-qubit Werner state based on |Psi+>."""
    p_tensor = torch.as_tensor(p, dtype=torch.float64 if dtype == torch.complex128 else torch.float32, device=device)
    if not bool(((p_tensor >= 0) & (p_tensor <= 1)).all()):
        raise ValueError("Werner parameter p must lie in [0, 1]")
    psi = bell_psi_plus(dtype=dtype, device=device)
    bell_density = psi[:, None] @ psi.conj()[None, :]
    identity = torch.eye(4, dtype=dtype, device=device) / 4.0
    weight = p_tensor.to(dtype)
    if weight.dim() > 0:
        weight = weight[..., None, None]
    return weight * bell_density + (1.0 - weight) * identity


def partial_transpose_two_qubit(rho: torch.Tensor, subsystem: int = 1) -> torch.Tensor:
    if rho.shape[-2:] != (4, 4):
        raise ValueError("rho must be a two-qubit density matrix")
    reshaped = rho.reshape(*rho.shape[:-2], 2, 2, 2, 2)
    if subsystem == 0:
        return reshaped.transpose(-4, -2).reshape(*rho.shape[:-2], 4, 4)
    if subsystem == 1:
        return reshaped.transpose(-3, -1).reshape(*rho.shape[:-2], 4, 4)
    raise ValueError("subsystem must be 0 or 1")


def negativity(rho: torch.Tensor) -> torch.Tensor:
    eigenvalues = torch.linalg.eigvalsh(partial_transpose_two_qubit(rho))
    return eigenvalues.clamp_max(0).abs().sum(dim=-1).real


def bell_population(rho: torch.Tensor) -> torch.Tensor:
    psi = bell_psi_plus(dtype=rho.dtype, device=rho.device)
    return torch.einsum("i,...ij,j->...", psi.conj(), rho, psi).real


def pauli_correlations(rho: torch.Tensor) -> dict[str, torch.Tensor]:
    real_dtype = torch.float64 if rho.dtype == torch.complex128 else torch.float32
    zero = torch.tensor(0.0, device=rho.device, dtype=real_dtype)
    one = torch.tensor(1.0, device=rho.device, dtype=real_dtype)
    x = torch.stack([torch.stack([zero, one]), torch.stack([one, zero])]).to(rho.dtype)
    y = torch.stack(
        [
            torch.stack([zero.to(rho.dtype), -1j * one.to(rho.dtype)]),
            torch.stack([1j * one.to(rho.dtype), zero.to(rho.dtype)]),
        ]
    )
    z = torch.diag(torch.stack([one, -one])).to(rho.dtype)
    result = {}
    for name, pauli in (("xx", x), ("yy", y), ("zz", z)):
        observable = torch.kron(pauli.contiguous(), pauli.contiguous())
        result[name] = torch.diagonal(rho @ observable, dim1=-2, dim2=-1).sum(dim=-1).real
    return result

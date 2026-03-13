# models.py
import torch
import torch.nn as nn
import qsw # liouvillian, lindblad_rhs, liouvillian_mv, evolve_expm, evolve_vec_rk4, evolve_from_operators

# 默认使用自适应演化器：小规模用 expm，大规模自动切换 Krylov
# 如需固定算法，可在脚本中临时覆盖 models.evolve。
evolve = qsw.evolve_auto


class QSNNFunction(nn.Module):
    def __init__(self, N_in=10, T=1.0, init_scale=0.05, device="cuda"):
        super().__init__()
        self.N_in = N_in
        self.N = N_in + 1  # last neuron is output
        self.T = T
        self.device = device

        self.H_raw = nn.Parameter(init_scale * torch.randn(self.N, self.N, device=device, dtype=torch.float32))

    def encode(self, x):
        """
        Eq.(7) with n=1: |psi> ∝ Σ_{i=0}^{N_in-1} x^i |i>
        """
        x = x.to(self.device)
        x = x.reshape(-1)  # (B,)
        B = x.shape[0]

        exponents = torch.arange(self.N_in, device=self.device, dtype=x.dtype)
        powers = x.unsqueeze(-1) ** exponents.unsqueeze(0)  # (B, N_in)

        psi = torch.zeros((B, self.N, 1), device=self.device, dtype=torch.complex64)
        psi[:, :self.N_in, 0] = powers.to(torch.complex64)
        psi = psi / torch.linalg.norm(psi, dim=1, keepdim=True).clamp_min(1e-12)
        rho = psi @ psi.mH  # (B, N, N)
        return rho

    def forward(self, x):
        Hf = self.H_raw.to(torch.complex64)
        H = 0.5 * (Hf + Hf.mH)  # Hermitian

        rho_in = self.encode(x)
        rho_out = evolve(rho_in, H, [], self.T)

        yhat = rho_out[:, self.N-1, self.N-1].real.clamp(0.0, 1.0)
        if yhat.numel() == 1:
            return yhat[0], rho_out[0]
        return yhat, rho_out
    

def basis(N, i, device):
    v = torch.zeros((N,1), device=device, dtype=torch.complex64)
    v[i,0] = 1
    return v

class QSNN2D(nn.Module):
    def __init__(self, N_in=12, T_u=1.0, T_d=1.0, init_h=0.1, init_g=0.1, device="cuda"):
        super().__init__()
        self.N_in = N_in
        self.N = N_in + 2
        self.T_u, self.T_d = T_u, T_d
        self.device = device

        # input-layer Hamiltonian params
        self.Hu_raw = nn.Parameter(init_h * torch.randn(N_in, N_in, device=device, dtype=torch.float32))
        # gammas: 2 outputs x N_in inputs
        self.gamma = nn.Parameter(init_g * torch.randn(2, N_in, device=device, dtype=torch.float32))

    def encode(self, x, y):
        # Eq.(7) with n=2, choose K=N_in/2
        K = self.N_in // 2
        assert 2*K == self.N_in

        x = x.to(self.device).reshape(-1)
        y = y.to(self.device).reshape(-1)
        B = x.shape[0]

        exponents = torch.arange(K, device=self.device, dtype=x.dtype)
        px = x.unsqueeze(-1) ** exponents.unsqueeze(0)
        py = y.unsqueeze(-1) ** exponents.unsqueeze(0)

        psi = torch.zeros((B, self.N, 1), device=self.device, dtype=torch.complex64)
        psi[:, :K, 0] = px.to(torch.complex64)
        psi[:, K:2*K, 0] = py.to(torch.complex64)
        psi = psi / torch.linalg.norm(psi, dim=1, keepdim=True).clamp_min(1e-12)
        return psi @ psi.mH

    def forward(self, xy):
        if isinstance(xy, (list, tuple)):
            x, y = xy[0], xy[1]
        else:
            x, y = xy[..., 0], xy[..., 1]
        N, N_in = self.N, self.N_in

        # Stage 1: unitary on input block
        Hu = self.Hu_raw.to(torch.complex64)
        Hu = 0.5 * (Hu + Hu.mH)
        H = torch.zeros((N,N), device=self.device, dtype=torch.complex64)
        H[:N_in,:N_in] = Hu

        rho0 = self.encode(x, y)
        rho_u = evolve(rho0, H, [], self.T_u)

        # Stage 2: dissipative input -> output
        out0, out1 = N_in, N_in + 1
        Ls = []
        for j in range(N_in):
            braj = basis(N, j, self.device).mH
            Ls.append(self.gamma[0,j].to(torch.complex64) * (basis(N,out0,self.device) @ braj))
            Ls.append(self.gamma[1,j].to(torch.complex64) * (basis(N,out1,self.device) @ braj))

        rho_out = evolve(rho_u, H, Ls, self.T_d)

        p0 = rho_out[:, out0, out0].real
        p1 = rho_out[:, out1, out1].real
        probs = torch.stack([p0, p1], dim=-1).clamp(1e-6, 1.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)  # normalize
        if probs.shape[0] == 1:
            return probs[0], rho_out[0]
        return probs, rho_out
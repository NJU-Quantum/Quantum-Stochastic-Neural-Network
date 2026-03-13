import torch
import torch.nn as nn

# ---------- linear algebra helpers ----------
def kron(A, B): return torch.kron(A, B)

def vec(rho):
    # column-stacking
    return rho.reshape(-1, 1)

def unvec(v, N):
    return v.reshape(N, N)

def basis(N, i, device=None, dtype=None):
    v = torch.zeros((N, 1), device=device, dtype=dtype)
    v[i, 0] = 1
    return v

def proj(N, i, device=None, dtype=None):
    v = basis(N, i, device=device, dtype=dtype)
    return v @ v.mH  # |i><i|

# ---------- Liouvillian builder (paper Eq.(6)) ----------
def liouvillian(H, Ls):
    """
    H: (N,N) complex
    Ls: list of (N,N) complex
    returns: (N^2,N^2) complex
    """
    N = H.shape[0]
    device, dtype = H.device, H.dtype
    I = torch.eye(N, device=device, dtype=dtype)

    # -i(H⊗I - I⊗H^T)
    L = (-1j) * (kron(H, I) - kron(I, H.T))

    for Lk in Ls:
        LdagL = Lk.mH @ Lk
        term = kron(Lk, Lk.conj()) - 0.5 * kron(LdagL, I) - 0.5 * kron(I, (Lk.T @ Lk.conj()))
        L = L + term
    return L

def evolve(rho0, L, T):
    U = torch.matrix_exp(L * T)
    vT = U @ vec(rho0)
    return unvec(vT, rho0.shape[0])

# ---------- Function approximation model ----------
class QSNNFunction(nn.Module):
    def __init__(self, N_in, T=1.0, init_scale=0.05, device="cuda"):
        super().__init__()
        self.N_in = N_in
        self.N = N_in + 1
        self.T = T
        self.device = device

        # Train full symmetric input+hidden+output Hamiltonian (simple first)
        # Parameterize as an unconstrained matrix then symmetrize
        self.H_raw = nn.Parameter(init_scale * torch.randn(self.N, self.N, device=device, dtype=torch.float32))

    def encode(self, x):
        """
        Eq.(7) for n=1: |psi> ∝ Σ_{i=0}^{K-1} x^i |i>
        Here K = N_in
        """
        x = x.to(self.device)
        powers = torch.stack([x**i for i in range(self.N_in)], dim=0)  # (N_in,)
        psi = torch.zeros((self.N, 1), device=self.device, dtype=torch.complex64)
        psi[:self.N_in, 0] = powers.to(torch.complex64)
        psi = psi / torch.linalg.norm(psi)
        rho = psi @ psi.mH
        return rho

    def forward(self, x):
        Hf = self.H_raw.to(torch.complex64)
        H = 0.5 * (Hf + Hf.mH)  # Hermitian

        L = liouvillian(H, [])
        rho_in = self.encode(x)
        rho_out = evolve(rho_in, L, self.T)
        yhat = rho_out[self.N-1, self.N-1].real  # Tr(|out><out| rho)
        return yhat, rho_out

# ---------- 2D classification model ----------
class QSNN2D(nn.Module):
    def __init__(self, N_in=12, T_u=1.0, T_d=1.0, init_h=0.1, init_g=0.1, device="cuda"):
        super().__init__()
        self.N_in = N_in
        self.N = N_in + 2  # last two are outputs
        self.T_u, self.T_d = T_u, T_d
        self.device = device

        # Hamiltonian only among input neurons (N_in x N_in)
        self.Hu_raw = nn.Parameter(init_h * torch.randn(N_in, N_in, device=device, dtype=torch.float32))

        # gamma from each input neuron -> each output neuron (2 x N_in)
        self.gamma = nn.Parameter(init_g * torch.randn(2, N_in, device=device, dtype=torch.float32))

    def encode(self, x, y, K=1):
        """
        Use paper Eq.(7) with n=2 (x,y). Simplify: choose K = N_in//2 (must be integer).
        Eq.(7): |psi> ∝ Σ_{i=0}^{K-1} Σ_{d=0}^{1} (x_d)^i |dK+i|
        Here x_0=x, x_1=y.
        """
        N_in = self.N_in
        K = N_in // 2
        assert 2 * K == N_in

        psi = torch.zeros((self.N, 1), device=self.device, dtype=torch.complex64)
        xs = torch.stack([x, y]).to(self.device)
        for d in range(2):
            for i in range(K):
                psi[d*K + i, 0] = (xs[d] ** i).to(torch.complex64)
        psi = psi / torch.linalg.norm(psi)
        return psi @ psi.mH

    def forward(self, x, y):
        N, N_in = self.N, self.N_in

        # Stage 1: unitary on input subspace
        Hu_f = self.Hu_raw.to(torch.complex64)
        Hu = 0.5 * (Hu_f + Hu_f.mH)
        H = torch.zeros((N, N), device=self.device, dtype=torch.complex64)
        H[:N_in, :N_in] = Hu
        L_u = liouvillian(H, [])

        rho0 = self.encode(x, y)
        rho_u = evolve(rho0, L_u, self.T_u)

        # Stage 2: dissipative input->output
        out0, out1 = N_in, N_in + 1
        Ls = []
        g = self.gamma  # (2,N_in), real
        for j in range(N_in):
            # L_out0,j and L_out1,j
            ket0 = basis(N, out0, device=self.device, dtype=torch.complex64)
            ket1 = basis(N, out1, device=self.device, dtype=torch.complex64)
            braj = basis(N, j, device=self.device, dtype=torch.complex64).mH
            Ls.append((g[0, j].to(torch.complex64)) * (ket0 @ braj))
            Ls.append((g[1, j].to(torch.complex64)) * (ket1 @ braj))

        L_d = liouvillian(torch.zeros((N, N), device=self.device, dtype=torch.complex64), Ls)
        rho_out = evolve(rho_u, L_d, self.T_d)

        p0 = rho_out[out0, out0].real
        p1 = rho_out[out1, out1].real
        return torch.stack([p0, p1]), rho_out
# qsw.py
import torch

def kron(A, B): 
    return torch.kron(A.contiguous(), B.contiguous())

def vec(rho): # rho is (N,N)
    # 修正：使用Fortran order (按列) flatten，这是标准的vec操作
    return rho.T.flatten().unsqueeze(-1)

def unvec(v, N): # v is (N^2,1)
    # 对应 vec 将 rho 转置后按列拉直，这里恢复原始形状并转置回来
    vv = v.squeeze()
    return vv.view(N, N).T

def liouvillian(H, Ls): # 显式构造Liouvillian超算符，适用于小系统（N不大）, 低效
    """
    H: (N,N) complex
    Ls: list of (N,N) complex Lindblad operators
    returns L: (N^2,N^2) complex
    """
    N = H.shape[0]
    device, dtype = H.device, H.dtype
    I = torch.eye(N, device=device, dtype=dtype)

    # -i(H⊗I - I⊗H^T)
    Ht = H.T.contiguous()
    L = (-1j) * (torch.kron(H.contiguous(), I) - torch.kron(I, Ht))

    for Lk in Ls:
        Lk  = Lk.contiguous()
        Lkc = Lk.conj().contiguous()
        Lkt = Lk.T.contiguous()

        LdagL = (Lk.mH @ Lk).contiguous()
        right = (Lkt @ Lkc).contiguous()

        L = L + (torch.kron(Lk, Lkc)
                 - 0.5 * torch.kron(LdagL, I)
                 - 0.5 * torch.kron(I, right))
    return L

def lindblad_rhs(rho, H, Ls): # 计算Liouvillian演化的右侧，即drho/dt
    """
    rho: (N,N) complex
    H: (N,N) complex Hermitian
    Ls: list of (N,N) complex Lindblad operators
    returns drho/dt
    """
    drho = -1j * (H @ rho - rho @ H)

    for Lk in Ls:
        Ldag = Lk.mH
        LdagL = Ldag @ Lk
        drho = drho + (
            Lk @ rho @ Ldag
            - 0.5 * (LdagL @ rho + rho @ LdagL)
        )
    return drho

def evolve_expm(rho0, H, Ls, T):
    L = liouvillian(H, Ls)
    U = torch.matrix_exp(L * T) # 显式构造 Liouvillian 的指数矩阵，适用于小系统（N不大）, 低效
    if rho0.dim() == 2:
        vT = U @ vec(rho0)
        return unvec(vT, rho0.shape[0])

    # batched rho0: (B, N, N)
    B, N, _ = rho0.shape
    v0 = rho0.transpose(-1, -2).reshape(B, N * N).T.contiguous()  # (N^2, B)
    vT = U @ v0
    return vT.T.reshape(B, N, N).transpose(-1, -2).contiguous()


def evolve_unitary(rho0, H, T):
    """
    仅幺正演化（Ls 为空时的高效路径）：
        rho(T) = U rho(0) U^†, U = exp(-i H T)
    """
    U = torch.matrix_exp((-1j) * H * T)
    Udag = U.mH

    if rho0.dim() == 2:
        return U @ rho0 @ Udag

    return U.unsqueeze(0) @ rho0 @ Udag.unsqueeze(0)

def evolve_vec_rk4(rho0, H, Ls, T, steps=100): # 直接在密度矩阵空间使用RK4方法数值求解Liouvillian演化
    rho = rho0.clone()
    dt = T / steps

    for _ in range(steps):
        k1 = lindblad_rhs(rho, H, Ls)
        k2 = lindblad_rhs(rho + 0.5 * dt * k1, H, Ls)
        k3 = lindblad_rhs(rho + 0.5 * dt * k2, H, Ls)
        k4 = lindblad_rhs(rho + dt * k3, H, Ls)

        rho = rho + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return rho



def liouvillian_mv(H, Ls): # Arnoldi/Krylov expm_multiply方法
    """
    返回一个函数 A_mv(x)，其中 x shape = (N^2,1)
    表示 A_mv(x) = L @ x，但不显式构造 L
    """
    N = H.shape[0]

    def A_mv(x):
        rho = unvec(x, N)
        drho = lindblad_rhs(rho, H, Ls)
        return vec(drho)

    return A_mv


def _norm2(x): # 计算向量的2范数
    return torch.linalg.vector_norm(x)


def _arnoldi(A_mv, v, m, tol=1e-8): # Arnoldi算法，构造Krylov子空间
    device, dtype = v.device, v.dtype
    n = v.shape[0]

    beta = _norm2(v)
    if beta.abs().item() == 0.0:
        V = torch.zeros((n, 1), dtype=dtype, device=device)
        H = torch.zeros((1, 0), dtype=dtype, device=device)
        return V, H, 0, beta

    # 我们不预先分配V和H以避免对需要梯度的张量进行原位修改
    V_cols = []  # 存储每一列
    H_rows = []  # 每个j时的一行元素

    V_cols.append(v / beta)
    k = 0

    for j in range(m):
        w = A_mv(V_cols[j])

        # Modified Gram-Schmidt on list elements
        row = []
        for i in range(j + 1):
            hij = torch.sum(V_cols[i].conj() * w)
            row.append(hij)
            w = w - hij * V_cols[i]

        h_next = _norm2(w)
        row.append(h_next)
        H_rows.append(row)

        if h_next.abs().item() < tol:
            k = j + 1
            break

        V_cols.append(w / h_next)
        k = j + 1

    # 将列拼接成矩阵V
    V = torch.cat(V_cols, dim=1)  # size (n, k+1) 或 (n, k) 取决于break
    # 构造H矩阵
    H = torch.zeros((k + 1, k), dtype=dtype, device=device)
    for j, row in enumerate(H_rows[:k]):
        for i, val in enumerate(row):
            H[i, j] = val

    return V[:, :k+1], H[:k+1, :k], k, beta


def _krylov_expm_multiply(A_mv, v, t, m=20, tol=1e-8):
    V, H, k, beta = _arnoldi(A_mv, v, m=m, tol=tol)

    if k == 0:
        return torch.zeros_like(v)

    if H.shape[0] == H.shape[1]:
        # happy breakdown
        Hk = H
        Vk = V
    else:
        Hk = H[:k, :k]
        Vk = V[:, :k]

    e1 = torch.zeros((k, 1), dtype=v.dtype, device=v.device)
    e1[0, 0] = 1.0

    y = beta * (torch.matrix_exp(t * Hk) @ e1)
    return Vk @ y


def evolve_from_operators(rho0, H, Ls, T, krylov_dim=20, steps=10, tol=1e-8):
    """
    推荐接口：
        不显式构造 Liouvillian
        直接近似 exp(L*T) @ vec(rho0)
    """
    if rho0.dim() == 2:
        v = vec(rho0)
        dt = T / steps
        A_mv = liouvillian_mv(H, Ls)

        for _ in range(steps):
            v = _krylov_expm_multiply(A_mv, v, dt, m=krylov_dim, tol=tol)

        return unvec(v, rho0.shape[0])

    # batched rho0: (B, N, N)
    outs = [
        evolve_from_operators(rho0[b], H, Ls, T, krylov_dim=krylov_dim, steps=steps, tol=tol)
        for b in range(rho0.shape[0])
    ]
    return torch.stack(outs, dim=0)


def evolve_auto(
    rho0,
    H,
    Ls,
    T,
    dense_cutoff_cpu=28,
    dense_cutoff_cuda=24,
    dense_cutoff_cuda_batched=40,
    krylov_dim=20,
    steps=10,
    tol=1e-8,
):
    """
    自适应演化接口：
    - 小规模系统：使用显式 expm（通常更快）
    - 较大规模系统：使用 Krylov expm_multiply（避免显式 Liouvillian 指数）
    """
    if len(Ls) == 0:
        return evolve_unitary(rho0, H, T)

    N = H.shape[0]
    is_batched = rho0.dim() == 3 and rho0.shape[0] > 1
    if H.device.type == "cuda":
        cutoff = dense_cutoff_cuda_batched if is_batched else dense_cutoff_cuda
        use_dense = N <= cutoff
    else:
        use_dense = N <= dense_cutoff_cpu

    if use_dense:
        return evolve_expm(rho0, H, Ls, T)

    return evolve_from_operators(
        rho0,
        H,
        Ls,
        T,
        krylov_dim=krylov_dim,
        steps=steps,
        tol=tol,
    )


def _lindblad_rhs_qsnn2d_structured(rho, H, gamma, N_in):
    """
    QSNN2D 第二阶段专用 RHS（结构化耗散）：
    L_{o,j} = gamma[o,j] |o><j|, o in {N_in, N_in+1}, j in [0, N_in)

    使用结构公式避免构造 L 列表与大规模矩阵乘法。
    """
    N = H.shape[0]
    out0, out1 = N_in, N_in + 1

    if rho.dim() == 2:
        rho = rho.unsqueeze(0)
        squeeze_back = True
    else:
        squeeze_back = False

    # coherent part
    drho = -1j * (H.unsqueeze(0) @ rho - rho @ H.unsqueeze(0))

    # dissipative damping coefficients on input nodes
    gabs2 = gamma.abs() ** 2  # (2, N_in)
    damp_in = gabs2.sum(dim=0)  # (N_in,)
    damp_full = torch.zeros((N,), device=H.device, dtype=H.dtype)
    damp_full[:N_in] = damp_in.to(H.dtype)

    # -1/2 {D, rho}, where D = diag(damp_full)
    drho = drho - 0.5 * (
        damp_full.view(1, N, 1) * rho + rho * damp_full.view(1, 1, N)
    )

    # jump terms: only contribute to output diagonal
    rho_in_diag = torch.diagonal(rho[:, :N_in, :N_in], dim1=-2, dim2=-1)  # (B, N_in)
    gain0 = (rho_in_diag * gabs2[0].to(rho.dtype).view(1, N_in)).sum(dim=1)
    gain1 = (rho_in_diag * gabs2[1].to(rho.dtype).view(1, N_in)).sum(dim=1)

    drho[:, out0, out0] = drho[:, out0, out0] + gain0
    drho[:, out1, out1] = drho[:, out1, out1] + gain1

    if squeeze_back:
        return drho[0]
    return drho


def evolve_qsnn2d_stage2_structured(rho0, H, gamma, T, N_in, steps=20):
    """
    QSNN2D 第二阶段专用演化器：结构化 Lindblad + RK4
    用于大 N（如 N≈100）场景，避免显式 Liouvillian 或逐样本 Krylov 退化。
    """
    rho = rho0.clone()
    dt = T / steps

    for _ in range(steps):
        k1 = _lindblad_rhs_qsnn2d_structured(rho, H, gamma, N_in)
        k2 = _lindblad_rhs_qsnn2d_structured(rho + 0.5 * dt * k1, H, gamma, N_in)
        k3 = _lindblad_rhs_qsnn2d_structured(rho + 0.5 * dt * k2, H, gamma, N_in)
        k4 = _lindblad_rhs_qsnn2d_structured(rho + dt * k3, H, gamma, N_in)
        rho = rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return rho


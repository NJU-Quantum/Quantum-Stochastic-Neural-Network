# qsw.py
import torch
from scipy import special

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


def evolve_state_exact(psi0, H, T):
    """
    纯态幺正演化：
        psi(T) = U psi(0), U = exp(-i H T)

    支持:
    - psi0: (N, 1)
    - psi0: (B, N, 1)
    """
    U = torch.matrix_exp((-1j) * H * T)

    if psi0.dim() == 2:
        return U @ psi0

    return U.unsqueeze(0) @ psi0


def evolve_state_suzuki(psi0, H, T, steps=12, order=2):
    """
    Suzuki splitting 纯态演化：
        H = H_diag + H_off
        支持偶数阶 order = 2, 4, 6, ...

    当前使用经典递归构造：
        S_{2}(dt) = exp(-i H_diag dt/2) exp(-i H_off dt) exp(-i H_diag dt/2)
        S_{2k+2}(dt) = S_{2k}(p dt)^2 S_{2k}((1-4p)dt) S_{2k}(p dt)^2
    """
    if order < 2 or order % 2 != 0:
        raise ValueError(f"Suzuki order must be an even integer >= 2, got {order}")

    N = H.shape[0]
    H_diag = torch.diag(torch.diagonal(H))
    H_off = H - H_diag
    dt = T / steps
    cache = {}

    def get_ops(dt_local):
        key = float(dt_local)
        if key not in cache:
            U_diag_half = torch.matrix_exp((-0.5j) * H_diag * dt_local)
            U_off = torch.matrix_exp((-1j) * H_off * dt_local)
            cache[key] = (U_diag_half, U_off)
        return cache[key]

    def apply_s2(psi, dt_local):
        U_diag_half, U_off = get_ops(dt_local)
        if psi.dim() == 2:
            psi = U_diag_half @ psi
            psi = U_off @ psi
            psi = U_diag_half @ psi
            return psi

        psi = U_diag_half.unsqueeze(0) @ psi
        psi = U_off.unsqueeze(0) @ psi
        psi = U_diag_half.unsqueeze(0) @ psi
        return psi

    def apply_suzuki_recursive(psi, dt_local, order_local):
        if order_local == 2:
            return apply_s2(psi, dt_local)

        k = order_local // 2 - 1
        p = 1.0 / (4.0 - 4.0 ** (1.0 / (2 * k + 1)))
        psi = apply_suzuki_recursive(psi, p * dt_local, order_local - 2)
        psi = apply_suzuki_recursive(psi, p * dt_local, order_local - 2)
        psi = apply_suzuki_recursive(psi, (1.0 - 4.0 * p) * dt_local, order_local - 2)
        psi = apply_suzuki_recursive(psi, p * dt_local, order_local - 2)
        psi = apply_suzuki_recursive(psi, p * dt_local, order_local - 2)
        return psi

    psi = psi0
    for _ in range(steps):
        psi = apply_suzuki_recursive(psi, dt, order)
    return psi


def evolve_state_chebyshev(psi0, H, T, max_order=128, tol=1e-10):
    """
    用 Chebyshev 展开近似计算 psi(T) = exp(-i H T) psi(0)。

    适用于 Hermitian H。支持:
    - psi0: (N, 1)
    - psi0: (B, N, 1)
    """
    if psi0.dim() == 2:
        psi = psi0.unsqueeze(0)
        squeeze_back = True
    else:
        psi = psi0
        squeeze_back = False

    N = H.shape[0]
    device = H.device
    dtype = H.dtype

    evals = torch.linalg.eigvalsh(H)
    e_min = evals[0].real.detach()
    e_max = evals[-1].real.detach()
    center = 0.5 * (e_max + e_min)
    radius = 0.5 * (e_max - e_min)

    if radius.abs().item() < 1e-12:
        phase = torch.exp((-1j) * center.to(dtype) * T)
        out = phase * psi
        if squeeze_back:
            return out[0]
        return out

    eye = torch.eye(N, device=device, dtype=dtype)
    H_scaled = (H - center.to(dtype) * eye) / radius.to(dtype)
    tau = float((radius * T).item())
    global_phase = torch.exp((-1j) * center.to(dtype) * T)

    t0 = psi
    out = special.jv(0, tau) * t0

    if max_order >= 1:
        t1 = H_scaled.unsqueeze(0) @ psi
        coeff = 2.0 * special.jv(1, tau) * (-1j)
        out = out + coeff * t1
    else:
        t1 = None

    for n in range(2, max_order + 1):
        coeff_n = 2.0 * special.jv(n, tau) * ((-1j) ** n)
        if abs(coeff_n) < tol and n > abs(tau):
            break

        tn = 2.0 * (H_scaled.unsqueeze(0) @ t1) - t0
        out = out + coeff_n * tn
        t0, t1 = t1, tn

    out = global_phase * out
    if squeeze_back:
        return out[0]
    return out

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


def _qsnn2d_structured_dissipative_exact_step(rho, gamma, dt, N_in):
    """
    QSNN2D 结构化耗散部分的精确一步更新：
        d rho / dt = -1/2 {D, rho} + J(rho)

    其中:
    - D 为输入节点上的对角阻尼
    - J 只将输入对角元泵入两个输出对角元
    """
    if rho.dim() == 2:
        rho_b = rho.unsqueeze(0)
        squeeze_back = True
    else:
        rho_b = rho
        squeeze_back = False

    B, N, _ = rho_b.shape
    out0, out1 = N_in, N_in + 1
    gabs2 = gamma.abs() ** 2  # (2, N_in)
    damp_in = gabs2.sum(dim=0).to(rho_b.real.dtype)  # (N_in,)

    damp_full = torch.zeros((N,), device=rho_b.device, dtype=rho_b.real.dtype)
    damp_full[:N_in] = damp_in

    decay = torch.exp(
        -0.5 * dt * (damp_full.view(N, 1) + damp_full.view(1, N))
    ).to(rho_b.dtype)
    rho_next = rho_b * decay.view(1, N, N)

    rho_in_diag0 = torch.diagonal(rho_b[:, :N_in, :N_in], dim1=-2, dim2=-1).real

    transfer = torch.where(
        damp_in > 1e-12,
        (1.0 - torch.exp(-damp_in * dt)) / damp_in,
        torch.full_like(damp_in, dt),
    )

    gain_weights0 = (gabs2[0].to(rho_b.real.dtype) * transfer).view(1, N_in)
    gain_weights1 = (gabs2[1].to(rho_b.real.dtype) * transfer).view(1, N_in)
    gain0 = (rho_in_diag0 * gain_weights0).sum(dim=1)
    gain1 = (rho_in_diag0 * gain_weights1).sum(dim=1)

    rho_next[:, out0, out0] = rho_next[:, out0, out0] + gain0.to(rho_b.dtype)
    rho_next[:, out1, out1] = rho_next[:, out1, out1] + gain1.to(rho_b.dtype)

    if squeeze_back:
        return rho_next[0]
    return rho_next


def evolve_qsnn2d_stage2_split(rho0, H, gamma, T, N_in, steps=20):
    """
    QSNN2D 第二阶段对称分裂演化器：
        exp(dt/2 L_H) exp(dt (L_D + L_J)) exp(dt/2 L_H)

    其中耗散部分 (L_D + L_J) 采用结构化精确一步更新。
    """
    rho = rho0.clone()
    dt = T / steps
    U_half = torch.matrix_exp((-0.5j) * H * dt)
    U_half_dag = U_half.mH

    if rho.dim() == 2:
        for _ in range(steps):
            rho = U_half @ rho @ U_half_dag
            rho = _qsnn2d_structured_dissipative_exact_step(rho, gamma, dt, N_in)
            rho = U_half @ rho @ U_half_dag
        return rho

    U_half_b = U_half.unsqueeze(0)
    U_half_dag_b = U_half_dag.unsqueeze(0)
    for _ in range(steps):
        rho = U_half_b @ rho @ U_half_dag_b
        rho = _qsnn2d_structured_dissipative_exact_step(rho, gamma, dt, N_in)
        rho = U_half_b @ rho @ U_half_dag_b
    return rho


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

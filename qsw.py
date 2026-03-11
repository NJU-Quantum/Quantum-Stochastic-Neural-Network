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
    vT = U @ vec(rho0)
    return unvec(vT, rho0.shape[0])

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
    v = vec(rho0)
    dt = T / steps
    A_mv = liouvillian_mv(H, Ls)

    for _ in range(steps):
        v = _krylov_expm_multiply(A_mv, v, dt, m=krylov_dim, tol=tol)

    return unvec(v, rho0.shape[0])


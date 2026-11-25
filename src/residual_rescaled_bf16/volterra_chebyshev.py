"""Clenshaw-Curtis Spectral Volterra Operators and CUDAGraph-Safe CGS2-QR Anderson Acceleration."""

from __future__ import annotations

import numpy as np
import torch


def build_cheb_volterra(M: int, device: str | torch.device = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """Construct exact Clenshaw-Curtis spectral Volterra integration matrix Q on [0, 1].

    Given Chebyshev-Lobatto nodes t_j = 0.5 * (1 - cos(pi * j / M)) for j = 0..M,
    the returned (M+1, M+1) operator Q satisfies:
        (Q @ V)[j] = int_0^{t_j} p_M(s) ds
    exactly for every polynomial p_M of degree <= M interpolating V at {t_j}.
    """
    j = np.arange(M + 1, dtype=np.float64)
    xi = -np.cos(np.pi * j / M)
    nodes = 0.5 * (xi + 1.0)
    k = np.arange(M + 1, dtype=np.float64)[:, None]
    T_kj = ((-1.0) ** k) * np.cos(np.pi * k * j[None, :] / M)
    p = np.ones(M + 1, dtype=np.float64)
    p[0] = 2.0
    p[M] = 2.0
    C = (2.0 / M) * (T_kj / (p[:, None] * p[None, :]))

    k_ext = np.arange(M + 2, dtype=np.float64)[:, None]
    T_ext = ((-1.0) ** k_ext) * np.cos(np.pi * k_ext * j[None, :] / M)
    I_kj = np.zeros((M + 1, M + 1), dtype=np.float64)
    I_kj[0] = T_ext[1] + T_ext[0]
    I_kj[1] = 0.25 * (T_ext[2] - T_ext[0])
    for kk in range(2, M + 1):
        I_kj[kk] = (
            T_ext[kk + 1] / (2.0 * (kk + 1))
            - T_ext[kk - 1] / (2.0 * (kk - 1))
            + ((-1.0) ** (kk + 1)) / (kk * kk - 1.0)
        )
    Q = 0.5 * (I_kj.T @ C)
    Q[0] = 0.0
    return (
        torch.tensor(nodes, dtype=torch.float32, device=device),
        torch.tensor(Q, dtype=torch.float32, device=device),
    )


def build_exp_chebyshev_panel(
    lam_a: float,
    lam_b: float,
    M: int,
    device: str | torch.device = "cuda",
) -> tuple[np.ndarray, torch.Tensor]:
    """Construct exponentially weighted Clenshaw-Curtis Volterra operator on [lam_a, lam_b].

    Integrates int_{lam_a}^{lam_j} (-exp(-s)) * eps_hat(s) ds exactly up to 64-point
    Gauss-Legendre precision in FP64, eliminating linear drift discretization error in
    half-log-SNR space.
    """
    j = np.arange(M + 1, dtype=np.float64)
    xi = -np.cos(np.pi * j / M)
    lam_nodes = lam_a + 0.5 * (lam_b - lam_a) * (xi + 1.0)
    k = np.arange(M + 1, dtype=np.float64)[:, None]
    T_kj = ((-1.0) ** k) * np.cos(np.pi * k * j[None, :] / M)
    p = np.ones(M + 1, dtype=np.float64)
    p[0] = 2.0
    p[M] = 2.0
    C = (2.0 / M) * (T_kj / (p[:, None] * p[None, :]))

    gl_x, gl_w = np.polynomial.legendre.leggauss(64)
    I_jk = np.zeros((M + 1, M + 1), dtype=np.float64)
    for jj in range(1, M + 1):
        la, lb = lam_a, lam_nodes[jj]
        s_pts = 0.5 * (lb - la) * gl_x + 0.5 * (lb + la)
        w_pts = 0.5 * (lb - la) * gl_w
        xi_pts = 2.0 * (s_pts - lam_a) / (lam_b - lam_a) - 1.0
        th_pts = np.arccos(np.clip(xi_pts, -1.0, 1.0))
        T_pts = np.cos(np.arange(M + 1, dtype=np.float64)[:, None] * th_pts[None, :])
        I_jk[jj, :] = (-np.exp(-s_pts)[None, :] * T_pts) @ w_pts
    Q_exp = I_jk @ C
    Q_exp[0, :] = 0.0
    return lam_nodes, torch.tensor(Q_exp, dtype=torch.float32, device=device)


def build_z_chebyshev_panel(
    za: float,
    zb: float,
    M: int,
    device: str | torch.device = "cuda",
) -> tuple[np.ndarray, torch.Tensor]:
    """Construct root-time z = sqrt(tau) Clenshaw-Curtis Volterra operator on [za, zb].

    Used on high-noise VP diffusion panels (SNR <= 1) where dividing BF16 eps_theta by
    alpha_T = 0.00657 would amplify mantissa rounding noise by 152x.
    """
    j = np.arange(M + 1, dtype=np.float64)
    xi = -np.cos(np.pi * j / M)
    z_nodes = za + 0.5 * (zb - za) * (xi + 1.0)
    k = np.arange(M + 1, dtype=np.float64)[:, None]
    T_kj = ((-1.0) ** k) * np.cos(np.pi * k * j[None, :] / M)
    p = np.ones(M + 1, dtype=np.float64)
    p[0] = 2.0
    p[M] = 2.0
    C = (2.0 / M) * (T_kj / (p[:, None] * p[None, :]))

    gl_x, gl_w = np.polynomial.legendre.leggauss(32)
    I_jk = np.zeros((M + 1, M + 1), dtype=np.float64)
    for jj in range(1, M + 1):
        u_pts = 0.5 * (xi[jj] + 1.0) * gl_x + 0.5 * (xi[jj] - 1.0)
        w_pts = 0.5 * (xi[jj] + 1.0) * gl_w * 0.5 * (zb - za)
        th = np.arccos(np.clip(u_pts, -1.0, 1.0))
        T_pts = np.cos(np.arange(M + 1, dtype=np.float64)[:, None] * th[None, :])
        I_jk[jj, :] = T_pts @ w_pts
    Q_z = I_jk @ C
    Q_z[0, :] = 0.0
    return z_nodes, torch.tensor(Q_z, dtype=torch.float32, device=device)


def graph_safe_anderson_cgs2(
    Xh: list[torch.Tensor],
    Gh: list[torch.Tensor],
    m_max: int = 4,
    reg: float = 1e-5,
) -> torch.Tensor:
    """Zero-cusolver, zero-CPU-sync CGS2-QR Type-II Anderson Acceleration (100% CUDAGraph capturable)."""
    k = len(Xh) - 1
    if k == 0:
        return Gh[-1]
    m = min(k, m_max)
    Fk = (Gh[-1] - Xh[-1]).reshape(-1)
    dF_cols = [
        (Gh[-1 - i + 1] - Xh[-1 - i + 1]).reshape(-1) - (Gh[-1 - i] - Xh[-1 - i]).reshape(-1)
        for i in range(1, m + 1)
    ]
    dG = torch.stack(
        [Gh[-1 - i + 1].reshape(-1) - Gh[-1 - i].reshape(-1) for i in range(1, m + 1)],
        dim=1,
    )

    Q_cols: list[torch.Tensor] = []
    inv_norms: list[torch.Tensor] = []
    R_rows: list[list[torch.Tensor | None]] = [[None] * m for _ in range(m)]
    for j in range(m):
        v = dF_cols[j]
        cn = torch.linalg.norm(v) + 1e-20
        inv_norms.append(1.0 / cn)
        v = v * inv_norms[j]
        for i in range(j):
            qi = Q_cols[i]
            h1 = torch.dot(qi, v)
            v = v - h1 * qi
            h2 = torch.dot(qi, v)
            v = v - h2 * qi
            R_rows[i][j] = h1 + h2
        rjj = torch.linalg.norm(v) + 1e-20
        R_rows[j][j] = rjj
        Q_cols.append(v / rjj)

    rhs = [-torch.dot(Q_cols[i], Fk) for i in range(m)]
    y: list[torch.Tensor] = [torch.empty(0)] * m
    for i in range(m - 1, -1, -1):
        s = rhs[i]
        for j in range(i + 1, m):
            rij = R_rows[i][j]
            assert rij is not None
            s = s - rij * y[j]
        rii = R_rows[i][i]
        assert rii is not None
        y[i] = s / (rii + reg)

    gamma = torch.stack([y[i] * inv_norms[i] for i in range(m)])
    return (Gh[-1].reshape(-1) + dG @ gamma).view_as(Gh[-1])


def batched_cgs2_anderson(
    Xh: list[torch.Tensor],
    Gh: list[torch.Tensor],
    m_max: int = 3,
    reg: float = 2e-5,
) -> torch.Tensor:
    """Per-sample batched CGS2-QR Type-II Anderson Acceleration for 5D image trajectories [B, M, C, H, W]."""
    k = len(Xh) - 1
    if k == 0:
        return Gh[-1]
    m = min(k, m_max)
    B_s = Xh[-1].shape[0]
    Fk = (Gh[-1] - Xh[-1]).reshape(B_s, -1)
    dF = [
        (Gh[-1 - i + 1] - Xh[-1 - i + 1]).reshape(B_s, -1)
        - (Gh[-1 - i] - Xh[-1 - i]).reshape(B_s, -1)
        for i in range(1, m + 1)
    ]
    dG = torch.stack(
        [Gh[-1 - i + 1].reshape(B_s, -1) - Gh[-1 - i].reshape(B_s, -1) for i in range(1, m + 1)],
        dim=2,
    )
    Q: list[torch.Tensor] = []
    inv_n: list[torch.Tensor] = []
    R: list[list[torch.Tensor | None]] = [[None] * m for _ in range(m)]
    for j in range(m):
        v = dF[j]
        cn = torch.linalg.norm(v, dim=1, keepdim=True) + 1e-20
        inv_n.append(1.0 / cn)
        v = v * inv_n[j]
        for i in range(j):
            qi = Q[i]
            h1 = torch.sum(qi * v, dim=1, keepdim=True)
            v = v - h1 * qi
            h2 = torch.sum(qi * v, dim=1, keepdim=True)
            v = v - h2 * qi
            R[i][j] = h1 + h2
        rjj = torch.linalg.norm(v, dim=1, keepdim=True) + 1e-20
        R[j][j] = rjj
        Q.append(v / rjj)
    rhs = [-torch.sum(Q[i] * Fk, dim=1, keepdim=True) for i in range(m)]
    y: list[torch.Tensor] = [torch.empty(0)] * m
    for i in range(m - 1, -1, -1):
        s = rhs[i]
        for j in range(i + 1, m):
            rij = R[i][j]
            assert rij is not None
            s = s - rij * y[j]
        rii = R[i][i]
        assert rii is not None
        y[i] = s / (rii + reg)
    gam = torch.stack([y[i] * inv_n[i] for i in range(m)], dim=1)
    return (Gh[-1].reshape(B_s, -1) + torch.bmm(dG, gam).squeeze(-1)).reshape_as(Gh[-1])

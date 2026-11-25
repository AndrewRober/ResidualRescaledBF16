"""CSP-BF16-IR Solver, PTX-Fused SwiGLU DiT Operators, and Sequential/ParaDiGMS Baselines."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ptx_kernels import launch_ptx_csp_pack, launch_ptx_csp_unpack, launch_ptx_swiglu
from .volterra_chebyshev import build_cheb_volterra, graph_safe_anderson_cgs2


class PTXDiTBlock(nn.Module):
    """SwiGLU adaLN Diffusion Transformer block backed by fused PTX activation kernels."""

    def __init__(self, d_model: int, d_ff: int, res_scale: float = 0.45, zero_init: bool = False):
        super().__init__()
        self.res_scale = res_scale
        self.d_ff = d_ff
        self.norm = nn.LayerNorm(d_model)
        self.ada = nn.Linear(d_model, 2 * d_model)
        self.w12 = nn.Linear(d_model, 2 * d_ff)
        self.w3 = nn.Linear(d_ff, d_model)
        if zero_init:
            nn.init.normal_(self.w3.weight, std=0.02 / math.sqrt(d_ff))
            nn.init.zeros_(self.w3.bias)
        else:
            nn.init.normal_(self.w12.weight, std=1.18 / math.sqrt(d_model))
            nn.init.normal_(self.w3.weight, std=0.88 / math.sqrt(d_ff))

    def forward(
        self,
        h: torch.Tensor,
        t_emb: torch.Tensor,
        act_buf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gamma, beta = self.ada(t_emb).chunk(2, dim=-1)
        u = self.norm(h) * (1.0 + 0.25 * gamma) + 0.25 * beta
        w12_out = self.w12(u)
        if (not torch.is_grad_enabled()) and h.dtype in (torch.bfloat16, torch.float32):
            if act_buf is None or act_buf.shape[0] != h.shape[0] or act_buf.dtype != h.dtype:
                act_buf = torch.empty((h.shape[0], self.d_ff), device=h.device, dtype=h.dtype)
            launch_ptx_swiglu(w12_out, act_buf)
            return h + self.res_scale * self.w3(act_buf)
        u1, u2 = w12_out.chunk(2, dim=-1)
        return h + self.res_scale * self.w3(F.silu(u1) * u2)


class PTXDiTVelocityOperator(nn.Module):
    """46.91M-Parameter Deep Residual SwiGLU DiT Velocity Operator (d=1280, h=3072)."""

    def __init__(self, d: int = 1280, h: int = 3072, n_blocks: int = 3, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.d = d
        self.h = h
        self.register_buffer(
            "freqs", torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float32) * math.pi
        )
        self.t_mlp = nn.Sequential(nn.Linear(8, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList(
            [PTXDiTBlock(d, h, res_scale=0.45, zero_init=False) for _ in range(n_blocks)]
        )
        A = torch.randn(d, 32) / math.sqrt(d * 32)
        B = torch.randn(d, 32) / math.sqrt(d * 32)
        W_rot = 18.0 * (A @ B.T - B @ A.T) - 0.25 * torch.eye(d)
        self.register_buffer("W_rot", W_rot)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        act_buf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dtp = x.dtype
        ang = t[:, None].to(self.freqs.dtype) * self.freqs[None, :]
        phi = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1).to(dtp)
        t_emb = self.t_mlp(phi)
        h = x
        for blk in self.blocks:
            h = blk(h, t_emb, act_buf)
        return (h - x) + (x @ self.W_rot.to(dtp))


class Spatial2DGlyphDiT(nn.Module):
    """34.65M-Parameter 2D Structured Image Flow-Matching DiT on R^{16x16}."""

    def __init__(self, img_hw: int = 16, d_model: int = 1024, d_ff: int = 2816, n_blocks: int = 3):
        super().__init__()
        self.img_hw = img_hw
        self.d_data = img_hw * img_hw
        self.d_ff = d_ff
        self.register_buffer(
            "freqs",
            torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=torch.float32) * math.pi,
        )
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, img_hw),
            torch.linspace(-1.0, 1.0, img_hw),
            indexing="ij",
        )
        pos2d = torch.stack([xx, yy, xx * xx + yy * yy], dim=0).reshape(3 * self.d_data)
        self.register_buffer("pos2d", pos2d)
        self.in_proj = nn.Linear(self.d_data, d_model)
        self.pos_proj = nn.Linear(3 * self.d_data, d_model, bias=False)
        self.t_mlp = nn.Sequential(
            nn.Linear(12, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [PTXDiTBlock(d_model, d_ff, res_scale=1.0, zero_init=True) for _ in range(n_blocks)]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, self.d_data)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        act_buf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dtp = x.dtype
        ang = t[:, None].to(self.freqs.dtype) * self.freqs[None, :]
        phi = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1).to(dtp)
        t_emb = self.t_mlp(phi)
        h = self.in_proj(x) + self.pos_proj(self.pos2d.to(dtp))[None, :]
        for blk in self.blocks:
            h = blk(h, t_emb, act_buf)
        return self.out_proj(self.out_norm(h))


def capture_cudagraph(body_fn, warm_iters: int = 1) -> torch.cuda.CUDAGraph:
    """Warm up on an isolated CUDA stream and capture `body_fn` into a replayable CUDAGraph."""
    s_warm = torch.cuda.Stream()
    s_warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s_warm):
        for _ in range(warm_iters):
            body_fn()
    torch.cuda.current_stream().wait_stream(s_warm)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body_fn()
    return g


@torch.inference_mode()
def gpu_seq_rk4_body(
    net: nn.Module,
    x0_buf: torch.Tensor,
    out_buf: torch.Tensor,
    S_nfe: int,
    use_bf16: bool = False,
) -> None:
    """Sequential classical 4th-order Runge-Kutta baseline (S_nfe total evaluations)."""
    steps = S_nfe // 4
    dt = 1.0 / steps
    x = x0_buf.clone()
    tz = torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)
    f = (lambda z, t: net(z.to(torch.bfloat16), t).float()) if use_bf16 else net
    for s in range(steps):
        t = tz + s * dt
        k1 = f(x, t)
        k2 = f(x + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = f(x + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = f(x + dt * k3, t + dt)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    out_buf.copy_(x)


@torch.inference_mode()
def gpu_seq_euler_body(
    net: nn.Module,
    x0_buf: torch.Tensor,
    out_buf: torch.Tensor,
    S: int,
) -> None:
    """Sequential forward Euler baseline."""
    dt = 1.0 / S
    x = x0_buf.clone()
    tz = torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)
    for s in range(S):
        x = x + dt * net(x, tz + s * dt)
    out_buf.copy_(x)


@torch.inference_mode()
def gpu_paradigms_body(
    net_bf16: nn.Module,
    x0_buf: torch.Tensor,
    out_buf: torch.Tensor,
    S: int,
    W: int = 32,
    K_sweeps: int = 4,
) -> None:
    """Steelmanned ParaDiGMS sliding-window parallel Picard solver in BF16."""
    dt = 1.0 / S
    n_win = S // W
    B, D = x0_buf.shape
    x_curr = x0_buf.clone()
    j_idx = torch.arange(1, W + 1, device=x0_buf.device, dtype=torch.float32)
    tz = torch.zeros(B, device=x0_buf.device, dtype=torch.float32)
    v_curr = net_bf16(x_curr.to(torch.bfloat16), tz).float()
    for w in range(n_win):
        t_win = (w * W * dt) + j_idx * dt
        t_flat = t_win.repeat(B)
        X = x_curr[:, None, :] + (j_idx[None, :, None] * dt) * v_curr[:, None, :]
        for _ in range(K_sweeps):
            V = net_bf16(X.reshape(B * W, D).to(torch.bfloat16), t_flat).float().view(B, W, D)
            V_all = torch.cat([v_curr[:, None, :], V], dim=1)
            seg = 0.5 * dt * (V_all[:, :-1, :] + V_all[:, 1:, :])
            X = x_curr[:, None, :] + torch.cumsum(seg, dim=1)
        v_curr = V[:, -1, :]
        x_curr = X[:, -1, :]
    out_buf.copy_(x_curr)


class PTXCspBf16Solver:
    """Central-Difference Scaled-Perturbation BF16 Spectral Volterra-Newton Solver (`CSP-BF16-IR`).

    Elevates sub-mantissa FP32 Volterra defect vectors R_k into the 7-bit BF16 sweet spot
    via +/- (sigma * ||X_k|| / ||R_k||) R_k with sigma = O(eps_bf16^{1/3}) = 0.125,
    cancelling the even Hessian D^2 V[R_k, R_k] in a single fused [2M, d] BF16 GEMM.
    """

    def __init__(
        self,
        net_bf16: nn.Module,
        net_fp32: nn.Module,
        M: int,
        d_state: int,
        h_ff: int,
        device: str | torch.device = "cuda",
    ):
        self.net_bf16 = net_bf16
        self.net_fp32 = net_fp32
        self.M = M
        self.D = d_state
        self.H = h_ff
        self.device = torch.device(device)
        nodes, Q = build_cheb_volterra(M, self.device)
        self.q0 = Q[1:, 0]
        self.Qa = Q[1:, 1:]
        self.ta = nodes[1:]
        self.ta2 = torch.cat([self.ta, self.ta], dim=0)
        self.tz1 = torch.zeros(1, device=self.device, dtype=torch.float32)

        self.N_tot = M * d_state
        self.pair_buf = torch.empty(2 * M, d_state, device=self.device, dtype=torch.bfloat16)
        self.j_buf = torch.empty(M, d_state, device=self.device, dtype=torch.float32)
        self.act_1f = torch.empty(1, h_ff, device=self.device, dtype=torch.float32)
        self.act_Mf = torch.empty(M, h_ff, device=self.device, dtype=torch.float32)
        self.act_Mb = torch.empty(M, h_ff, device=self.device, dtype=torch.bfloat16)
        self.act_2Mb = torch.empty(2 * M, h_ff, device=self.device, dtype=torch.bfloat16)

    @torch.inference_mode()
    def run_b1_into(
        self,
        x0_buf: torch.Tensor,
        out_buf: torch.Tensor,
        bf_init: int = 4,
        outer_ir: int = 3,
        inner_jvp: int = 2,
        sigma: float = 0.125,
        central: bool = True,
    ) -> None:
        if outer_ir == 0:
            v0 = self.net_bf16(x0_buf.to(torch.bfloat16), self.tz1, self.act_Mb[:1]).float()
        else:
            v0 = self.net_fp32(x0_buf, self.tz1, self.act_1f)
        base = x0_buf + self.q0[:, None] * v0
        X = x0_buf + self.ta[:, None] * v0
        Xh: list[torch.Tensor] = []
        Gh: list[torch.Tensor] = []
        for k in range(bf_init):
            G = base + self.Qa @ self.net_bf16(X.to(torch.bfloat16), self.ta, self.act_Mb).float()
            if outer_ir == 0 and k == bf_init - 1:
                out_buf.copy_(G[-1:])
                return
            Xh.append(X)
            Gh.append(G)
            if len(Xh) > 5:
                Xh.pop(0)
                Gh.pop(0)
            X = graph_safe_anderson_cgs2(Xh, Gh, m_max=4, reg=1e-5)

        for _ in range(outer_ir):
            R = (base + self.Qa @ self.net_fp32(X, self.ta, self.act_Mf)) - X
            nx2 = (X * X).sum()
            delta = R
            if central:
                for _ in range(inner_jvp):
                    nd2 = (delta * delta).sum()
                    launch_ptx_csp_pack(X, delta, nx2, nd2, self.pair_buf, sigma, self.N_tot)
                    Vp_pair = self.net_bf16(self.pair_buf, self.ta2, self.act_2Mb)
                    launch_ptx_csp_unpack(Vp_pair, nx2, nd2, self.j_buf, sigma, self.N_tot)
                    delta = R + self.Qa @ self.j_buf
            else:
                Vb = self.net_bf16(X.to(torch.bfloat16), self.ta, self.act_Mb).float()
                for _ in range(inner_jvp):
                    sc = sigma * torch.sqrt((nx2 + 1e-24) / ((delta * delta).sum() + 1e-30))
                    Vp = self.net_bf16((X + sc * delta).to(torch.bfloat16), self.ta, self.act_Mb).float()
                    delta = R + self.Qa @ ((Vp - Vb) / sc)
            X = X + delta
        out_buf.copy_(X[-1:])

    @torch.inference_mode()
    def solve_batch(
        self,
        x0: torch.Tensor,
        bf_init: int = 4,
        outer_ir: int = 2,
        inner_jvp: int = 1,
        sigma: float = 0.125,
    ) -> torch.Tensor:
        B, D = x0.shape
        M = self.M
        tz = torch.zeros(B, device=self.device, dtype=torch.float32)
        t_flat = self.ta.repeat(B)
        t_pair = torch.cat([t_flat, t_flat], dim=0)
        v0 = (
            self.net_bf16(x0.to(torch.bfloat16), tz).float()
            if outer_ir == 0
            else self.net_fp32(x0, tz)
        )
        base = x0[:, None, :] + self.q0[None, :, None] * v0[:, None, :]
        X = x0[:, None, :] + self.ta[None, :, None] * v0[:, None, :]
        Xh: list[torch.Tensor] = []
        Gh: list[torch.Tensor] = []
        for k in range(bf_init):
            V = self.net_bf16(X.reshape(B * M, D).to(torch.bfloat16), t_flat).float().view(B, M, D)
            G = base + torch.einsum("ij,bjd->bid", self.Qa, V)
            if outer_ir == 0 and k == bf_init - 1:
                return G[:, -1, :]
            Xh.append(X)
            Gh.append(G)
            if len(Xh) > 5:
                Xh.pop(0)
                Gh.pop(0)
            X = graph_safe_anderson_cgs2(Xh, Gh, m_max=4, reg=1e-5)

        for _ in range(outer_ir):
            V_ex = self.net_fp32(X.reshape(B * M, D), t_flat).view(B, M, D)
            R = (base + torch.einsum("ij,bjd->bid", self.Qa, V_ex)) - X
            xn = torch.linalg.norm(X, dim=(1, 2), keepdim=True) + 1e-12
            delta = R
            for _ in range(inner_jvp):
                sc = (sigma * xn) / (torch.linalg.norm(delta, dim=(1, 2), keepdim=True) + 1e-20)
                X_pair = (
                    torch.cat([X + sc * delta, X - sc * delta], dim=0)
                    .reshape(2 * B * M, D)
                    .to(torch.bfloat16)
                )
                V_pair = self.net_bf16(X_pair, t_pair).float().view(2, B, M, D)
                J_c = (V_pair[0] - V_pair[1]) / (2.0 * sc)
                delta = R + torch.einsum("ij,bjd->bid", self.Qa, J_c)
            X = X + delta
        return X[:, -1, :]

"""Log-SNR Desingularized Spectral Volterra & CSP-BF16-IR Solvers for Pretrained VP Diffusion.

Implements:
  1. `AnalyticVPSchedule`: C^infty continuous variance-preserving diffusion schedule.
  2. High-noise root-time `z = sqrt(tau)` desingularization (cancelling the 1/alpha_T = 152x
     BF16 quantization singularity and backward drift for SNR <= 1).
  3. Hybrid `z + exp` and pure `exp` `Log-SNR ParaFlow-BF16` and `Log-SNR CSP-BF16-IR` solvers,
     supporting arbitrary spatial resolutions (e.g. 32x32 CIFAR-10 and 256x256 CelebA-HQ).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from .volterra_chebyshev import (
    batched_cgs2_anderson,
    build_exp_chebyshev_panel,
    build_z_chebyshev_panel,
)


class AnalyticVPSchedule:
    """Exact C^infty continuous VP diffusion schedule corresponding to linear beta in [beta_0, beta_1]."""

    def __init__(self, beta_0: float = 0.1, beta_1: float = 20.0, eps_0: float = 1e-3):
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.eps_0 = eps_0
        self.lam_min = self.tau_to_lambda(1.0)
        self.lam_max = self.tau_to_lambda(eps_0)

    def B_int(self, tau):
        return self.beta_0 * tau + 0.5 * (self.beta_1 - self.beta_0) * (tau**2)

    def tau_to_alpha_sigma(self, tau):
        B = self.B_int(tau)
        return np.exp(-0.5 * B), np.sqrt(-np.expm1(-B))

    def tau_to_lambda(self, tau: float) -> float:
        a, s = self.tau_to_alpha_sigma(tau)
        return float(np.log(a / s))

    def lambda_to_tau_t_alpha_sigma(self, lam):
        lam = np.asarray(lam, dtype=np.float64)
        B = np.logaddexp(0.0, -2.0 * lam)
        tau = (
            -self.beta_0 + np.sqrt(self.beta_0**2 + 2.0 * (self.beta_1 - self.beta_0) * B)
        ) / (self.beta_1 - self.beta_0)
        t_unet = 999.0 * tau
        alpha = 1.0 / np.sqrt(1.0 + np.exp(-2.0 * lam))
        sigma = np.exp(-lam) * alpha
        return tau, t_unet, alpha, sigma

    def make_arclength_knots(self, P: int, gamma: float = 0.35) -> np.ndarray:
        taus = np.linspace(1.0, self.eps_0, 20001)
        B = self.B_int(taus)
        lams = -0.5 * B - 0.5 * np.log(-np.expm1(-B))
        arc = (lams - self.lam_min) / (self.lam_max - self.lam_min) + gamma * (1.0 - taus)
        targets = np.linspace(arc[0], arc[-1], P + 1)
        knots = np.interp(targets, arc, lams)
        knots[0], knots[-1] = self.lam_min, self.lam_max
        return knots


def precompute_hybrid_panels(
    vp: AnalyticVPSchedule,
    knots: Sequence[float],
    M: int = 8,
    use_z_below_zero: bool = True,
    device: str | torch.device = "cuda",
) -> list[dict]:
    """Build hybrid z=sqrt(tau) (for lam_b <= 0.05) + exp(-lam) Clenshaw-Curtis panels."""
    panels: list[dict] = []
    for p in range(len(knots) - 1):
        la, lb = float(knots[p]), float(knots[p + 1])
        if use_z_below_zero and lb <= 0.05:
            ta, _, _, _ = vp.lambda_to_tau_t_alpha_sigma(la)
            tb, _, _, _ = vp.lambda_to_tau_t_alpha_sigma(lb)
            za, zb = math.sqrt(float(ta)), math.sqrt(float(tb))
            zn, Qz = build_z_chebyshev_panel(za, zb, M, device)
            taus_n = zn**2
            tn = 999.0 * taus_n
            _, sig_n = vp.tau_to_alpha_sigma(taus_n)
            bet_n = vp.beta_0 + (vp.beta_1 - vp.beta_0) * taus_n
            panels.append(
                {
                    "mode": "z",
                    "M": M,
                    "t0": float(tn[0]),
                    "t_end": float(tn[-1]),
                    "t_act": torch.tensor(tn[1:], dtype=torch.float32, device=device),
                    "t_pair": torch.tensor(tn[1:], dtype=torch.float32, device=device).repeat(2),
                    "coef0": float(zn[0] * bet_n[0] / sig_n[0]),
                    "sig0": float(sig_n[0]),
                    "coef_act": torch.tensor(
                        zn[1:] * bet_n[1:] / sig_n[1:], dtype=torch.float32, device=device
                    )[None, :, None, None, None],
                    "sig_act": torch.tensor(sig_n[1:], dtype=torch.float32, device=device)[
                        None, :, None, None, None
                    ],
                    "q0": Qz[1:, 0][None, :, None, None, None],
                    "Qa": Qz[1:, 1:],
                    "dz": torch.tensor(zn[1:] - za, dtype=torch.float32, device=device)[
                        None, :, None, None, None
                    ],
                }
            )
        else:
            ln, Qe = build_exp_chebyshev_panel(la, lb, M, device)
            _, tn, an, _ = vp.lambda_to_tau_t_alpha_sigma(ln)
            panels.append(
                {
                    "mode": "exp",
                    "M": M,
                    "t0": float(tn[0]),
                    "t_end": float(tn[-1]),
                    "aa": float(an[0]),
                    "r_act": torch.tensor(
                        an[1:] / float(an[0]), dtype=torch.float32, device=device
                    )[None, :, None, None, None],
                    "a_act": torch.tensor(an[1:], dtype=torch.float32, device=device)[
                        None, :, None, None, None
                    ],
                    "t_act": torch.tensor(tn[1:], dtype=torch.float32, device=device),
                    "t_pair": torch.tensor(tn[1:], dtype=torch.float32, device=device).repeat(2),
                    "q0": Qe[1:, 0][None, :, None, None, None],
                    "Qa": Qe[1:, 1:],
                    "dh": torch.tensor(
                        np.exp(-ln[1:]) - math.exp(-la), dtype=torch.float32, device=device
                    )[None, :, None, None, None],
                }
            )
    return panels


def precompute_paraflow_panels(
    vp: AnalyticVPSchedule, M: int = 8, device: str | torch.device = "cuda"
) -> list[dict]:
    """Canonical 4-panel hybrid z+exp schedule for pure-BF16 Log-SNR ParaFlow-BF16."""
    knots = [vp.lam_min, -2.0, 0.0, 2.35, vp.lam_max]
    return precompute_hybrid_panels(vp, knots, M=M, use_z_below_zero=True, device=device)


def precompute_csp_ir_panels(
    vp: AnalyticVPSchedule,
    P: int = 9,
    M: int = 8,
    hybrid_z: bool = False,
    device: str | torch.device = "cuda",
) -> list[dict]:
    """Build P-panel arc-length schedule for Log-SNR CSP-BF16-IR."""
    if hybrid_z and P == 4:
        knots = [vp.lam_min, -2.0, 0.0, 2.35, vp.lam_max]
    elif hybrid_z and P == 6:
        knots = [vp.lam_min, -2.0, 0.0, 1.35, 2.75, 4.10, vp.lam_max]
    else:
        knots = list(vp.make_arclength_knots(P, gamma=0.35))
    return precompute_hybrid_panels(vp, knots, M=M, use_z_below_zero=hybrid_z, device=device)


def run_hybrid_csp_bf16_ir(
    x_init: torch.Tensor,
    panels: list[dict],
    unet_bf16: nn.Module,
    unet_fp32: nn.Module | None = None,
    bsw: int = 2,
    oir: int = 1,
    inner_jvp: int = 1,
    sigma_c: float = 0.125,
) -> torch.Tensor:
    """Unified arbitrary-resolution solver for both Log-SNR ParaFlow-BF16 (oir=0) and CSP-BF16-IR (oir>=1).

    Works identically for 32x32 CIFAR-10 and 256x256 CelebA-HQ images (`[B, C, H, W]`).
    """
    B_s, C, H, W = x_init.shape
    xa = x_init[:, None]
    ea: torch.Tensor | None = None

    for pan in panels:
        M = pan["M"]
        t_act_b = pan["t_act"].repeat(B_s)
        t_pair_b = pan["t_act"].repeat(2 * B_s)

        if pan["mode"] == "z":
            if ea is None:
                t0_b = torch.full((B_s,), pan["t0"], dtype=torch.float32, device=x_init.device)
                if oir == 0 or unet_fp32 is None:
                    ea = unet_bf16(
                        xa[:, 0].to(dtype=torch.bfloat16, memory_format=torch.channels_last), t0_b
                    ).float()[:, None]
                else:
                    ea = unet_fp32(
                        xa[:, 0].to(memory_format=torch.channels_last), t0_b
                    ).float()[:, None]
            f0 = pan["coef0"] * (ea - pan["sig0"] * xa)
            base = xa + pan["q0"] * f0
            X = xa + pan["dz"] * f0
            Xh: list[torch.Tensor] = []
            Gh: list[torch.Tensor] = []
            for k in range(bsw):
                Eb = (
                    unet_bf16(
                        X.reshape(B_s * M, C, H, W).to(
                            dtype=torch.bfloat16, memory_format=torch.channels_last
                        ),
                        t_act_b,
                    )
                    .float()
                    .reshape(B_s, M, C, H, W)
                )
                Fb = pan["coef_act"] * (Eb - pan["sig_act"] * X)
                G = base + torch.einsum("ij,bjchw->bichw", pan["Qa"], Fb)
                if oir == 0 and k == bsw - 1:
                    X = G
                    ea = Eb[:, -1:]
                    break
                Xh.append(X)
                Gh.append(G)
                X = batched_cgs2_anderson(Xh, Gh, m_max=3 if oir == 0 else 2, reg=2e-5)

            if oir > 0 and unet_fp32 is not None:
                for _ in range(oir):
                    Eex = (
                        unet_fp32(
                            X.reshape(B_s * M, C, H, W).to(memory_format=torch.channels_last),
                            t_act_b,
                        )
                        .float()
                        .reshape(B_s, M, C, H, W)
                    )
                    Fex = pan["coef_act"] * (Eex - pan["sig_act"] * X)
                    Gex = base + torch.einsum("ij,bjchw->bichw", pan["Qa"], Fex)
                    R = Gex - X
                    xn = torch.linalg.norm(X.reshape(B_s, -1), dim=1).view(B_s, 1, 1, 1, 1) + 1e-12
                    delta = R
                    for _ in range(inner_jvp):
                        sc = sigma_c * xn / (
                            torch.linalg.norm(delta.reshape(B_s, -1), dim=1).view(B_s, 1, 1, 1, 1)
                            + 1e-24
                        )
                        Xp = (
                            torch.cat([X + sc * delta, X - sc * delta], dim=0)
                            .reshape(2 * B_s * M, C, H, W)
                            .to(dtype=torch.bfloat16, memory_format=torch.channels_last)
                        )
                        Ep = unet_bf16(Xp, t_pair_b).float().reshape(2, B_s, M, C, H, W)
                        Jc = (Ep[0] - Ep[1]) / (2.0 * sc)
                        dF = pan["coef_act"] * (Jc - pan["sig_act"] * delta)
                        delta = R + torch.einsum("ij,bjchw->bichw", pan["Qa"], dF)
                    X = X + delta
                ea = Eex[:, -1:] + Jc[:, -1:]
            xa = X[:, -1:]

        else:
            if ea is None:
                t0_b = torch.full((B_s,), pan["t0"], dtype=torch.float32, device=x_init.device)
                if oir == 0 or unet_fp32 is None:
                    ea = unet_bf16(
                        xa[:, 0].to(dtype=torch.bfloat16, memory_format=torch.channels_last), t0_b
                    ).float()[:, None]
                else:
                    ea = unet_fp32(
                        xa[:, 0].to(memory_format=torch.channels_last), t0_b
                    ).float()[:, None]
            base = pan["r_act"] * xa + pan["a_act"] * (pan["q0"] * ea)
            X = pan["r_act"] * xa + pan["a_act"] * (pan["dh"] * ea)
            Xh = []
            Gh = []
            for k in range(bsw):
                Eb = (
                    unet_bf16(
                        X.reshape(B_s * M, C, H, W).to(
                            dtype=torch.bfloat16, memory_format=torch.channels_last
                        ),
                        t_act_b,
                    )
                    .float()
                    .reshape(B_s, M, C, H, W)
                )
                G = base + pan["a_act"] * torch.einsum("ij,bjchw->bichw", pan["Qa"], Eb)
                if oir == 0 and k == bsw - 1:
                    X = G
                    ea = Eb[:, -1:]
                    break
                Xh.append(X)
                Gh.append(G)
                X = batched_cgs2_anderson(Xh, Gh, m_max=3 if oir == 0 else 2, reg=2e-5)

            if oir > 0 and unet_fp32 is not None:
                for _ in range(oir):
                    Eex = (
                        unet_fp32(
                            X.reshape(B_s * M, C, H, W).to(memory_format=torch.channels_last),
                            t_act_b,
                        )
                        .float()
                        .reshape(B_s, M, C, H, W)
                    )
                    Gex = base + pan["a_act"] * torch.einsum("ij,bjchw->bichw", pan["Qa"], Eex)
                    R = Gex - X
                    xn = torch.linalg.norm(X.reshape(B_s, -1), dim=1).view(B_s, 1, 1, 1, 1) + 1e-12
                    delta = R
                    for _ in range(inner_jvp):
                        sc = sigma_c * xn / (
                            torch.linalg.norm(delta.reshape(B_s, -1), dim=1).view(B_s, 1, 1, 1, 1)
                            + 1e-24
                        )
                        Xp = (
                            torch.cat([X + sc * delta, X - sc * delta], dim=0)
                            .reshape(2 * B_s * M, C, H, W)
                            .to(dtype=torch.bfloat16, memory_format=torch.channels_last)
                        )
                        Ep = unet_bf16(Xp, t_pair_b).float().reshape(2, B_s, M, C, H, W)
                        Jc = (Ep[0] - Ep[1]) / (2.0 * sc)
                        delta = R + pan["a_act"] * torch.einsum("ij,bjchw->bichw", pan["Qa"], Jc)
                    X = X + delta
                ea = Eex[:, -1:] + Jc[:, -1:]
            xa = X[:, -1:]

    return xa[:, 0].contiguous()


def precompute_paradigms_windows(
    vp: AnalyticVPSchedule,
    S: int = 96,
    W: int = 8,
    device: str | torch.device = "cuda",
) -> list[dict]:
    """Build sliding-window Picard coefficients in half-log-SNR space for ParaDiGMS-BF16."""
    lg = np.linspace(vp.lam_min, vp.lam_max, S + 1)
    _, tg, ag, _ = vp.lambda_to_tau_t_alpha_sigma(lg)
    hg = np.exp(-lg)
    windows: list[dict] = []
    for s0 in range(0, S, W):
        w = min(W, S - s0)
        windows.append(
            {
                "w": w,
                "t0": float(tg[0]),
                "a_s0": float(ag[s0]),
                "a_in": torch.tensor(ag[s0 : s0 + w], dtype=torch.float32, device=device)[
                    None, :, None, None, None
                ],
                "dh_init": torch.tensor(
                    hg[s0 : s0 + w] - hg[s0], dtype=torch.float32, device=device
                )[None, :, None, None, None],
                "a_out": torch.tensor(ag[s0 + 1 : s0 + w + 1], dtype=torch.float32, device=device)[
                    None, :, None, None, None
                ],
                "dh": torch.tensor(
                    hg[s0 + 1 : s0 + w + 1] - hg[s0 : s0 + w], dtype=torch.float32, device=device
                )[None, :, None, None, None],
                "t_win": torch.tensor(tg[s0 : s0 + w], dtype=torch.float32, device=device),
            }
        )
    return windows


def run_paradigms_bf16(
    x_init: torch.Tensor,
    windows: list[dict],
    unet_bf16: nn.Module,
    sweeps: int = 3,
) -> torch.Tensor:
    """Execute sliding-window Picard iterations (`ParaDiGMS-BF16`) for arbitrary `[B, C, H, W]`."""
    B_s, C, H, W = x_init.shape
    xc = x_init[:, None]
    t0_b = torch.full((B_s,), windows[0]["t0"], dtype=torch.float32, device=x_init.device)
    e_prev = unet_bf16(
        xc[:, 0].to(dtype=torch.bfloat16, memory_format=torch.channels_last), t0_b
    ).float()[:, None]
    for win in windows:
        w = win["w"]
        t_b = win["t_win"].repeat(B_s)
        X_in = (win["a_in"] / win["a_s0"]) * xc + win["a_in"] * (win["dh_init"] * e_prev)
        for k in range(sweeps):
            Ew = (
                unet_bf16(
                    X_in.reshape(B_s * w, C, H, W).to(
                        dtype=torch.bfloat16, memory_format=torch.channels_last
                    ),
                    t_b,
                )
                .float()
                .reshape(B_s, w, C, H, W)
            )
            csum = torch.cumsum(win["dh"] * Ew, dim=1)
            X_next = (win["a_out"] / win["a_s0"]) * xc + win["a_out"] * csum
            if k < sweeps - 1 and w > 1:
                X_in = torch.cat([xc, X_next[:, :-1]], dim=1)
        e_prev = Ew[:, -1:]
        xc = X_next[:, -1:]
    return xc[:, 0].contiguous()


def build_ddim_schedule(vp: AnalyticVPSchedule, S: int) -> list[tuple[float, float, float]]:
    lg = np.linspace(vp.lam_min, vp.lam_max, S + 1)
    _, tg, ag, _ = vp.lambda_to_tau_t_alpha_sigma(lg)
    hg = np.exp(-lg)
    return [
        (float(tg[i]), float(ag[i + 1] / ag[i]), float(ag[i + 1] * (hg[i + 1] - hg[i])))
        for i in range(S)
    ]


def build_dpm_solver2_schedule(
    vp: AnalyticVPSchedule, S_nfe: int
) -> list[tuple[float, float, float, float, float, float]]:
    n_st = S_nfe // 2
    lg = np.linspace(vp.lam_min, vp.lam_max, n_st + 1)
    steps: list[tuple[float, float, float, float, float, float]] = []
    for i in range(n_st):
        la, lb = lg[i], lg[i + 1]
        lm = 0.5 * (la + lb)
        h = lb - la
        _, t_v, a_v, s_v = vp.lambda_to_tau_t_alpha_sigma([la, lm, lb])
        aa, am, ab = float(a_v[0]), float(a_v[1]), float(a_v[2])
        sm, sb = float(s_v[1]), float(s_v[2])
        steps.append(
            (
                float(t_v[0]),
                float(t_v[1]),
                am / aa,
                -sm * math.expm1(0.5 * h),
                ab / aa,
                -sb * math.expm1(h),
            )
        )
    return steps

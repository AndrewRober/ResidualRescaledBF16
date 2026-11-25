"""Stage C Benchmark: Trained Flow-Matching Generative Models (8-Mode OT-CFM & 2D Spatial Glyph DiT)."""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from residual_rescaled_bf16 import (
    PTXCspBf16Solver,
    PTXDiTBlock,
    Spatial2DGlyphDiT,
    capture_cudagraph,
    gpu_paradigms_body,
    gpu_seq_euler_body,
    gpu_seq_rk4_body,
)


class Manifold8ModeDiT(nn.Module):
    """33.6M-parameter SwiGLU adaLN Flow-Matching DiT on R^128."""

    def __init__(self, d_data: int = 128, d_model: int = 1024, d_ff: int = 2816, n_blocks: int = 3):
        super().__init__()
        self.d_data = d_data
        self.d_ff = d_ff
        self.register_buffer(
            "freqs", torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=torch.float32) * math.pi
        )
        self.in_proj = nn.Linear(d_data, d_model)
        self.t_mlp = nn.Sequential(
            nn.Linear(12, d_model), nn.SiLU(), nn.Linear(d_model, d_model), nn.SiLU()
        )
        self.blocks = nn.ModuleList(
            [PTXDiTBlock(d_model, d_ff, res_scale=1.0, zero_init=True) for _ in range(n_blocks)]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_data)

    def forward(self, x: torch.Tensor, t: torch.Tensor, act_buf: torch.Tensor | None = None) -> torch.Tensor:
        dtp = x.dtype
        ang = t[:, None].to(self.freqs.dtype) * self.freqs[None, :]
        phi = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1).to(dtp)
        t_emb = self.t_mlp(phi)
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h, t_emb, act_buf)
        return self.out_proj(self.out_norm(h))


def sample_8mode_manifold(n: int, d: int = 128, device: str = "cuda") -> torch.Tensor:
    idx = torch.randint(0, 8, (n,), device=device)
    ang = idx.float() * (2.0 * math.pi / 8.0) + torch.randn(n, device=device) * 0.08
    r = 2.4 + 0.18 * torch.sin(3.0 * ang)
    c0 = r * torch.cos(ang)
    c1 = r * torch.sin(ang)
    c2 = 0.65 * torch.sin(2.0 * ang)
    c3 = 0.65 * torch.cos(2.0 * ang)
    low = torch.stack([c0, c1, c2, c3], dim=1)
    gen = torch.Generator(device=device).manual_seed(31415)
    basis = torch.linalg.qr(torch.randn(d, 4, generator=gen, device=device), mode="reduced")[0]
    return low @ basis.T + torch.randn(n, d, device=device) * 0.05


def build_2d_glyph_sampler(hw: int = 16, device: str = "cuda"):
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, hw, device=device),
        torch.linspace(-1.0, 1.0, hw, device=device),
        indexing="ij",
    )

    def sample_glyphs(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        cls = torch.randint(0, 4, (n,), device=device)
        cx = (torch.rand(n, 1, 1, device=device) - 0.5) * 0.24
        cy = (torch.rand(n, 1, 1, device=device) - 0.5) * 0.24
        dx = xx[None, :, :] - cx
        dy = yy[None, :, :] - cy
        sharp = 9.0

        d_ring = (torch.sqrt(dx * dx + dy * dy + 1e-6) - 0.58).abs() - 0.16
        d_plus = torch.minimum(dx.abs(), dy.abs()) - 0.16
        mask_box = torch.maximum(dx.abs(), dy.abs()) - 0.72
        d_plus = torch.maximum(d_plus, mask_box)
        d_x = torch.minimum((dx - dy).abs(), (dx + dy).abs()) * 0.7071 - 0.15
        d_x = torch.maximum(d_x, mask_box)
        d_diam = (dx.abs() + dy.abs() - 0.72).abs() - 0.15

        sdf = torch.where(
            cls[:, None, None] == 0,
            d_ring,
            torch.where(cls[:, None, None] == 1, d_plus, torch.where(cls[:, None, None] == 2, d_x, d_diam)),
        )
        imgs = torch.tanh(-sharp * sdf) + torch.randn(n, hw, hw, device=device) * 0.03
        return imgs.reshape(n, hw * hw), cls

    return sample_glyphs


def sliced_wasserstein_2(X: torch.Tensor, Y: torch.Tensor, n_proj: int = 256) -> float:
    gen = torch.Generator(device=X.device).manual_seed(999)
    dirs = torch.randn(X.shape[1], n_proj, generator=gen, device=X.device)
    dirs = dirs / (torch.linalg.norm(dirs, dim=0, keepdim=True) + 1e-12)
    px, _ = torch.sort(X @ dirs, dim=0)
    py, _ = torch.sort(Y @ dirs, dim=0)
    return float(torch.sqrt(torch.mean((px - py) ** 2)).cpu())


def rbf_mmd(X: torch.Tensor, Y: torch.Tensor, bw: float = 2.0) -> float:
    k_xx = torch.exp(-torch.cdist(X, X).pow(2) / (2 * bw * bw)).mean()
    k_yy = torch.exp(-torch.cdist(Y, Y).pow(2) / (2 * bw * bw)).mean()
    k_xy = torch.exp(-torch.cdist(X, Y).pow(2) / (2 * bw * bw)).mean()
    return float(torch.sqrt(torch.clamp(k_xx + k_yy - 2.0 * k_xy, min=0.0)).cpu())


def image_edge_rel_l2(X_gen: torch.Tensor, X_ref: torch.Tensor, hw: int = 16) -> float:
    g = X_gen.view(-1, hw, hw)
    r = X_ref.view(-1, hw, hw)
    gx_g, gy_g = g[:, :, 1:] - g[:, :, :-1], g[:, 1:, :] - g[:, :-1, :]
    gx_r, gy_r = r[:, :, 1:] - r[:, :, :-1], r[:, 1:, :] - r[:, :-1, :]
    num = torch.linalg.norm((gx_g - gx_r).reshape(-1)) ** 2 + torch.linalg.norm((gy_g - gy_r).reshape(-1)) ** 2
    den = torch.linalg.norm(gx_r.reshape(-1)) ** 2 + torch.linalg.norm(gy_r.reshape(-1)) ** 2 + 1e-24
    return float(torch.sqrt(num / den).cpu())


def train_ot_cfm(model: nn.Module, sample_fn, d_dim: int, steps: int = 210) -> float:
    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-4, fused=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-4)
    half_pi = 0.5 * math.pi
    t0 = time.perf_counter()
    model.train()
    for _ in range(steps):
        x0 = torch.randn(256, d_dim, device="cuda")
        x1 = sample_fn(256)
        with torch.no_grad():
            cost = torch.cdist(x0, x1, p=2.0).pow(2).cpu().numpy()
            _, col_ind = linear_sum_assignment(cost)
            x1 = x1[torch.as_tensor(col_ind, device="cuda", dtype=torch.long)]
        t = torch.rand(256, device="cuda")
        c, s = torch.cos(half_pi * t)[:, None], torch.sin(half_pi * t)[:, None]
        xt = c * x0 + s * x1
        ut = half_pi * (-s * x0 + c * x1)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = F.mse_loss(model(xt, t).float(), ut)
        loss.backward()
        opt.step()
        sched.step()
    torch.cuda.synchronize()
    model.eval()
    return time.perf_counter() - t0


def time_fn(fn, iters: int = 9) -> float:
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(float(e0.elapsed_time(e1)))
    ts.sort()
    return ts[len(ts) // 2]


def main() -> None:
    tot_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    torch.cuda.set_per_process_memory_fraction(0.25 if tot_gb > 32 else 0.40)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    gpu_name = torch.cuda.get_device_name(0)
    print(f"=== Stage C: Flow-Matching Generative Models on {gpu_name} ===")

    # 1. 33.6M 8-Mode Curved Manifold OT-CFM in R^128
    torch.manual_seed(101)
    m_ot = Manifold8ModeDiT(128, 1024, 2816, 3).to("cuda")
    ot_train_s = train_ot_cfm(m_ot, lambda n: sample_8mode_manifold(n, 128, "cuda"), 128, steps=200)
    m_ot_bf16 = copy.deepcopy(m_ot).to(torch.bfloat16).eval()
    sol_ot = PTXCspBf16Solver(m_ot_bf16, m_ot, M=16, d_state=128, h_ff=2816, device="cuda")

    with torch.inference_mode():
        z256 = torch.randn(256, 128, generator=torch.Generator(device="cuda").manual_seed(55), device="cuda")
        real_ot1 = sample_8mode_manifold(256, 128, "cuda")
        real_ot2 = sample_8mode_manifold(256, 128, "cuda")
        xref_ot = torch.empty_like(z256)
        gpu_seq_rk4_body(m_ot, z256, xref_ot, S_nfe=512)
        ref_ot_n = float(torch.linalg.norm(xref_ot).cpu())

        x1_ot = z256[0:1].clone()
        o1_ot = torch.empty_like(x1_ot)
        x_pop = torch.empty_like(z256)

        def gpu_heun(net, x0_b, out_b, S_nfe=128):
            st = S_nfe // 2
            dt = 1.0 / st
            x = x0_b.clone()
            tz = torch.zeros(x.shape[0], device="cuda")
            for s in range(st):
                t = tz + s * dt
                k1 = net(x, t)
                k2 = net(x + dt * k1, t + dt)
                x = x + 0.5 * dt * (k1 + k2)
            out_b.copy_(x)

        ot_samplers = {}
        for name, b1_fn, pop_fn in [
            ("GPU Euler-128 (FP32)", lambda: gpu_seq_euler_body(m_ot, x1_ot, o1_ot, 128), lambda: (gpu_seq_euler_body(m_ot, z256, x_pop, 128), x_pop)[1]),
            ("GPU Heun-128 (FP32)", lambda: gpu_heun(m_ot, x1_ot, o1_ot, 128), lambda: (gpu_heun(m_ot, z256, x_pop, 128), x_pop)[1]),
            ("GPU RK4-256 (FP32)", lambda: gpu_seq_rk4_body(m_ot, x1_ot, o1_ot, 256), lambda: (gpu_seq_rk4_body(m_ot, z256, x_pop, 256), x_pop)[1]),
            ("GPU ParaDiGMS-128 (BF16, W=32)", lambda: gpu_paradigms_body(m_ot_bf16, x1_ot, o1_ot, 128, 32, 4), lambda: (gpu_paradigms_body(m_ot_bf16, z256, x_pop, 128, 32, 4), x_pop)[1]),
            ("ParaFlow-BF16 (M=16, 5+0)", lambda: sol_ot.run_b1_into(x1_ot, o1_ot, bf_init=5, outer_ir=0), lambda: sol_ot.solve_batch(z256, bf_init=5, outer_ir=0)),
            ("SP-BF16-IR (M=16, 4+2)", lambda: sol_ot.run_b1_into(x1_ot, o1_ot, bf_init=4, outer_ir=2, inner_jvp=2), lambda: sol_ot.solve_batch(z256, bf_init=4, outer_ir=2, inner_jvp=2)),
        ]:
            b1_ms = time_fn(b1_fn, iters=7)
            pop_ms = time_fn(pop_fn, iters=3)
            Xb = pop_fn().clone()
            ot_samplers[name] = {
                "b1_latency_ms": round(b1_ms, 3),
                "pop256_latency_ms": round(pop_ms, 3),
                "sliced_wasserstein_2": round(sliced_wasserstein_2(Xb, real_ot1), 5),
                "rbf_mmd": round(rbf_mmd(Xb, real_ot1), 5),
                "traj_rel_l2_vs_rk4_512": float(torch.linalg.norm(Xb - xref_ot).cpu()) / ref_ot_n,
            }
            print(f"[OT-CFM] {name:32s} | B=1 {b1_ms:6.2f} ms | Rel L2 = {ot_samplers[name]['traj_rel_l2_vs_rk4_512']:.3e}")

    # 2. 34.65M 2D Spatial Glyph DiT (16x16 Structured Images)
    torch.manual_seed(2026)
    m_gl = Spatial2DGlyphDiT(16, 1024, 2816, 3).to("cuda")
    gl_sample_fn = build_2d_glyph_sampler(16, "cuda")
    gl_train_s = train_ot_cfm(m_gl, lambda n: gl_sample_fn(n)[0], 256, steps=210)
    m_gl_bf16 = copy.deepcopy(m_gl).to(torch.bfloat16).eval()
    sol_gl = PTXCspBf16Solver(m_gl_bf16, m_gl, M=12, d_state=256, h_ff=2816, device="cuda")

    with torch.inference_mode():
        z128 = torch.randn(128, 256, generator=torch.Generator(device="cuda").manual_seed(77), device="cuda")
        real_gl1, _ = gl_sample_fn(128)
        real_gl2, _ = gl_sample_fn(128)
        xref_gl = torch.empty_like(z128)
        gpu_seq_rk4_body(m_gl, z128, xref_gl, S_nfe=512)
        ref_gl_n = float(torch.linalg.norm(xref_gl).cpu())

        x1_gl = z128[0:1].clone()
        o1_gl = torch.empty_like(x1_gl)

        def eval_gl(b1_fn, use_cg: bool, Xb: torch.Tensor) -> dict:
            if use_cg:
                g = capture_cudagraph(b1_fn)
                b1_ms = time_fn(lambda: g.replay(), iters=15)
                del g
            else:
                b1_ms = time_fn(b1_fn, iters=5)
            mse = float(torch.mean((Xb - xref_gl) ** 2).cpu()) + 1e-30
            psnr = 10.0 * math.log10(4.0 / mse)
            return {
                "b1_latency_ms": round(b1_ms, 3),
                "psnr_vs_exact_ode_db": round(psnr, 2),
                "edge_gradient_rel_l2": image_edge_rel_l2(Xb, xref_gl, 16),
                "image_rel_l2_vs_exact_ode": float(torch.linalg.norm(Xb - xref_gl).cpu()) / ref_gl_n,
                "sliced_wasserstein_2": round(sliced_wasserstein_2(Xb, real_gl1), 5),
            }

        X_eu = torch.empty_like(z128)
        gpu_seq_euler_body(m_gl, z128, X_eu, 128)
        X_rk = torch.empty_like(z128)
        gpu_seq_rk4_body(m_gl, z128, X_rk, 256)
        X_pd = torch.empty_like(z128)
        gpu_paradigms_body(m_gl_bf16, z128, X_pd, 128, 32, 4)
        X_pf = sol_gl.solve_batch(z128, bf_init=5, outer_ir=0)
        X_csp = sol_gl.solve_batch(z128, bf_init=4, outer_ir=2, inner_jvp=2)

        gl_samplers = {
            "Eager Euler-128 (FP32)": eval_gl(lambda: gpu_seq_euler_body(m_gl, x1_gl, o1_gl, 128), False, X_eu),
            "CUDAGraph RK4-256 (FP32)": eval_gl(lambda: gpu_seq_rk4_body(m_gl, x1_gl, o1_gl, 256), True, X_rk),
            "CUDAGraph ParaDiGMS-128 (BF16)": eval_gl(lambda: gpu_paradigms_body(m_gl_bf16, x1_gl, o1_gl, 128, 32, 4), True, X_pd),
            "PTX+CG ParaFlow-BF16 (5,0,0)": eval_gl(lambda: sol_gl.run_b1_into(x1_gl, o1_gl, bf_init=5, outer_ir=0), True, X_pf),
            "PTX+CG CSP-BF16-IR (4,2,2)": eval_gl(lambda: sol_gl.run_b1_into(x1_gl, o1_gl, bf_init=4, outer_ir=2, inner_jvp=2), True, X_csp),
        }
        for k, v in gl_samplers.items():
            print(f"[2D Glyph] {k:32s} | B=1 {v['b1_latency_ms']:6.2f} ms | PSNR = {v['psnr_vs_exact_ode_db']:.2f} dB")

    out_json = ROOT / "results" / "stage_c_flow_matching.json"
    out_json.write_text(
        json.dumps(
            {
                "stage": "Stage C — Trained Flow-Matching Generative Models (33.6M 8-Mode OT-CFM & 34.65M 2D Spatial Glyph DiT)",
                "gpu_device": gpu_name,
                "pytorch_version": torch.__version__,
                "ot_cfm_8mode_r128": {
                    "model_params_m": round(sum(p.numel() for p in m_ot.parameters()) / 1e6, 2),
                    "train_wall_s": round(ot_train_s, 2),
                    "oracle_holdout_swd": round(sliced_wasserstein_2(real_ot1, real_ot2), 5),
                    "oracle_holdout_rbf_mmd": round(rbf_mmd(real_ot1, real_ot2), 5),
                    "samplers": ot_samplers,
                },
                "spatial_2d_glyph_dit_16x16": {
                    "model_params_m": round(sum(p.numel() for p in m_gl.parameters()) / 1e6, 2),
                    "train_wall_s": round(gl_train_s, 2),
                    "oracle_real_vs_real_swd": round(sliced_wasserstein_2(real_gl1, real_gl2), 5),
                    "samplers": gl_samplers,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[Done] Saved {out_json}")


if __name__ == "__main__":
    main()

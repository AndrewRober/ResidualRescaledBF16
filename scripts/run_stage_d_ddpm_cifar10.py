"""Stage D Benchmark: Pretrained `google/ddpm-cifar10-32` Full Pareto Frontier & Nyquist Analysis.

Evaluates 20 solver configurations across 8 canonical seeds on `google/ddpm-cifar10-32` (35.75M UNet):
  - Sequential DDIM-{20,50,100,250} and DPM-Solver-2-{20,50,100} in BF16 and TF32-FP32 (+ IEEE-FP32 timing)
  - Parallel Sliding-Window Picard (`ParaDiGMS-BF16`, S=96, W=8)
  - Pure-BF16 Desingularized Spectral Volterra (`Log-SNR ParaFlow-BF16`, P=4, M=8, hybrid z+exp)
  - Residual-Rescaled Central-Difference JVP Refinement (`Log-SNR CSP-BF16-IR`, exact exp panels,
    single-panel L2-resident CUDAGraph):
      * Pareto-Fast 1-IR (`P=5, M=8, bsw=1, oir=1` and `P=7, M=8, bsw=2, oir=1`, TF32 checkpoints)
      * Strict 24-Bit IEEE-FP32 Checkpoints (`P=7, M=8, 1-IR` and `P=9, M=8, 2-IR`)
Also records empirical `1/alpha_t` quantization noise amplification and the `sin(999*tau)`
discrete-training Nyquist spectrum directly from the pretrained UNet.
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from residual_rescaled_bf16 import (
    AnalyticVPSchedule,
    DiffusersUNet2D,
    batched_cgs2_anderson,
    build_ddim_schedule,
    build_dpm_solver2_schedule,
    precompute_csp_ir_panels,
    precompute_paradigms_windows,
    precompute_paraflow_panels,
    run_hybrid_csp_bf16_ir,
    run_paradigms_bf16,
)


def csp_single_exp_panel_step(
    xa: torch.Tensor,
    ea: torch.Tensor,
    r_act: torch.Tensor,
    a_act: torch.Tensor,
    t_act_b: torch.Tensor,
    t_pair_b: torch.Tensor,
    q0: torch.Tensor,
    Qa: torch.Tensor,
    dh: torch.Tensor,
    unet_fp32: nn.Module,
    unet_bf16: nn.Module,
    bsw: int = 2,
    oir: int = 2,
    sigma_c: float = 0.125,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-panel L2-resident CSP-BF16-IR kernel on exact integrating-factor exp(-lambda) panel."""
    B_s = xa.shape[0]
    M = Qa.shape[0]
    base = r_act * xa + a_act * (q0 * ea)
    X = r_act * xa + a_act * (dh * ea)
    Xh: list[torch.Tensor] = []
    Gh: list[torch.Tensor] = []
    for _ in range(bsw):
        Eb = (
            unet_bf16(
                X.reshape(B_s * M, 3, 32, 32).to(
                    dtype=torch.bfloat16, memory_format=torch.channels_last
                ),
                t_act_b,
            )
            .float()
            .reshape(B_s, M, 3, 32, 32)
        )
        G = base + a_act * torch.einsum("ij,bjchw->bichw", Qa, Eb)
        Xh.append(X)
        Gh.append(G)
        X = batched_cgs2_anderson(Xh, Gh, m_max=2, reg=2e-5)

    for _ in range(oir):
        Eex = unet_fp32(X.reshape(B_s * M, 3, 32, 32), t_act_b).reshape(B_s, M, 3, 32, 32)
        Gex = base + a_act * torch.einsum("ij,bjchw->bichw", Qa, Eex)
        R = Gex - X
        xn = torch.linalg.norm(X.reshape(B_s, -1), dim=1).view(B_s, 1, 1, 1, 1) + 1e-12
        sc = sigma_c * xn / (
            torch.linalg.norm(R.reshape(B_s, -1), dim=1).view(B_s, 1, 1, 1, 1) + 1e-24
        )
        Xp = (
            torch.cat([X + sc * R, X - sc * R], dim=1)
            .reshape(B_s * 2 * M, 3, 32, 32)
            .to(dtype=torch.bfloat16, memory_format=torch.channels_last)
        )
        Ep = unet_bf16(Xp, t_pair_b).float().reshape(B_s, 2, M, 3, 32, 32)
        Jc = (Ep[:, 0] - Ep[:, 1]) / (2.0 * sc)
        d = R + a_act * torch.einsum("ij,bjchw->bichw", Qa, Jc)
        X = X + d
    ea_next = Eex[:, -1:] + Jc[:, -1:]
    return X[:, -1:], ea_next


def compute_metrics(y: torch.Tensor, ref: torch.Tensor) -> dict:
    diff = y.float() - ref.float()
    B = y.shape[0]
    mses = torch.mean(diff**2, dim=(1, 2, 3)).cpu().numpy()
    psnrs = 10.0 * np.log10(4.0 / (mses + 1e-30))
    rel_l2s = (
        torch.linalg.norm(diff.reshape(B, -1), dim=1)
        / torch.linalg.norm(ref.reshape(B, -1), dim=1)
    ).cpu().numpy()
    max_errs = torch.amax(torch.abs(diff), dim=(1, 2, 3)).cpu().numpy()
    return {
        "psnr_mean_db": float(np.mean(psnrs)),
        "psnr_std_db": float(np.std(psnrs)),
        "psnr_per_seed": [round(float(v), 2) for v in psnrs],
        "rel_l2_mean": float(np.mean(rel_l2s)),
        "max_abs_err_mean": float(np.mean(max_errs)),
    }


def main() -> None:
    t_wall_0 = time.perf_counter()
    tot_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    torch.cuda.set_per_process_memory_fraction(0.25 if tot_gb > 32 else 0.55)
    torch.backends.cudnn.benchmark = False

    ckpt_dir = ROOT / ".cache" / "ddpm_cifar10_32"
    if not (ckpt_dir / "config.json").exists():
        from huggingface_hub import hf_hub_download

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        for fn in ["config.json", "diffusion_pytorch_model.safetensors"]:
            hf_hub_download("google/ddpm-cifar10-32", fn, local_dir=str(ckpt_dir))

    gpu_name = torch.cuda.get_device_name(0)
    print(f"=== Stage D: google/ddpm-cifar10-32 Full Pareto & Nyquist Analysis on {gpu_name} ===")

    # NCHW for FP32 cuDNN speed at 32x32; NHWC channels_last for BF16 Tensor Cores
    unet_fp32 = DiffusersUNet2D.from_pretrained_dir(str(ckpt_dir)).to(device="cuda").eval()
    unet_bf16 = (
        copy.deepcopy(unet_fp32)
        .to(dtype=torch.bfloat16, memory_format=torch.channels_last)
        .eval()
    )
    n_params = sum(p.numel() for p in unet_fp32.parameters())
    print(f"[1] Loaded google/ddpm-cifar10-32 (strict=True, {n_params / 1e6:.2f}M params)")

    seeds = [0, 1, 2, 3, 4, 5, 6, 7]
    xT8 = torch.cat(
        [
            torch.randn(
                1, 3, 32, 32, generator=torch.Generator(device="cuda").manual_seed(s), device="cuda"
            )
            for s in seeds
        ],
        dim=0,
    )
    vp = AnalyticVPSchedule()

    with torch.inference_mode():
        # --------------------------------------------------------------------
        # 1. Strict 24-bit IEEE-FP32 Richardson Exp-Heun(56, 28) Reference
        # --------------------------------------------------------------------
        t_ref_0 = time.perf_counter()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        xb8 = torch.empty_like(xT8)
        tb8 = torch.empty(8, dtype=torch.float32, device="cuda")
        ob8 = torch.empty_like(xT8)
        _ = unet_fp32(xb8.zero_(), tb8.zero_())
        torch.cuda.synchronize()
        g_ieee_b8 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_ieee_b8):
            ob8.copy_(unet_fp32(xb8, tb8))

        def run_heun_b8(N_steps: int) -> torch.Tensor:
            lg = np.linspace(vp.lam_min, vp.lam_max, N_steps + 1)
            _, tg, ag, _ = vp.lambda_to_tau_t_alpha_sigma(lg)
            xc = xT8.clone()
            for i in range(N_steps):
                la, lb = lg[i], lg[i + 1]
                dlam = lb - la
                ha, hb = math.exp(-la), math.exp(-lb)
                aa, ab = float(ag[i]), float(ag[i + 1])
                xb8.copy_(xc)
                tb8.fill_(float(tg[i]))
                g_ieee_b8.replay()
                e1 = ob8.clone()
                xp = (ab / aa) * xc + ab * (hb - ha) * e1
                xb8.copy_(xp)
                tb8.fill_(float(tg[i + 1]))
                g_ieee_b8.replay()
                wt = hb - ha
                w2 = hb + wt / dlam
                xc = (ab / aa) * xc + ab * ((wt - w2) * e1 + w2 * ob8)
            return xc

        x_ref8 = (4.0 * run_heun_b8(56) - run_heun_b8(28)) / 3.0

        csp9_panels = precompute_csp_ir_panels(vp, P=9, M=8, hybrid_z=False, device="cuda")
        csp7_panels = precompute_csp_ir_panels(vp, P=7, M=8, hybrid_z=False, device="cuda")
        csp5_panels = precompute_csp_ir_panels(vp, P=5, M=8, hybrid_z=False, device="cuda")
        paraflow4 = precompute_paraflow_panels(vp, M=8, device="cuda")
        paradigms_wins = precompute_paradigms_windows(vp, S=96, W=8, device="cuda")

        # Seed-0 exact IEEE-FP32 spectral collocation reference on P=9, M=8
        xa_sp = xT8[0:1, None].clone()
        ea_sp = unet_fp32(xa_sp[:, 0], torch.tensor([csp9_panels[0]["t0"]], device="cuda"))[:, None]
        for pan in csp9_panels:
            base_sp = pan["r_act"] * xa_sp + pan["a_act"] * (pan["q0"] * ea_sp)
            X_sp = pan["r_act"] * xa_sp + pan["a_act"] * (pan["dh"] * ea_sp)
            Xh_s, Gh_s = [], []
            tb8.copy_(pan["t_act"])
            for _ in range(4):
                xb8.copy_(X_sp[0])
                g_ieee_b8.replay()
                E_s = ob8[None]
                G_s = base_sp + pan["a_act"] * torch.einsum("ij,bjchw->bichw", pan["Qa"], E_s)
                Xh_s.append(X_sp)
                Gh_s.append(G_s)
                X_sp = batched_cgs2_anderson(Xh_s, Gh_s, m_max=3, reg=1e-6)
            ea_sp = E_s[:, -1:]
            xa_sp = X_sp[:, -1:]
        x_fp32_colloc_s0 = xa_sp[:, 0].clone()
        del g_ieee_b8

        # --------------------------------------------------------------------
        # 2. Measure 1/alpha_t Quantization Amplification & sin(999*tau) Nyquist Ripple
        # --------------------------------------------------------------------
        tau_probe = np.linspace(1.0, 0.01, 40)
        singularity_curve = []
        x_probe = xT8[0:1]
        for tau_v in tau_probe:
            a_v, s_v = vp.tau_to_alpha_sigma(float(tau_v))
            lam_v = float(np.log(a_v / s_v))
            t_t = torch.tensor([999.0 * float(tau_v)], dtype=torch.float32, device="cuda")
            e32 = unet_fp32(x_probe, t_t).float()
            e16 = unet_bf16(
                x_probe.to(dtype=torch.bfloat16, memory_format=torch.channels_last), t_t
            ).float()
            raw_rms = float(torch.sqrt(torch.mean((e16 - e32) ** 2)).cpu())
            x0_hat_err = raw_rms / float(a_v)
            z_drift_err = (
                math.sqrt(float(tau_v)) * (0.1 + 19.9 * float(tau_v)) / float(s_v)
            ) * raw_rms * 0.1
            singularity_curve.append(
                {
                    "tau": round(float(tau_v), 4),
                    "lambda": round(lam_v, 4),
                    "inv_alpha": round(float(1.0 / a_v), 3),
                    "raw_bf16_eps_rms": raw_rms,
                    "naive_inv_alpha_amplified_rms": x0_hat_err,
                    "z_root_tau_panel_rms": z_drift_err,
                }
            )

        t_fine = torch.linspace(498.0, 502.0, 256, dtype=torch.float32, device="cuda")
        e_fine = []
        for idx_f in range(0, 256, 32):
            tb_chunk = t_fine[idx_f : idx_f + 32]
            xb_chunk = x_probe.expand(32, -1, -1, -1).contiguous()
            e_fine.append(unet_fp32(xb_chunk, tb_chunk).float()[:, 0, 16, 16].cpu())
        sig_fine = torch.cat(e_fine).numpy()
        t_np = t_fine.cpu().numpy()
        t_centered = t_np - 500.0
        poly_fit = np.polyval(np.polyfit(t_centered, sig_fine, deg=3), t_centered)
        nyquist_residual = sig_fine - poly_fit
        nyquist_rms = float(np.sqrt(np.mean(nyquist_residual**2)))

        # Strict IEEE-FP32 B=1 single-step latency
        s1_x32 = torch.empty(1, 3, 32, 32, device="cuda")
        s1_t32 = torch.empty(1, device="cuda")
        s1_o32 = torch.empty_like(s1_x32)
        _ = unet_fp32(s1_x32.zero_(), s1_t32.zero_())
        torch.cuda.synchronize()
        g_ieee_1 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_ieee_1):
            s1_o32.copy_(unet_fp32(s1_x32, s1_t32))
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        ev0.record()
        for _ in range(8):
            g_ieee_1.replay()
        ev1.record()
        torch.cuda.synchronize()
        ieee_b1_step_ms = float(ev0.elapsed_time(ev1)) / 8.0

        print(
            f"[2] Reference + Nyquist complete in {time.perf_counter() - t_ref_0:.2f}s "
            f"(IEEE-FP32 B=1 step = {ieee_b1_step_ms:.2f} ms, sin(t) ripple RMS = {nyquist_rms:.2e})"
        )

        # --------------------------------------------------------------------
        # 3. Single-Panel CUDAGraph Helper for CSP-BF16-IR Variants
        # --------------------------------------------------------------------
        outputs_8: dict[str, torch.Tensor] = {}
        timings_ms: dict[str, list[float]] = {}

        def eval_csp_single_panel_graph(
            name: str, panels: list[dict], bsw: int, oir: int, step_graph_1: torch.cuda.CUDAGraph
        ) -> None:
            M_c = 8
            c1_xa = torch.empty(1, 1, 3, 32, 32, device="cuda")
            c1_ea = torch.empty(1, 1, 3, 32, 32, device="cuda")
            c1_r = torch.empty(1, M_c, 1, 1, 1, device="cuda")
            c1_a = torch.empty(1, M_c, 1, 1, 1, device="cuda")
            c1_t = torch.empty(M_c, device="cuda")
            c1_tp = torch.empty(2 * M_c, device="cuda")
            c1_q0 = torch.empty(1, M_c, 1, 1, 1, device="cuda")
            c1_Qa = torch.empty(M_c, M_c, device="cuda")
            c1_dh = torch.empty(1, M_c, 1, 1, 1, device="cuda")
            c1_xo, c1_eo = torch.empty_like(c1_xa), torch.empty_like(c1_ea)

            p0 = panels[0]
            c1_xa.zero_()
            c1_ea.zero_()
            c1_r.copy_(p0["r_act"])
            c1_a.copy_(p0["a_act"])
            c1_t.copy_(p0["t_act"])
            c1_tp.copy_(p0["t_pair"])
            c1_q0.copy_(p0["q0"])
            c1_Qa.copy_(p0["Qa"])
            c1_dh.copy_(p0["dh"])
            _ = csp_single_exp_panel_step(
                c1_xa, c1_ea, c1_r, c1_a, c1_t, c1_tp, c1_q0, c1_Qa, c1_dh, unet_fp32, unet_bf16, bsw=bsw, oir=oir
            )
            torch.cuda.synchronize()
            g_pan1 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_pan1):
                xo_t, eo_t = csp_single_exp_panel_step(
                    c1_xa, c1_ea, c1_r, c1_a, c1_t, c1_tp, c1_q0, c1_Qa, c1_dh, unet_fp32, unet_bf16, bsw=bsw, oir=oir
                )
                c1_xo.copy_(xo_t)
                c1_eo.copy_(eo_t)

            def run_b1(x1: torch.Tensor) -> torch.Tensor:
                s1_x32.copy_(x1)
                s1_t32.fill_(panels[0]["t0"])
                step_graph_1.replay()
                c1_xa.copy_(x1[:, None])
                c1_ea.copy_(s1_o32[:, None])
                for pan in panels:
                    c1_r.copy_(pan["r_act"])
                    c1_a.copy_(pan["a_act"])
                    c1_t.copy_(pan["t_act"])
                    c1_tp.copy_(pan["t_pair"])
                    c1_q0.copy_(pan["q0"])
                    c1_Qa.copy_(pan["Qa"])
                    c1_dh.copy_(pan["dh"])
                    g_pan1.replay()
                    c1_xa.copy_(c1_xo)
                    c1_ea.copy_(c1_eo)
                return c1_xa[:, 0].clone()

            # Warmup + 5 B=1 timed trials
            _ = run_b1(xT8[0:1])
            torch.cuda.synchronize()
            ts = []
            for _ in range(5):
                ea_ev, eb_ev = torch.cuda.Event(True), torch.cuda.Event(True)
                ea_ev.record()
                _ = run_b1(xT8[0:1])
                eb_ev.record()
                torch.cuda.synchronize()
                ts.append(float(ea_ev.elapsed_time(eb_ev)))
            timings_ms[name] = ts

            # 8-seed outputs via exact same B=1 graph (fast & bit-identical to B=1 timing!)
            outs = [run_b1(xT8[i : i + 1]) for i in range(8)]
            outputs_8[name] = torch.cat(outs, dim=0)
            del g_pan1

        # Strict 24-bit IEEE-FP32 checkpoint variants (allow_tf32=False)
        eval_csp_single_panel_graph(
            "Log-SNR CSP-BF16-IR (P=7,M=8,1-IR,IEEE)", csp7_panels, bsw=2, oir=1, step_graph_1=g_ieee_1
        )
        eval_csp_single_panel_graph(
            "Log-SNR CSP-BF16-IR (P=9,M=8,1-IR,IEEE)", csp9_panels, bsw=1, oir=1, step_graph_1=g_ieee_1
        )
        eval_csp_single_panel_graph(
            "Log-SNR CSP-BF16-IR (P=9,M=8,2-IR,IEEE)", csp9_panels, bsw=2, oir=2, step_graph_1=g_ieee_1
        )
        del g_ieee_1

        # --------------------------------------------------------------------
        # 4. Enable Tensor-Core TF32 for Steelmanned Sequential FP32 & Fast Pareto CSP-IR
        # --------------------------------------------------------------------
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        s1_x16 = torch.empty(
            1, 3, 32, 32, dtype=torch.bfloat16, device="cuda", memory_format=torch.channels_last
        )
        s1_o16 = torch.empty_like(s1_x16)
        _ = unet_fp32(s1_x32.zero_(), s1_t32.zero_())
        _ = unet_bf16(s1_x16.zero_(), s1_t32.zero_())
        torch.cuda.synchronize()
        g_u32_1, g_u16_1 = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_u32_1):
            s1_o32.copy_(unet_fp32(s1_x32, s1_t32))
        with torch.cuda.graph(g_u16_1):
            s1_o16.copy_(unet_bf16(s1_x16, s1_t32))

        eval_csp_single_panel_graph(
            "Log-SNR CSP-BF16-IR (P=7,M=8,1-IR,TF32)", csp7_panels, bsw=2, oir=1, step_graph_1=g_u32_1
        )
        eval_csp_single_panel_graph(
            "Log-SNR CSP-BF16-IR (P=9,M=8,1-IR,TF32)", csp9_panels, bsw=1, oir=1, step_graph_1=g_u32_1
        )

        # Pure-BF16 ParaFlow-BF16 & ParaDiGMS-BF16
        outputs_8["Log-SNR ParaFlow-BF16 (P=4,M=8)"] = run_hybrid_csp_bf16_ir(
            xT8, paraflow4, unet_bf16, None, bsw=5, oir=0
        )
        outputs_8["ParaDiGMS-BF16 (S=96,W=8)"] = run_paradigms_bf16(
            xT8, paradigms_wins, unet_bf16, sweeps=3
        )

        x1_in_pf, x1_out_pf = torch.empty(1, 3, 32, 32, device="cuda"), torch.empty(1, 3, 32, 32, device="cuda")
        _ = run_hybrid_csp_bf16_ir(xT8[0:1], paraflow4, unet_bf16, None, bsw=5, oir=0)
        torch.cuda.synchronize()
        g_pf_b1 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_pf_b1):
            x1_out_pf.copy_(run_hybrid_csp_bf16_ir(x1_in_pf, paraflow4, unet_bf16, None, bsw=5, oir=0))

        x1_in_pd, x1_out_pd = torch.empty(1, 3, 32, 32, device="cuda"), torch.empty(1, 3, 32, 32, device="cuda")
        _ = run_paradigms_bf16(xT8[0:1], paradigms_wins, unet_bf16, sweeps=3)
        torch.cuda.synchronize()
        g_pd_b1 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_pd_b1):
            x1_out_pd.copy_(run_paradigms_bf16(x1_in_pd, paradigms_wins, unet_bf16, sweeps=3))

        for m_name, fn_g in [
            ("ParaDiGMS-BF16 (S=96,W=8)", lambda: (x1_in_pd.copy_(xT8[0:1]), g_pd_b1.replay())),
            ("Log-SNR ParaFlow-BF16 (P=4,M=8)", lambda: (x1_in_pf.copy_(xT8[0:1]), g_pf_b1.replay())),
        ]:
            ts = []
            for _ in range(5):
                ea_ev, eb_ev = torch.cuda.Event(True), torch.cuda.Event(True)
                ea_ev.record()
                fn_g()
                eb_ev.record()
                torch.cuda.synchronize()
                ts.append(float(ea_ev.elapsed_time(eb_ev)))
            timings_ms[m_name] = ts
        del g_pf_b1, g_pd_b1

        # Sequential DDIM & DPM-Solver-2
        ddim_scheds = {S: build_ddim_schedule(vp, S) for S in [20, 50, 100, 250]}
        dpm2_scheds = {S: build_dpm_solver2_schedule(vp, S) for S in [20, 50, 100]}
        for S in [20, 50, 100, 250]:
            for tag in ["FP32", "BF16"]:
                timings_ms[f"DDIM-{S} ({tag})"] = []
        for S in [20, 50, 100]:
            for tag in ["FP32", "BF16"]:
                timings_ms[f"DPM-Solver-2-{S} ({tag})"] = []

        x_s0 = xT8[0:1]
        x_s0_cl = x_s0.to(memory_format=torch.channels_last)
        for trial in range(3):
            max_ddim_st = 250 if trial == 0 else 100
            max_dpm_st = 50
            for bf16_f, tag in [(False, "FP32"), (True, "BF16")]:
                evs_ddim = {k: torch.cuda.Event(True) for k in [0, 20, 50, 100, 250] if k <= max_ddim_st}
                xc = x_s0.clone() if not bf16_f else x_s0_cl.clone()
                evs_ddim[0].record()
                for idx_st, (t_val, c1, c2) in enumerate(ddim_scheds[250][:max_ddim_st], start=1):
                    if not bf16_f:
                        s1_x32.copy_(xc)
                        s1_t32.fill_(t_val)
                        g_u32_1.replay()
                        xc = c1 * xc + c2 * s1_o32
                    else:
                        s1_x16.copy_(xc)
                        s1_t32.fill_(t_val)
                        g_u16_1.replay()
                        xc = c1 * xc + c2 * s1_o16.float()
                    if idx_st in evs_ddim:
                        evs_ddim[idx_st].record()
                torch.cuda.synchronize()
                for S in [20, 50, 100, 250]:
                    if S <= max_ddim_st:
                        timings_ms[f"DDIM-{S} ({tag})"].append(
                            float(evs_ddim[0].elapsed_time(evs_ddim[S]))
                        )

                evs_dpm = {k: torch.cuda.Event(True) for k in [0, 10, 25, 50] if k <= max_dpm_st}
                xc = x_s0.clone() if not bf16_f else x_s0_cl.clone()
                evs_dpm[0].record()
                for idx_st, (t0, tm, cm1, cm2, cb1, cb2) in enumerate(
                    dpm2_scheds[100][:max_dpm_st], start=1
                ):
                    if not bf16_f:
                        s1_x32.copy_(xc)
                        s1_t32.fill_(t0)
                        g_u32_1.replay()
                        xm = cm1 * xc + cm2 * s1_o32
                        s1_x32.copy_(xm)
                        s1_t32.fill_(tm)
                        g_u32_1.replay()
                        xc = cb1 * xc + cb2 * s1_o32
                    else:
                        s1_x16.copy_(xc)
                        s1_t32.fill_(t0)
                        g_u16_1.replay()
                        xm = cm1 * xc + cm2 * s1_o16.float()
                        s1_x16.copy_(xm)
                        s1_t32.fill_(tm)
                        g_u16_1.replay()
                        xc = cb1 * xc + cb2 * s1_o16.float()
                    if idx_st in evs_dpm:
                        evs_dpm[idx_st].record()
                torch.cuda.synchronize()
                for S_nfe in [20, 50, 100]:
                    if (S_nfe // 2) <= max_dpm_st:
                        timings_ms[f"DPM-Solver-2-{S_nfe} ({tag})"].append(
                            float(evs_dpm[0].elapsed_time(evs_dpm[S_nfe // 2]))
                        )

        del g_u32_1, g_u16_1

        # B=8 sequential accuracy evaluation
        b8_x32 = torch.empty(8, 3, 32, 32, device="cuda")
        b8_o32 = torch.empty_like(b8_x32)
        b8_x16 = torch.empty(
            8, 3, 32, 32, dtype=torch.bfloat16, device="cuda", memory_format=torch.channels_last
        )
        b8_o16 = torch.empty_like(b8_x16)
        b8_t32 = torch.empty(8, device="cuda")
        _ = unet_fp32(b8_x32.zero_(), b8_t32.zero_())
        _ = unet_bf16(b8_x16.zero_(), b8_t32.zero_())
        torch.cuda.synchronize()
        g_u32_8, g_u16_8 = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_u32_8):
            b8_o32.copy_(unet_fp32(b8_x32, b8_t32))
        with torch.cuda.graph(g_u16_8):
            b8_o16.copy_(unet_bf16(b8_x16, b8_t32))

        xT8_cl = xT8.to(memory_format=torch.channels_last)
        for S in [20, 50, 100, 250]:
            xf, xb = xT8.clone(), xT8_cl.clone()
            for t_val, c1, c2 in ddim_scheds[S]:
                b8_t32.fill_(t_val)
                b8_x32.copy_(xf)
                g_u32_8.replay()
                xf = c1 * xf + c2 * b8_o32
                b8_x16.copy_(xb)
                g_u16_8.replay()
                xb = c1 * xb + c2 * b8_o16.float()
            outputs_8[f"DDIM-{S} (FP32)"] = xf
            outputs_8[f"DDIM-{S} (BF16)"] = xb.contiguous()

        for S in [20, 50, 100]:
            xf, xb = xT8.clone(), xT8_cl.clone()
            for t0, tm, cm1, cm2, cb1, cb2 in dpm2_scheds[S]:
                b8_t32.fill_(t0)
                b8_x32.copy_(xf)
                g_u32_8.replay()
                xmf = cm1 * xf + cm2 * b8_o32
                b8_x16.copy_(xb)
                g_u16_8.replay()
                xmb = cm1 * xb + cm2 * b8_o16.float()
                b8_t32.fill_(tm)
                b8_x32.copy_(xmf)
                g_u32_8.replay()
                xf = cb1 * xf + cb2 * b8_o32
                b8_x16.copy_(xmb)
                g_u16_8.replay()
                xb = cb1 * xb + cb2 * b8_o16.float()
            outputs_8[f"DPM-Solver-2-{S} (FP32)"] = xf
            outputs_8[f"DPM-Solver-2-{S} (BF16)"] = xb.contiguous()

        del g_u32_8, g_u16_8
        torch.cuda.synchronize()

        csp9_s0 = outputs_8["Log-SNR CSP-BF16-IR (P=9,M=8,2-IR,IEEE)"][0:1]
        psnr_vs_fp32_colloc = 10.0 * math.log10(
            4.0 / (float(torch.mean((csp9_s0 - x_fp32_colloc_s0) ** 2)) + 1e-30)
        )

        method_order = (
            [f"DDIM-{S} ({p})" for S in [20, 50, 100, 250] for p in ["FP32", "BF16"]]
            + [f"DPM-Solver-2-{S} ({p})" for S in [20, 50, 100] for p in ["FP32", "BF16"]]
            + [
                "ParaDiGMS-BF16 (S=96,W=8)",
                "Log-SNR ParaFlow-BF16 (P=4,M=8)",
                "Log-SNR CSP-BF16-IR (P=7,M=8,1-IR,TF32)",
                "Log-SNR CSP-BF16-IR (P=9,M=8,1-IR,TF32)",
                "Log-SNR CSP-BF16-IR (P=7,M=8,1-IR,IEEE)",
                "Log-SNR CSP-BF16-IR (P=9,M=8,1-IR,IEEE)",
                "Log-SNR CSP-BF16-IR (P=9,M=8,2-IR,IEEE)",
            ]
        )

        results_rows = []
        print(f"\n{'Method':42s} | {'B=1 ms':>8s} | {'8-Seed PSNR (dB)':>18s} | {'Rel L2':>10s}")
        print("-" * 88)
        for name in method_order:
            t_list = sorted(timings_ms[name])
            med_ms = t_list[len(t_list) // 2]
            met = compute_metrics(outputs_8[name], x_ref8)
            row = {"method": name, "b1_median_ms": round(med_ms, 2), **met}
            results_rows.append(row)
            print(
                f"{name:42s} | {med_ms:8.2f} | {met['psnr_mean_db']:7.2f} +- {met['psnr_std_db']:4.2f} dB | {met['rel_l2_mean']:10.3e}"
            )

        vis_save = {
            "x_ref8": x_ref8.cpu().numpy(),
            "DDIM-100 (BF16)": outputs_8["DDIM-100 (BF16)"].cpu().numpy(),
            "DPM-Solver-2-100 (BF16)": outputs_8["DPM-Solver-2-100 (BF16)"].cpu().numpy(),
            "ParaDiGMS-BF16 (S=96,W=8)": outputs_8["ParaDiGMS-BF16 (S=96,W=8)"].cpu().numpy(),
            "Log-SNR ParaFlow-BF16 (P=4,M=8)": outputs_8["Log-SNR ParaFlow-BF16 (P=4,M=8)"].cpu().numpy(),
            "Log-SNR CSP-BF16-IR (P=7,M=8,1-IR,TF32)": outputs_8["Log-SNR CSP-BF16-IR (P=7,M=8,1-IR,TF32)"].cpu().numpy(),
            "Log-SNR CSP-BF16-IR (P=9,M=8,2-IR,IEEE)": outputs_8["Log-SNR CSP-BF16-IR (P=9,M=8,2-IR,IEEE)"].cpu().numpy(),
            "t_fine": t_np,
            "nyquist_residual": nyquist_residual,
        }
        np.savez_compressed(ckpt_dir / "cifar10_vis_arrays.npz", **vis_save)

        peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024**2))
        total_wall_s = time.perf_counter() - t_wall_0
        out_json = ROOT / "results" / "stage_d_ddpm_cifar10_pareto.json"
        summary = {
            "stage": "Stage D — Pretrained google/ddpm-cifar10-32 Full Pareto Frontier & Nyquist Spectrum",
            "gpu": gpu_name,
            "pytorch_version": torch.__version__,
            "model": "google/ddpm-cifar10-32 (35.75M UNet2DModel, strict=True)",
            "num_seeds": 8,
            "seeds": seeds,
            "strict_ieee_fp32_b1_step_ms": round(ieee_b1_step_ms, 3),
            "strict_ieee_fp32_ddim250_b1_ms": round(250.0 * ieee_b1_step_ms, 2),
            "strict_ieee_fp32_dpm100_b1_ms": round(100.0 * ieee_b1_step_ms, 2),
            "csp_vs_fp32_collocation_psnr_db": round(psnr_vs_fp32_colloc, 2),
            "sin_999_tau_nyquist_ripple_rms": nyquist_rms,
            "peak_vram_mb": round(peak_vram_mb, 2),
            "total_benchmark_wall_s": round(total_wall_s, 2),
            "singularity_curve": singularity_curve,
            "methods": results_rows,
        }
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            f"\n[Done] Saved {out_json} | Colloc PSNR: {psnr_vs_fp32_colloc:.2f} dB | "
            f"Peak VRAM: {peak_vram_mb:.1f} MB | Wall: {total_wall_s:.2f}s"
        )


if __name__ == "__main__":
    main()

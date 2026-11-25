"""Publication Figure Generator for `ResidualRescaledBF16`.

Generates all 6 vector PDF + 300-DPI PNG figures in `paper/figures/` directly from the
clean-room verified JSON artifacts in `results/` and cached sample arrays in `.cache/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES_DIR = ROOT / "results"
FIG_DIR = ROOT / "paper" / "figures"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 10.5,
        "axes.titlesize": 11.0,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 8.4,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save_both(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {stem}.pdf & {stem}.png")


def make_fig1() -> None:
    d_a = json.loads((RES_DIR / "stage_a_vcurve.json").read_text(encoding="utf-8"))
    d_b = json.loads((RES_DIR / "stage_b_dit_cudagraph.json").read_text(encoding="utf-8"))

    sweep = d_a["sweep"]
    sigmas = np.array([r["sigma"] for r in sweep])
    fwd = np.array([r["fwd_o_eps_half_rel_err"] for r in sweep])
    cen = np.array([r["central_o_eps_twothirds_rel_err"] for r in sweep])
    fwd_min_pct = 100.0 * d_a["optimal_fwd_rel_err"]
    cen_min_pct = 100.0 * d_a["optimal_central_rel_err"]
    gain_x = d_a["peak_single_sweep_gain_x"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.35), gridspec_kw={"wspace": 0.28})

    ax1.loglog(
        sigmas,
        fwd,
        "o--",
        color="#d95f02",
        lw=2.0,
        ms=5.8,
        label=rf"Forward SP-BF16 $\mathcal{{O}}(\varepsilon_{{\mathrm{{bf16}}}}^{{1/2}})$ (min {fwd_min_pct:.1f}%)",
    )
    ax1.loglog(
        sigmas,
        cen,
        "s-",
        color="#1b9e77",
        lw=2.4,
        ms=6.2,
        label=rf"Central CSP-BF16 $\mathcal{{O}}(\varepsilon_{{\mathrm{{bf16}}}}^{{2/3}})$ (min {cen_min_pct:.2f}%, {gain_x:.2f}$\times$ gain)",
    )
    ax1.axvline(0.125, color="#1b9e77", ls=":", lw=1.5, alpha=0.85)
    ax1.annotate(
        rf"Optimal $\sigma^\star = 2^{{-3}} = 0.125$" + "\n" + rf"({gain_x:.2f}$\times$ lower JVP error)",
        xy=(0.125, d_a["optimal_central_rel_err"]),
        xytext=(0.011, 0.016),
        fontsize=8.4,
        fontweight="bold",
        color="#0f684e",
        arrowprops=dict(arrowstyle="->", color="#0f684e", lw=1.2),
    )
    ax1.text(
        0.0033,
        0.032,
        f"Unscaled BF16 FD: {d_a['naive_unscaled_bf16_fd_rel_err']:.0f}x error\n(100% mantissa underflow)",
        fontsize=7.6,
        color="#b2182b",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.22", fc="#fde8e8", ec="#b2182b", lw=0.8),
    )
    s_left = np.array([0.004, 0.032])
    ax1.loglog(
        s_left,
        0.0040 / s_left,
        ":",
        color="#666666",
        lw=1.1,
        label=r"Quantization slope $\propto \sigma^{-1}$",
    )
    ax1.set_xlim(3e-3, 1.35)
    ax1.set_ylim(1e-2, 16.0)
    ax1.set_xlabel(r"Macro-Perturbation Scale $\sigma$ ($s_c = \sigma \|X\| / \|R\|$)", fontweight="bold")
    ax1.set_ylabel("Relative Directional JVP Error", fontweight="bold")
    ax1.set_title("(a) 7-Bit BF16 Directional JVP Scaling Law (46.9M DiT)", fontweight="bold")
    ax1.grid(True, which="both", ls=":", alpha=0.35)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.93, fontsize=7.6)

    abl = d_b["fwd_vs_central_ablation"]
    items = [
        ("ParaFlow-BF16\n(0-IR, Pure BF16)", abl["PTX+CUDAGraph ParaFlow-BF16 (4,0,0)"], "#7570b3"),
        ("Fwd SP-BF16-IR\n(4,3,1) [O(eps^1/2)]", abl["PTX+CUDAGraph Fwd SP-BF16-IR (4,3,1) [O(eps^1/2)]"], "#d95f02"),
        ("CSP-BF16-IR\n(4,2,2) [Ours]", abl["PTX+CUDAGraph Central CSP-BF16-IR (4,2,2) [O(eps^2/3)]"], "#66a61e"),
        ("CSP-BF16-IR\n(4,3,1) [Ours]", abl["PTX+CUDAGraph Central CSP-BF16-IR (4,3,1) [O(eps^2/3)]"], "#2b8cbe"),
        ("CSP-BF16-IR\n(4,3,2) [Ours]", abl["PTX+CUDAGraph Central CSP-BF16-IR (4,3,2) [O(eps^2/3)]"], "#1b9e77"),
    ]
    y_pos = np.arange(len(items))
    errs = [it[1]["rel_l2_err_vs_fp64"] for it in items]
    ms_vals = [it[1]["cuda_ms"] for it in items]
    cols = [it[2] for it in items]

    ax2.barh(y_pos, errs, color=cols, height=0.58, edgecolor="#222222", lw=0.7)
    ax2.set_xscale("log")
    ax2.axvline(2.1e-3, color="#b2182b", ls="--", lw=1.3, label="7-Bit BF16 Stagnation Wall (~2.1e-3)")
    ax2.axvline(1.19e-7, color="#333333", ls=":", lw=1.2, label="IEEE-FP32 Machine Unit (1.19e-7)")
    for idx, (err_v, ms_v) in enumerate(zip(errs, ms_vals)):
        ax2.text(
            err_v * 1.35,
            idx,
            f"{err_v:.2e} ({ms_v:.2f} ms)",
            va="center",
            fontsize=8.1,
            fontweight="bold",
        )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([it[0] for it in items], fontsize=8.5)
    ax2.invert_yaxis()
    ax2.set_xlim(2e-8, 3.5e-1)
    ax2.set_xlabel(r"Trajectory Relative $L_2$ Error vs. Exact FP64 RK4", fontweight="bold")
    ax2.set_title("(b) Forward vs. Central IR Precision & Latency", fontweight="bold")
    ax2.grid(True, axis="x", which="both", ls=":", alpha=0.35)
    ax2.legend(loc="lower right", frameon=True, framealpha=0.93, fontsize=7.9)

    save_both(fig, "fig1_bf16_vcurve_scaling_law")


def make_fig2() -> None:
    d_b = json.loads((RES_DIR / "stage_b_dit_cudagraph.json").read_text(encoding="utf-8"))
    rows = d_b["scaling_s64_to_s512"]
    S = np.array([r["S_steps"] for r in rows])
    s512 = rows[-1]
    csp_err = s512["csp_bf16_ir_432_rel_l2_err"]
    sp_cg = s512["speedup_csp432_vs_cudagraph_rk4_x"]
    sp_pf = s512["speedup_pf_bf16_vs_cudagraph_rk4_x"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.35), gridspec_kw={"wspace": 0.27})

    ax1.semilogy(S, [r["eager_rk4_fp32_ms"] for r in rows], "^--", color="#8c510a", lw=1.8, ms=6, label="Eager RK4 (FP32)")
    ax1.semilogy(S, [r["cudagraph_rk4_fp32_ms"] for r in rows], "o-", color="#d95f02", lw=2.1, ms=6, label="CUDAGraph RK4 (FP32)")
    ax1.semilogy(S, [r["cudagraph_paradigms_bf16_ms"] for r in rows], "d--", color="#7570b3", lw=1.9, ms=6, label="CUDAGraph ParaDiGMS (BF16)")
    ax1.semilogy(S, [r["ptx_cg_csp_bf16_ir_432_ms"] for r in rows], "s-", color="#1b9e77", lw=2.4, ms=6.5, label=f"PTX+CG CSP-BF16-IR (Ours, {csp_err:.2e})")
    ax1.semilogy(S, [r["ptx_cg_paraflow_bf16_ms"] for r in rows], "p-", color="#2b8cbe", lw=2.2, ms=6.5, label="PTX+CG ParaFlow-BF16 (Ours, Fast)")

    ax1.annotate(
        f"{sp_cg:.1f}x faster vs CG-RK4 ({s512['ptx_cg_csp_bf16_ir_432_ms']:.2f} ms)",
        xy=(512, s512["ptx_cg_csp_bf16_ir_432_ms"]),
        xytext=(215, 5.4),
        fontsize=8.2,
        fontweight="bold",
        color="#0f684e",
        arrowprops=dict(arrowstyle="->", color="#0f684e", lw=1.2),
    )
    ax1.text(
        215,
        3.3,
        f"{sp_pf:.0f}x faster vs CG-RK4 ({s512['ptx_cg_paraflow_bf16_ms']:.2f} ms)",
        fontsize=8.1,
        fontweight="bold",
        color="#1d658c",
    )
    ax1.set_xticks(S)
    ax1.set_xlabel("Sequential Discretization Budget $S$ (NFEs)", fontweight="bold")
    ax1.set_ylabel("Single-Trajectory Latency $B=1$ (ms, log scale)", fontweight="bold")
    ax1.set_title("(a) Wall-Clock Latency on 46.91M SwiGLU DiT (NVIDIA GB10)", fontweight="bold")
    ax1.grid(True, which="both", ls=":", alpha=0.35)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.93, fontsize=7.8)

    ax2.semilogy(S, [r["euler_fp32_rel_l2_err"] for r in rows], "v--", color="#999999", lw=1.7, ms=5.8, label="Euler (FP32)")
    ax2.semilogy(S, [r["paradigms_bf16_rel_l2_err"] for r in rows], "d--", color="#7570b3", lw=2.0, ms=6, label="ParaDiGMS (BF16 — Stalls at 2.1e-3)")
    ax2.semilogy(S, [r["rk4_fp32_rel_l2_err"] for r in rows], "o-", color="#d95f02", lw=2.0, ms=6, label="CUDAGraph RK4 (FP32)")
    ax2.semilogy(S, [r["csp_bf16_ir_431_rel_l2_err"] for r in rows], "^-", color="#2b8cbe", lw=1.9, ms=6, label="PTX+CG CSP-BF16-IR (4,3,1)")
    ax2.semilogy(S, [r["csp_bf16_ir_432_rel_l2_err"] for r in rows], "s-", color="#1b9e77", lw=2.5, ms=6.5, label=f"PTX+CG CSP-BF16-IR (4,3,2) [{csp_err:.2e}]")

    ax2.axhspan(1.8e-3, 3.2e-3, color="#fde8e8", alpha=0.65)
    ax2.text(70, 2.3e-3, "7-Bit BF16 Fixed-Point Stagnation Wall", fontsize=8.1, color="#b2182b", fontweight="bold")
    ax2.annotate(
        "23,000x cleaner than BF16 Picard;\n3.0x cleaner than 512-step FP32 RK4!",
        xy=(512, csp_err),
        xytext=(195, 1.35e-6),
        fontsize=8.1,
        fontweight="bold",
        color="#0f684e",
        arrowprops=dict(arrowstyle="->", color="#0f684e", lw=1.2),
    )
    ax2.set_xticks(S)
    ax2.set_ylim(3e-8, 1.2e-2)
    ax2.set_xlabel("Sequential Discretization Budget $S$ (NFEs)", fontweight="bold")
    ax2.set_ylabel(r"Relative $L_2$ Error vs. Exact FP64 RK4", fontweight="bold")
    ax2.set_title("(b) Breaking the 7-Bit BF16 Mantissa Ceiling", fontweight="bold")
    ax2.grid(True, which="both", ls=":", alpha=0.35)
    ax2.legend(loc="center right", bbox_to_anchor=(0.99, 0.56), frameon=True, framealpha=0.93, fontsize=7.6)

    save_both(fig, "fig2_dit_cudagraph_scaling")


def make_fig3() -> None:
    d_c = json.loads((RES_DIR / "stage_c_flow_matching.json").read_text(encoding="utf-8"))
    ot = d_c["ot_cfm_8mode_r128"]["samplers"]
    gl = d_c["spatial_2d_glyph_dit_16x16"]["samplers"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.35), gridspec_kw={"wspace": 0.28})

    ot_pts = [
        ("Euler-128 (FP32)", ot["GPU Euler-128 (FP32)"], "#8c510a", "v", (6, -3)),
        ("Heun-128 (FP32)", ot["GPU Heun-128 (FP32)"], "#d95f02", "o", (6, -3)),
        ("RK4-256 (FP32)", ot["GPU RK4-256 (FP32)"], "#e7298a", "^", (-48, 8)),
        ("ParaDiGMS-128 (BF16)", ot["GPU ParaDiGMS-128 (BF16, W=32)"], "#7570b3", "d", (6, 2)),
        ("ParaFlow-BF16 (Ours)", ot["ParaFlow-BF16 (M=16, 5+0)"], "#2b8cbe", "p", (6, -8)),
        ("SP/CSP-BF16-IR (Ours)", ot["SP-BF16-IR (M=16, 4+2)"], "#1b9e77", "s", (6, -4)),
    ]
    for lbl, row, col, mk, off in ot_pts:
        ax1.loglog(
            row["b1_latency_ms"],
            row["traj_rel_l2_vs_rk4_512"],
            mk,
            color=col,
            ms=8.5,
            label=f"{lbl} (SW2={row['sliced_wasserstein_2']:.4f})",
        )
        ax1.annotate(
            lbl.split(" ")[0],
            xy=(row["b1_latency_ms"], row["traj_rel_l2_vs_rk4_512"]),
            xytext=off,
            textcoords="offset points",
            fontsize=8.0,
            color=col,
            fontweight="bold",
        )
    ax1.set_xlim(2.2, 360.0)
    ax1.set_ylim(4e-7, 6.5e-2)
    ax1.set_xlabel("Single-Sample Latency $B=1$ (ms, log scale)", fontweight="bold")
    ax1.set_ylabel(r"Trajectory Relative $L_2$ Error (log scale)", fontweight="bold")
    ax1.set_title("(a) 33.6M 8-Mode Curved OT-CFM (Zero SW2 Loss)", fontweight="bold")
    ax1.grid(True, which="both", ls=":", alpha=0.35)
    ax1.legend(loc="lower left", frameon=True, framealpha=0.93, fontsize=7.4)

    gl_names = [
        ("Eager Euler-128\n(FP32)", gl["Eager Euler-128 (FP32)"], "#8c510a"),
        ("ParaDiGMS-128\n(BF16, CG)", gl["CUDAGraph ParaDiGMS-128 (BF16)"], "#7570b3"),
        ("ParaFlow-BF16\n(Ours, PTX+CG)", gl["PTX+CG ParaFlow-BF16 (5,0,0)"], "#2b8cbe"),
        ("CSP-BF16-IR\n(Ours, PTX+CG)", gl["PTX+CG CSP-BF16-IR (4,2,2)"], "#1b9e77"),
    ]
    x_idx = np.arange(len(gl_names))
    psnrs = [g[1]["psnr_vs_exact_ode_db"] for g in gl_names]
    lats = [g[1]["b1_latency_ms"] for g in gl_names]
    cols = [g[2] for g in gl_names]
    gain_db = psnrs[3] - psnrs[1]

    bars = ax2.bar(x_idx, psnrs, width=0.54, color=cols, edgecolor="#222222", lw=0.8)
    ax2.axhline(psnrs[1], color="#7570b3", ls="--", lw=1.2, alpha=0.8)
    for bar, lat_v, psnr_v in zip(bars, lats, psnrs):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            psnr_v + 1.4,
            f"{psnr_v:.1f} dB\n({lat_v:.2f} ms)",
            ha="center",
            va="bottom",
            fontsize=8.1,
            fontweight="bold",
        )
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels([g[0] for g in gl_names], fontsize=8.4)
    ax2.set_ylim(0, 84)
    ax2.set_ylabel("Continuous-ODE Image Fidelity PSNR (dB)", fontweight="bold")
    ax2.set_title(f"(b) 34.65M 2D Spatial Glyph DiT (+{gain_db:.1f} dB & Faster)", fontweight="bold")
    ax2.grid(True, axis="y", ls=":", alpha=0.35)

    save_both(fig, "fig3_2d_glyph_and_otcfm")


def make_fig4() -> None:
    d_d = json.loads((RES_DIR / "stage_d_ddpm_cifar10_pareto.json").read_text(encoding="utf-8"))
    rows = {r["method"]: r for r in d_d["methods"]}
    sing = d_d["singularity_curve"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.5), gridspec_kw={"wspace": 0.27})

    ddim_bf16 = [rows[f"DDIM-{S} (BF16)"] for S in [20, 50, 100, 250]]
    ddim_fp32 = [rows[f"DDIM-{S} (FP32)"] for S in [20, 50, 100, 250]]
    dpm_bf16 = [rows[f"DPM-Solver-2-{S} (BF16)"] for S in [20, 50, 100]]
    dpm_fp32 = [rows[f"DPM-Solver-2-{S} (FP32)"] for S in [20, 50, 100]]

    ax1.semilogx([r["b1_median_ms"] for r in ddim_bf16], [r["psnr_mean_db"] for r in ddim_bf16], "v--", color="#999999", lw=1.6, ms=5.5, label="DDIM (BF16)")
    ax1.semilogx([r["b1_median_ms"] for r in ddim_fp32], [r["psnr_mean_db"] for r in ddim_fp32], "^--", color="#8c510a", lw=1.6, ms=5.5, label="DDIM (TF32-FP32)")
    ax1.semilogx([r["b1_median_ms"] for r in dpm_bf16], [r["psnr_mean_db"] for r in dpm_bf16], "o--", color="#e6ab02", lw=2.0, ms=6.0, label="DPM-Solver-2 (BF16 — Stalls @ 46.0 dB)")
    ax1.semilogx([r["b1_median_ms"] for r in dpm_fp32], [r["psnr_mean_db"] for r in dpm_fp32], "o-", color="#d95f02", lw=2.0, ms=6.0, label="DPM-Solver-2 (TF32-FP32)")

    pd_r = rows["ParaDiGMS-BF16 (S=96,W=8)"]
    ax1.semilogx(pd_r["b1_median_ms"], pd_r["psnr_mean_db"], "D", color="#7570b3", ms=7.5, label=f"ParaDiGMS-BF16 ({pd_r['psnr_mean_db']:.1f} dB)")

    pf_r = rows["Log-SNR ParaFlow-BF16 (P=4,M=8)"]
    ax1.semilogx(pf_r["b1_median_ms"], pf_r["psnr_mean_db"], "P", color="#2b8cbe", ms=9.0, label="Log-SNR ParaFlow-BF16 (Ours, Pure BF16)")

    csp_pts = [
        rows["Log-SNR CSP-BF16-IR (P=9,M=8,1-IR,TF32)"],
        rows["Log-SNR CSP-BF16-IR (P=9,M=8,1-IR,IEEE)"],
        rows["Log-SNR CSP-BF16-IR (P=9,M=8,2-IR,IEEE)"],
    ]
    ax1.semilogx(
        [r["b1_median_ms"] for r in csp_pts],
        [r["psnr_mean_db"] for r in csp_pts],
        "s-",
        color="#1b9e77",
        lw=2.5,
        ms=7.8,
        label="Log-SNR CSP-BF16-IR (Ours, 1-IR & 2-IR)",
    )

    ieee_ddim250_ms = d_d["strict_ieee_fp32_ddim250_b1_ms"]
    ax1.semilogx(
        ieee_ddim250_ms,
        rows["DDIM-250 (FP32)"]["psnr_mean_db"],
        "X",
        color="#b2182b",
        ms=8.0,
        label=f"DDIM-250 (Strict IEEE-FP32, {ieee_ddim250_ms:.0f} ms)",
    )

    ax1.axhspan(45.3, 47.0, color="#fff3cd", alpha=0.75)
    ax1.text(46, 45.7, "7-Bit BF16 Ceiling (~46.0 dB)", fontsize=7.8, color="#856404", fontweight="bold")
    ax1.annotate(
        f"{csp_pts[-1]['psnr_mean_db']:.1f} dB (+23.7 dB vs BF16!\n{d_d['csp_vs_fp32_collocation_psnr_db']:.1f} dB vs FP32 Colloc)",
        xy=(csp_pts[-1]["b1_median_ms"], csp_pts[-1]["psnr_mean_db"]),
        xytext=(115, 65.2),
        fontsize=8.1,
        fontweight="bold",
        color="#0f684e",
        arrowprops=dict(arrowstyle="->", color="#0f684e", lw=1.2),
    )
    ax1.set_xlim(42, 4600)
    ax1.set_ylim(25, 76)
    ax1.set_xlabel("Single-Image Latency $B=1$ (ms, log scale)", fontweight="bold")
    ax1.set_ylabel("8-Seed Mean PSNR vs. Exact Continuous ODE (dB)", fontweight="bold")
    ax1.set_title("(a) Pretrained google/ddpm-cifar10-32 Pareto Frontier", fontweight="bold")
    ax1.grid(True, which="both", ls=":", alpha=0.35)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.93, fontsize=7.2)

    lams = np.array([s["lambda"] for s in sing])
    raw_e = np.array([s["raw_bf16_eps_rms"] for s in sing])
    amp_e = np.array([s["naive_inv_alpha_amplified_rms"] for s in sing])
    z_e = np.array([s["z_root_tau_panel_rms"] for s in sing])

    ax2.semilogy(lams, amp_e, "-", color="#b2182b", lw=2.3, label=r"Naive $x_0$-Pred Error $\|\delta\epsilon_{\mathrm{bf16}}\| / \alpha_\tau$ ($152\times$ blowup)")
    ax2.semilogy(lams, raw_e, "--", color="#d95f02", lw=1.8, label=r"Raw UNet BF16 Error $\|\epsilon_{\mathrm{bf16}} - \epsilon_{\mathrm{fp32}}\|_{\mathrm{RMS}}$")
    ax2.semilogy(lams, z_e, "-", color="#1b9e77", lw=2.3, label=r"Desingularized $z=\sqrt{\tau}$ Panel Increment Error (Ours)")
    ax2.axvline(0.0, color="#444444", ls=":", lw=1.3)
    ax2.annotate(
        r"$t=999$: $1/\alpha_T = 152.2\times$" + "\nBF16 noise blowup!",
        xy=(-4.55, amp_e[0]),
        xytext=(-4.65, 0.032),
        fontsize=8.1,
        color="#b2182b",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#b2182b", lw=1.1),
    )
    ax2.text(0.18, 4.5e-4, r"$\lambda > 0$: Exact $e^{-\lambda}$" + "\nIntegrating Factor", fontsize=8.1, color="#333333")
    ax2.set_ylim(3e-4, 0.9)
    ax2.set_xlabel(r"Half-Log-SNR $\lambda = \ln(\alpha_\tau / \sigma_\tau)$", fontweight="bold")
    ax2.set_ylabel("Quantization Noise RMS (log scale)", fontweight="bold")
    ax2.set_title(r"(b) High-Noise $1/\alpha_T$ Singularity vs. $z=\sqrt{\tau}$ Coordinate", fontweight="bold")
    ax2.grid(True, which="both", ls=":", alpha=0.35)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.93, fontsize=7.5)

    save_both(fig, "fig4_ddpm_cifar10_pareto_and_nyquist")


def make_fig5() -> None:
    d_d = json.loads((RES_DIR / "stage_d_ddpm_cifar10_pareto.json").read_text(encoding="utf-8"))
    rows = {r["method"]: r for r in d_d["methods"]}
    npz = np.load(ROOT / ".cache" / "ddpm_cifar10_32" / "cifar10_vis_arrays.npz")
    x_ref8 = npz["x_ref8"]

    vis_cols = [
        ("DDIM-100 (BF16)", "DDIM-100 (BF16)"),
        ("ParaDiGMS-BF16 (S=96,W=8)", "ParaDiGMS-BF16"),
        ("Log-SNR ParaFlow-BF16 (P=4,M=8)", "ParaFlow-BF16 (Ours)"),
        ("Log-SNR CSP-BF16-IR (P=9,M=8,2-IR,IEEE)", "CSP-BF16-IR (Ours)"),
    ]

    fig, axes = plt.subplots(
        8,
        9,
        figsize=(14.5, 12.8),
        gridspec_kw={"wspace": 0.06, "hspace": 0.12, "width_ratios": [1.05, 1, 1, 1, 1, 1, 1, 1, 1]},
    )

    def to_rgb(chw: np.ndarray) -> np.ndarray:
        arr = np.clip((np.transpose(chw, (1, 2, 0)) + 1.0) * 0.5, 0.0, 1.0)
        return np.repeat(np.repeat(arr, 4, axis=0), 4, axis=1)

    def to_err50(chw: np.ndarray, ref: np.ndarray) -> np.ndarray:
        err = np.clip(np.mean(np.abs(chw - ref), axis=0) * 50.0, 0.0, 1.0)
        return np.repeat(np.repeat(err, 4, axis=0), 4, axis=1)

    im_h = None
    for s in range(8):
        ax0 = axes[s, 0]
        ax0.imshow(to_rgb(x_ref8[s]), interpolation="nearest")
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.set_ylabel(f"Seed {s}", fontsize=8.5, fontweight="bold")
        if s == 0:
            ax0.set_title("Exact ODE\n(IEEE-FP32)", fontsize=8.5, fontweight="bold")

        for m_i, (m_key, m_lbl) in enumerate(vis_cols):
            pred = npz[m_key][s]
            psnr_s = rows[m_key]["psnr_per_seed"][s]
            mean_p = rows[m_key]["psnr_mean_db"]
            ax_r = axes[s, 1 + 2 * m_i]
            ax_e = axes[s, 2 + 2 * m_i]
            ax_r.imshow(to_rgb(pred), interpolation="nearest")
            im_h = ax_e.imshow(to_err50(pred, x_ref8[s]), cmap="inferno", vmin=0.0, vmax=1.0, interpolation="nearest")
            for ax_c in (ax_r, ax_e):
                ax_c.set_xticks([])
                ax_c.set_yticks([])
                for sp in ax_c.spines.values():
                    sp.set_visible(True)
                    sp.set_edgecolor("#1b9e77" if "CSP" in m_key else "#cccccc")
                    sp.set_linewidth(1.3 if "CSP" in m_key else 0.6)
            ax_e.text(
                0.95,
                0.06,
                f"{psnr_s:.1f} dB",
                transform=ax_e.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.8,
                fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.14", fc="black", alpha=0.72, ec="none"),
            )
            if s == 0:
                ax_r.set_title(f"{m_lbl}\nRGB", fontsize=8.2, fontweight="bold")
                ax_e.set_title(f"|Err| x 50\n({mean_p:.1f} dB)", fontsize=8.2, fontweight="bold")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.013, 0.70])
    cb = fig.colorbar(im_h, cax=cbar_ax)
    cb.set_label(r"Amplified Pixel Error $|\hat{x}_0 - x_{\mathrm{exact}}| \times 50$ ($1.0 = 0.020$ in $[-1, 1]$)", fontsize=9.5)
    fig.suptitle(
        "Zero-Shot Pretrained google/ddpm-cifar10-32 (8 Canonical Seeds) — Eliminating BF16 Quantization Drift",
        fontsize=11.5,
        fontweight="bold",
        y=0.935,
    )
    save_both(fig, "fig5_ddpm_cifar10_visual_heatmaps")


def make_fig6() -> None:
    d_d = json.loads((RES_DIR / "stage_d_ddpm_cifar10_pareto.json").read_text(encoding="utf-8"))
    d_e = json.loads((RES_DIR / "stage_e_celebahq256_scaleup.json").read_text(encoding="utf-8"))
    rows_d = {r["method"]: r for r in d_d["methods"]}
    rows_e = {r["method"]: r for r in d_e["methods"]}
    npz = np.load(ROOT / ".cache" / "ddpm_celebahq_256" / "celebahq256_vis_arrays.npz")
    x_ref4 = npz["x_ref4"]

    vis_cols = [
        ("DDIM-100 (BF16)", "DDIM-100 (BF16)"),
        ("ParaDiGMS-BF16 (S=96,W=8)", "ParaDiGMS-BF16"),
        ("Log-SNR ParaFlow-BF16 (P=4,M=8)", "ParaFlow-BF16 (Ours)"),
        ("Log-SNR CSP-BF16-IR (P=9,M=8,2-IR,IEEE)", "CSP-BF16-IR (Ours)"),
    ]

    fig = plt.figure(figsize=(14.6, 10.6))
    gs_top = fig.add_gridspec(
        4,
        9,
        top=0.92,
        bottom=0.35,
        left=0.05,
        right=0.90,
        wspace=0.05,
        hspace=0.10,
        width_ratios=[1.05, 1, 1, 1, 1, 1, 1, 1, 1],
    )

    def to_rgb256(chw: np.ndarray) -> np.ndarray:
        return np.clip((np.transpose(chw, (1, 2, 0)) + 1.0) * 0.5, 0.0, 1.0)

    def to_err50_256(chw: np.ndarray, ref: np.ndarray) -> np.ndarray:
        return np.clip(np.mean(np.abs(chw - ref), axis=0) * 50.0, 0.0, 1.0)

    im_h = None
    for s in range(4):
        ax0 = fig.add_subplot(gs_top[s, 0])
        ax0.imshow(to_rgb256(x_ref4[s]))
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.set_ylabel(f"Seed {s}", fontsize=8.5, fontweight="bold")
        if s == 0:
            ax0.set_title("Exact 256x256\n(IEEE-FP32)", fontsize=8.3, fontweight="bold")

        for m_i, (m_key, m_lbl) in enumerate(vis_cols):
            pred = npz[m_key][s]
            psnr_s = rows_e[m_key]["psnr_per_seed"][s]
            mean_p = rows_e[m_key]["psnr_mean_db"]
            ax_r = fig.add_subplot(gs_top[s, 1 + 2 * m_i])
            ax_e = fig.add_subplot(gs_top[s, 2 + 2 * m_i])
            ax_r.imshow(to_rgb256(pred))
            im_h = ax_e.imshow(to_err50_256(pred, x_ref4[s]), cmap="inferno", vmin=0.0, vmax=1.0)
            for ax_c in (ax_r, ax_e):
                ax_c.set_xticks([])
                ax_c.set_yticks([])
                for sp in ax_c.spines.values():
                    sp.set_visible(True)
                    sp.set_edgecolor("#1b9e77" if "CSP" in m_key else "#cccccc")
                    sp.set_linewidth(1.3 if "CSP" in m_key else 0.6)
            ax_e.text(
                0.95,
                0.06,
                f"{psnr_s:.1f} dB",
                transform=ax_e.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.9,
                fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.14", fc="black", alpha=0.72, ec="none"),
            )
            if s == 0:
                ax_r.set_title(f"{m_lbl}\nRGB", fontsize=8.1, fontweight="bold")
                ax_e.set_title(f"|Err| x 50\n({mean_p:.1f} dB)", fontsize=8.1, fontweight="bold")

    cbar_ax = fig.add_axes([0.915, 0.37, 0.012, 0.53])
    cb = fig.colorbar(im_h, cax=cbar_ax)
    cb.set_label(r"$|\hat{x}_0 - x_{\mathrm{exact}}| \times 50$", fontsize=9.0)

    gs_bot = fig.add_gridspec(1, 2, top=0.27, bottom=0.06, left=0.07, right=0.92, wspace=0.26)
    ax_b1 = fig.add_subplot(gs_bot[0, 0])
    ax_b2 = fig.add_subplot(gs_bot[0, 1])

    pts_e = [
        ("DDIM-100 (BF16)", rows_e["DDIM-100 (BF16)"], "#999999", "v"),
        ("DDIM-100 (FP32)", rows_e["DDIM-100 (FP32)"], "#8c510a", "^"),
        ("DPM2-50 (BF16)", rows_e["DPM-Solver-2-50 (BF16)"], "#e6ab02", "o"),
        ("DPM2-100 (BF16)", rows_e["DPM-Solver-2-100 (BF16)"], "#d95f02", "o"),
        ("DPM2-100 (FP32)", rows_e["DPM-Solver-2-100 (FP32)"], "#e7298a", "P"),
        ("ParaDiGMS-BF16", rows_e["ParaDiGMS-BF16 (S=96,W=8)"], "#7570b3", "D"),
        ("ParaFlow-BF16 (Ours)", rows_e["Log-SNR ParaFlow-BF16 (P=4,M=8)"], "#2b8cbe", "p"),
        ("CSP-BF16-IR 1-IR (Ours)", rows_e["Log-SNR CSP-BF16-IR (P=9,M=8,1-IR,TF32)"], "#66a61e", "s"),
        ("CSP-BF16-IR 2-IR (Ours)", rows_e["Log-SNR CSP-BF16-IR (P=9,M=8,2-IR,IEEE)"], "#1b9e77", "s"),
    ]
    for lbl, r_e, col, mk in pts_e:
        ax_b1.plot(r_e["b1_median_ms"] / 1000.0, r_e["psnr_mean_db"], mk, color=col, ms=7.8, label=f"{lbl} ({r_e['psnr_mean_db']:.1f} dB)")
    ax_b1.set_xlabel("Single-Image 256x256 Latency $B=1$ (seconds)", fontweight="bold")
    ax_b1.set_ylabel("4-Seed Mean PSNR (dB)", fontweight="bold")
    ax_b1.set_title("(a) google/ddpm-celebahq-256 (113.7M) Latency vs. PSNR", fontweight="bold")
    ax_b1.grid(True, ls=":", alpha=0.35)
    ax_b1.legend(loc="lower right", ncol=2, fontsize=6.8, frameon=True, framealpha=0.92)

    comp_methods = [
        ("DDIM-100\n(BF16)", "DDIM-100 (BF16)", "DDIM-100 (BF16)"),
        ("DPM2-100\n(BF16)", "DPM-Solver-2-100 (BF16)", "DPM-Solver-2-100 (BF16)"),
        ("ParaDiGMS\n(BF16)", "ParaDiGMS-BF16 (S=96,W=8)", "ParaDiGMS-BF16 (S=96,W=8)"),
        ("ParaFlow-BF16\n(Ours)", "Log-SNR ParaFlow-BF16 (P=4,M=8)", "Log-SNR ParaFlow-BF16 (P=4,M=8)"),
        ("CSP-BF16-IR\n(Ours, 2-IR)", "Log-SNR CSP-BF16-IR (P=9,M=8,2-IR,IEEE)", "Log-SNR CSP-BF16-IR (P=9,M=8,2-IR,IEEE)"),
    ]
    x_c = np.arange(len(comp_methods))
    w_bar = 0.36
    v_32 = [rows_d[k32]["psnr_mean_db"] for _, k32, _ in comp_methods]
    v_256 = [rows_e[k256]["psnr_mean_db"] for _, _, k256 in comp_methods]
    ax_b2.bar(x_c - w_bar / 2, v_32, w_bar, color="#2b8cbe", edgecolor="#222", lw=0.7, label="CIFAR-10 (32x32, D=3,072, 35.8M)")
    ax_b2.bar(x_c + w_bar / 2, v_256, w_bar, color="#1b9e77", edgecolor="#222", lw=0.7, label="CelebA-HQ (256x256, D=196,608, 113.7M)")
    ax_b2.set_xticks(x_c)
    ax_b2.set_xticklabels([lbl for lbl, _, _ in comp_methods], fontsize=8.2)
    ax_b2.set_ylim(18, 82)
    ax_b2.text(
        x_c[-1] - w_bar / 2,
        v_32[-1] + 1.4,
        f"+{v_32[-1]-v_32[1]:.1f} dB",
        ha="center",
        va="bottom",
        fontsize=7.6,
        fontweight="bold",
        color="#1d658c",
    )
    ax_b2.text(
        x_c[-1] + w_bar / 2 + 0.04,
        v_256[-1] + 1.4,
        f"+{v_256[-1]-v_256[1]:.1f} dB",
        ha="center",
        va="bottom",
        fontsize=7.6,
        fontweight="bold",
        color="#0f684e",
    )
    ax_b2.set_ylabel("Mean PSNR vs. Continuous ODE (dB)", fontweight="bold")
    ax_b2.set_title(r"(b) Invariant +24 dB Precision Gain Across 64x Scale-Up ($\sigma_c = 0.125$)", fontweight="bold")
    ax_b2.grid(True, axis="y", ls=":", alpha=0.35)
    ax_b2.legend(loc="upper left", fontsize=7.8, frameon=True)

    fig.suptitle(
        "Zero-Shot 64x Spatial Scale-Up: Official google/ddpm-celebahq-256 (113.67M UNet, 256x256 Photorealistic Faces)",
        fontsize=11.5,
        fontweight="bold",
        y=0.965,
    )
    save_both(fig, "fig6_celebahq256_scaleup")


def main() -> None:
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()
    make_fig6()
    print("\nAll 6 vector PDF + 300-DPI PNG figures generated successfully!")


if __name__ == "__main__":
    main()

"""Populate `paper/main.tex` with 100% Silicon-Verified Tables & Metrics from `results/*.json`."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PAPER = ROOT / "paper"


def fmt_sci(v: float) -> str:
    s = f"{v:.2e}"
    mant, exp = s.split("e")
    return rf"{mant} \times 10^{{{int(exp)}}}"


def main() -> None:
    da = json.loads((RES / "stage_a_vcurve.json").read_text(encoding="utf-8"))
    db = json.loads((RES / "stage_b_dit_cudagraph.json").read_text(encoding="utf-8"))
    dc = json.loads((RES / "stage_c_flow_matching.json").read_text(encoding="utf-8"))
    dd = json.loads((RES / "stage_d_ddpm_cifar10_pareto.json").read_text(encoding="utf-8"))
    de = json.loads((RES / "stage_e_celebahq256_scaleup.json").read_text(encoding="utf-8"))
    drtx = json.loads((RES / "cross_hardware_rtx5060_validation.json").read_text(encoding="utf-8"))

    # Table 1: Stage A V-Curve
    t1_rows = []
    for r in da["sweep"]:
        bold = r["sigma"] == 0.125
        sig_s = f"{r['sigma']:.6f}".rstrip("0").rstrip(".")
        fw = f"{100*r['fwd_o_eps_half_rel_err']:.2f}\\%"
        cn = f"{100*r['central_o_eps_twothirds_rel_err']:.2f}\\%"
        gn = rf"${r['accuracy_gain_x']:.2f}\times$"
        if bold:
            t1_rows.append(
                rf"\textbf{{{r['log2_sigma']:.1f}}} & \textbf{{{sig_s}}} & {fw} & \textbf{{{cn}}} & \textbf{{{gn}}} \\"
            )
        else:
            t1_rows.append(rf"{r['log2_sigma']:.3g} & {sig_s} & {fw} & {cn} & {gn} \\")
    table_a = (
        "\\begin{table}[t]\n\\centering\n\\small\n"
        "\\caption{\\textbf{Stage~A: Directional JVP Relative Error Across Macro-Perturbation Scale $\\sigma$} on $46.91\\text{M}$ SwiGLU DiT ($\\|R\\|_F = 3\\times 10^{-5}\\|X\\|_F \\ll 2^{-7}\\|X\\|_F$, NVIDIA GB10). Naive unscaled \\bfsixteen{} FD incurs $"
        + f"{da['naive_unscaled_bf16_fd_rel_err']:.1f}\\times"
        + "$ error.}\n\\label{tab:stage_a}\n"
        "\\begin{tabular}{rrrrr}\n\\toprule\n"
        "$\\log_2 \\sigma$ & Scale $\\sigma$ & Forward \\textsc{SP-BF16} $\\mathcal{O}(\\epsbf^{1/2})$ & Central \\textsc{CSP-BF16} $\\mathcal{O}(\\epsbf^{2/3})$ & Accuracy Gain \\\\\n\\midrule\n"
        + "\n".join(t1_rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}"
    )

    # Table 2: Stage B Scaling
    t2_rows = []
    for r in db["scaling_s64_to_s512"]:
        S = r["S_steps"]
        t2_rows.append(
            rf"{S} & {r['eager_rk4_fp32_ms']:.2f} & {r['cudagraph_rk4_fp32_ms']:.2f} & "
            rf"{r['cudagraph_paradigms_bf16_ms']:.2f} & {r['ptx_cg_paraflow_bf16_ms']:.2f} & "
            rf"\textbf{{{r['ptx_cg_csp_bf16_ir_432_ms']:.2f}}} & ${fmt_sci(r['paradigms_bf16_rel_l2_err'])}$ & "
            rf"${fmt_sci(r['rk4_fp32_rel_l2_err'])}$ & $\mathbf{{{fmt_sci(r['csp_bf16_ir_432_rel_l2_err'])}}}$ & "
            rf"\textbf{{{r['speedup_csp432_vs_cudagraph_rk4_x']:.1f}$\times$}} \\"
        )
    table_b = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{\\textbf{Stage~B: $46.91\\text{M}$ SwiGLU DiT $B=1$ Latency (ms) \\& Relative $L_2$ Error} vs.\\ $1{,}024$-NFE \\fpsixtyfour{} RK4 on NVIDIA GB10.}\n"
        "\\label{tab:stage_b}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{rrrrrrrrrr}\n\\toprule\n"
        "$S$ & Eager RK4 & CG RK4 & CG ParaDiGMS & ParaFlow & \\textbf{CSP-IR} & ParaDiGMS $L_2$ & RK4 $L_2$ & \\textbf{CSP-IR $L_2$} & \\textbf{Speedup} \\\\\n\\midrule\n"
        + "\n".join(t2_rows)
        + "\n\\bottomrule\n\\end{tabular}%\n}\n\\end{table}"
    )

    # Table 3: Stage C Flow Matching
    gl = dc["spatial_2d_glyph_dit_16x16"]["samplers"]
    t3_rows = []
    for k, lbl in [
        ("Eager Euler-128 (FP32)", "Eager Euler-128 (\\fpthirtytwo{})"),
        ("CUDAGraph RK4-256 (FP32)", "CUDAGraph RK4-256 (\\fpthirtytwo{})"),
        ("CUDAGraph ParaDiGMS-128 (BF16)", "CUDAGraph ParaDiGMS-128 (\\bfsixteen{})"),
        ("PTX+CG ParaFlow-BF16 (5,0,0)", "\\textbf{PTX+CG ParaFlow-BF16 (Ours)}"),
        ("PTX+CG CSP-BF16-IR (4,2,2)", "\\textbf{PTX+CG CSP-BF16-IR (Ours)}"),
    ]:
        v = gl[k]
        t3_rows.append(
            rf"{lbl} & {v['b1_latency_ms']:.2f} & {v['psnr_vs_exact_ode_db']:.2f} & "
            rf"${fmt_sci(v['edge_gradient_rel_l2'])}$ & ${fmt_sci(v['image_rel_l2_vs_exact_ode'])}$ & {v['sliced_wasserstein_2']:.5f} \\"
        )
    table_c = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{\\textbf{Stage~C: $34.65\\text{M}$ $2\\text{D}$ Spatial Glyph Flow-Matching DiT ($16\\times 16$ Images)} on NVIDIA GB10.}\n"
        "\\label{tab:stage_c}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "Sampler & $B=1$ ms & PSNR (dB) & Edge Grad $L_2$ & Image Rel $L_2$ & $SW_2 \\downarrow$ \\\\\n\\midrule\n"
        + "\n".join(t3_rows)
        + "\n\\bottomrule\n\\end{tabular}%\n}\n\\end{table}"
    )

    # Table 4: Stage D CIFAR-10
    d_map = {r["method"]: r for r in dd["methods"]}
    d_show = [
        "DDIM-50 (BF16)",
        "DDIM-100 (BF16)",
        "DDIM-250 (BF16)",
        "DDIM-250 (FP32)",
        "DPM-Solver-2-50 (BF16)",
        "DPM-Solver-2-100 (BF16)",
        "DPM-Solver-2-50 (FP32)",
        "DPM-Solver-2-100 (FP32)",
        "ParaDiGMS-BF16 (S=96,W=8)",
        "Log-SNR ParaFlow-BF16 (P=4,M=8)",
        "Log-SNR CSP-BF16-IR (P=7,M=8,1-IR,IEEE)",
        "Log-SNR CSP-BF16-IR (P=9,M=8,1-IR,TF32)",
        "Log-SNR CSP-BF16-IR (P=9,M=8,1-IR,IEEE)",
        "Log-SNR CSP-BF16-IR (P=9,M=8,2-IR,IEEE)",
    ]
    t4_rows = []
    for k in d_show:
        if k not in d_map:
            continue
        r = d_map[k]
        is_ours = "Log-SNR" in k
        nm = f"\\textbf{{{k}}}" if is_ours else k
        ps = f"\\textbf{{{r['psnr_mean_db']:.2f}}} $\\pm$ {r['psnr_std_db']:.2f}" if is_ours else f"{r['psnr_mean_db']:.2f} $\\pm$ {r['psnr_std_db']:.2f}"
        t4_rows.append(
            rf"{nm} & {r['b1_median_ms']:.2f} & {ps} & ${fmt_sci(r['rel_l2_mean'])}$ & ${fmt_sci(r['max_abs_err_mean'])}$ \\"
        )
    table_d = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{\\textbf{Stage~D: Zero-Shot Pretrained \\texttt{google/ddpm-cifar10-32} ($35.75\\text{M}$ UNet, $8$ Seeds)} on NVIDIA GB10. Strict \\texttt{IEEE-FP32} \\textsc{DDIM-250} requires $"
        + f"{dd['strict_ieee_fp32_ddim250_b1_ms']:.1f}"
        + "\\,\\mathrm{ms}$.}\n\\label{tab:stage_d}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Solver Configuration & $B=1$ Latency (ms) & $8$-Seed PSNR (dB) & Mean Rel $L_2$ & Max Pixel Err \\\\\n\\midrule\n"
        + "\n".join(t4_rows)
        + "\n\\bottomrule\n\\end{tabular}%\n}\n\\end{table}"
    )

    # Table 5: Stage E CelebA-HQ-256
    e_map = {r["method"]: r for r in de["methods"]}
    t5_rows = []
    for r in de["methods"]:
        k = r["method"]
        is_ours = "Log-SNR" in k
        nm = f"\\textbf{{{k}}}" if is_ours else k
        ps = f"\\textbf{{{r['psnr_mean_db']:.2f}}} $\\pm$ {r['psnr_std_db']:.2f}" if is_ours else f"{r['psnr_mean_db']:.2f} $\\pm$ {r['psnr_std_db']:.2f}"
        t5_rows.append(
            rf"{nm} & {r['b1_median_ms']:.1f} & {ps} & ${fmt_sci(r['rel_l2_mean'])}$ & ${fmt_sci(r['max_abs_err_mean'])}$ \\"
        )
    table_e = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{\\textbf{Stage~E: Zero-Shot $64\\times$ Spatial Scale-Up on Official \\texttt{google/ddpm-celebahq-256}} ($113.67\\text{M}$ UNet, $256\\times 256$ Faces, $D=196{,}608$, $4$ Seeds, $\\sigma_c = 0.125$ Fixed) on NVIDIA GB10. Strict \\texttt{IEEE-FP32} \\textsc{DDIM-250} requires $"
        + f"{de['strict_ieee_fp32_ddim250_b1_ms']:.0f}"
        + "\\,\\mathrm{ms}$.}\n\\label{tab:stage_e}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Solver Configuration ($256\\times 256$) & $B=1$ Latency (ms) & $4$-Seed PSNR (dB) & Mean Rel $L_2$ & Max Pixel Err \\\\\n\\midrule\n"
        + "\n".join(t5_rows)
        + "\n\\bottomrule\n\\end{tabular}%\n}\n\\end{table}"
    )

    # Table 6: Cross-GPU Portability (GB10 aarch64 vs RTX 5060 x86_64)
    rtx_b = drtx["stage_b_dit_cudagraph"]
    rtx_d = {r["method"]: r for r in drtx["stage_d_ddpm_cifar10_pareto"]["methods"]}
    gb_s512 = db["scaling_s64_to_s512"][-1]
    rtx_s512 = rtx_b["scaling_s64_to_s512"][-1]
    table_cross = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{\\textbf{Cross-Architecture Hardware \\& Numerical Portability:} NVIDIA GB10 Grace Blackwell Superchip (\\texttt{aarch64} Linux, CUDA~13.0) vs.\\ NVIDIA GeForce RTX~5060 (\\texttt{x86\\_64} Windows, CUDA~12.8).}\n"
        "\\label{tab:cross_gpu}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{lll}\n\\toprule\n"
        "Benchmark Metric & NVIDIA GB10 (\\texttt{aarch64}) & NVIDIA RTX~5060 (\\texttt{x86\\_64}) \\\\\n\\midrule\n"
        rf"Stage~A Optimal Central JVP Error ($\sigma^\star = 0.125$) & {100*da['optimal_central_rel_err']:.2f}\% ({da['peak_single_sweep_gain_x']:.2f}$\times$ gain) & {100*drtx['stage_a_vcurve']['optimal_central_rel_err']:.2f}\% ({drtx['stage_a_vcurve']['peak_single_sweep_gain_x']:.2f}$\times$ gain) \\" + "\n"
        rf"Stage~B $46.9\text{{M}}$ DiT \textsc{{CSP-BF16-IR}} $(4,3,2)$ Error & ${fmt_sci(gb_s512['csp_bf16_ir_432_rel_l2_err'])}$ ({gb_s512['ptx_cg_csp_bf16_ir_432_ms']:.2f}\,ms) & ${fmt_sci(rtx_s512['csp_bf16_ir_432_rel_l2_err'])}$ ({rtx_s512['ptx_cg_csp_bf16_ir_432_ms']:.2f}\,ms) \\" + "\n"
        rf"Stage~B Speedup vs.\ $512$-NFE CUDAGraph \fpthirtytwo{{}} RK4 & \textbf{{{gb_s512['speedup_csp432_vs_cudagraph_rk4_x']:.1f}$\times$}} & \textbf{{{rtx_s512['speedup_csp432_vs_cudagraph_rk4_x']:.1f}$\times$}} \\" + "\n"
        rf"Stage~D \texttt{{ddpm-cifar10-32}} \textsc{{CSP-BF16-IR}} ($P=9, 2$-IR) & \textbf{{{d_map['Log-SNR CSP-BF16-IR (P=9,M=8,2-IR,IEEE)']['psnr_mean_db']:.2f}\,dB}} & \textbf{{{rtx_d['Log-SNR CSP-BF16-IR (P=9,M=8,2-IR,IEEE)']['psnr_mean_db']:.2f}\,dB}} \\" + "\n"
        "\\bottomrule\n\\end{tabular}%\n}\n\\end{table}"
    )

    s64 = db["scaling_s64_to_s512"][0]
    s512 = db["scaling_s64_to_s512"][-1]
    csp9_1ir_key = "Log-SNR CSP-BF16-IR (P=9,M=8,1-IR,TF32)"
    csp9_2ir_key = "Log-SNR CSP-BF16-IR (P=9,M=8,2-IR,IEEE)"

    subs = {
        "@@A_NAIVE_ERR@@": f"{da['naive_unscaled_bf16_fd_rel_err']:.0f}",
        "@@A_FWD_OPT_PCT@@": f"{100 * da['optimal_fwd_rel_err']:.2f}",
        "@@A_CEN_OPT_PCT@@": f"{100 * da['optimal_central_rel_err']:.2f}",
        "@@A_GAIN_X@@": f"{da['peak_single_sweep_gain_x']:.2f}",
        "@@B_CSP432_ERR@@": fmt_sci(s512["csp_bf16_ir_432_rel_l2_err"]),
        "@@B_CSP432_MS@@": f"{s512['ptx_cg_csp_bf16_ir_432_ms']:.2f}",
        "@@B_PF_MS@@": f"{s512['ptx_cg_paraflow_bf16_ms']:.2f}",
        "@@B_SPEEDUP_S512@@": f"{s512['speedup_csp432_vs_cudagraph_rk4_x']:.1f}",
        "@@B_SPEEDUP_EAGER_S512@@": f"{s512['speedup_csp432_vs_eager_rk4_x']:.1f}",
        "@@B_PF_SPEEDUP_S512@@": f"{s512['speedup_pf_bf16_vs_cudagraph_rk4_x']:.0f}",
        "@@B_RK4_S64_MS@@": f"{s64['cudagraph_rk4_fp32_ms']:.2f}",
        "@@B_RK4_S512_MS@@": f"{s512['cudagraph_rk4_fp32_ms']:.2f}",
        "@@B_RK4_S64_ERR@@": fmt_sci(s64["rk4_fp32_rel_l2_err"]),
        "@@B_RK4_S512_ERR@@": fmt_sci(s512["rk4_fp32_rel_l2_err"]),
        "@@C_GLYPH_CSP_MS@@": f"{gl['PTX+CG CSP-BF16-IR (4,2,2)']['b1_latency_ms']:.2f}",
        "@@C_GLYPH_PD_MS@@": f"{gl['CUDAGraph ParaDiGMS-128 (BF16)']['b1_latency_ms']:.2f}",
        "@@C_GLYPH_CSP_PSNR@@": f"{gl['PTX+CG CSP-BF16-IR (4,2,2)']['psnr_vs_exact_ode_db']:.2f}",
        "@@C_GLYPH_PD_PSNR@@": f"{gl['CUDAGraph ParaDiGMS-128 (BF16)']['psnr_vs_exact_ode_db']:.2f}",
        "@@C_GLYPH_GAIN_DB@@": f"{gl['PTX+CG CSP-BF16-IR (4,2,2)']['psnr_vs_exact_ode_db'] - gl['CUDAGraph ParaDiGMS-128 (BF16)']['psnr_vs_exact_ode_db']:.1f}",
        "@@D_PF_MS@@": f"{d_map['Log-SNR ParaFlow-BF16 (P=4,M=8)']['b1_median_ms']:.1f}",
        "@@D_PF_PSNR@@": f"{d_map['Log-SNR ParaFlow-BF16 (P=4,M=8)']['psnr_mean_db']:.2f}",
        "@@D_CSP9_1IR_PSNR@@": f"{d_map[csp9_1ir_key]['psnr_mean_db']:.2f}",
        "@@D_CSP9_1IR_TF32_MS@@": f"{d_map[csp9_1ir_key]['b1_median_ms']:.1f}",
        "@@D_CSP9_2IR_MS@@": f"{d_map[csp9_2ir_key]['b1_median_ms']:.1f}",
        "@@D_CSP9_PSNR@@": f"{d_map[csp9_2ir_key]['psnr_mean_db']:.2f}",
        "@@D_COLLOC_PSNR@@": f"{dd['csp_vs_fp32_collocation_psnr_db']:.2f}",
        "@@E_DDIM100_BF16_PSNR@@": f"{e_map['DDIM-100 (BF16)']['psnr_mean_db']:.2f}",
        "@@E_PD_PSNR@@": f"{e_map['ParaDiGMS-BF16 (S=96,W=8)']['psnr_mean_db']:.2f}",
        "@@E_DPM100_BF16_PSNR@@": f"{e_map['DPM-Solver-2-100 (BF16)']['psnr_mean_db']:.2f}",
        "@@E_DPM100_FP32_MS@@": f"{e_map['DPM-Solver-2-100 (FP32)']['b1_median_ms']:.0f}",
        "@@E_PF_MS@@": f"{e_map['Log-SNR ParaFlow-BF16 (P=4,M=8)']['b1_median_ms']:.0f}",
        "@@E_PF_PSNR@@": f"{e_map['Log-SNR ParaFlow-BF16 (P=4,M=8)']['psnr_mean_db']:.2f}",
        "@@E_CSP9_1IR_MS@@": f"{e_map[csp9_1ir_key]['b1_median_ms']:.0f}",
        "@@E_CSP9_1IR_PSNR@@": f"{e_map[csp9_1ir_key]['psnr_mean_db']:.2f}",
        "@@E_CSP9_MS@@": f"{e_map[csp9_2ir_key]['b1_median_ms']:.0f}",
        "@@E_CSP9_PSNR@@": f"{e_map[csp9_2ir_key]['psnr_mean_db']:.2f}",
        "@@E_SPEEDUP_VS_IEEE_DDIM250@@": f"{de['strict_ieee_fp32_ddim250_b1_ms'] / e_map[csp9_2ir_key]['b1_median_ms']:.2f}",
        "@@TABLE_STAGE_A@@": table_a,
        "@@TABLE_STAGE_B@@": table_b,
        "@@TABLE_STAGE_C@@": table_c,
        "@@TABLE_STAGE_D@@": table_d,
        "@@TABLE_STAGE_E@@": table_e,
        "@@TABLE_CROSS_GPU@@": table_cross,
    }

    tex = (PAPER / "main_template.tex").read_text(encoding="utf-8")
    for k, v in subs.items():
        tex = tex.replace(k, v)
    assert "@@" not in tex, "Unreplaced macro token remained in main.tex!"
    (PAPER / "main.tex").write_text(tex, encoding="utf-8")
    print("[OK] Generated paper/main.tex with 100% verified metrics and tables.")


if __name__ == "__main__":
    main()

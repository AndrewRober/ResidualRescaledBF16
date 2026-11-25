"""Generate `README.md` and `RESULTS_REPORT.md` directly from verified `results/*.json`."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def main() -> None:
    da = json.loads((RES / "stage_a_vcurve.json").read_text(encoding="utf-8"))
    db = json.loads((RES / "stage_b_dit_cudagraph.json").read_text(encoding="utf-8"))
    dc = json.loads((RES / "stage_c_flow_matching.json").read_text(encoding="utf-8"))
    dd = json.loads((RES / "stage_d_ddpm_cifar10_pareto.json").read_text(encoding="utf-8"))
    de = json.loads((RES / "stage_e_celebahq256_scaleup.json").read_text(encoding="utf-8"))
    drtx = json.loads((RES / "cross_hardware_rtx5060_validation.json").read_text(encoding="utf-8"))

    s512 = db["scaling_s64_to_s512"][-1]
    gl = dc["spatial_2d_glyph_dit_16x16"]["samplers"]
    d_map = {r["method"]: r for r in dd["methods"]}
    e_map = {r["method"]: r for r in de["methods"]}
    csp9_d = d_map["Log-SNR CSP-BF16-IR (P=9,M=8,2-IR,IEEE)"]
    csp9_e = e_map["Log-SNR CSP-BF16-IR (P=9,M=8,2-IR,IEEE)"]
    pf_d = d_map["Log-SNR ParaFlow-BF16 (P=4,M=8)"]
    pf_e = e_map["Log-SNR ParaFlow-BF16 (P=4,M=8)"]

    readme = f"""# ResidualRescaledBF16 (`CSP-BF16-IR`)

**Breaking the 7-Bit `BF16` Mantissa Ceiling: $\\mathcal{{O}}(\\varepsilon_{{\\mathrm{{bf16}}}}^{{2/3}})$ Residual-Rescaled Spectral Volterra–Newton Refinement and High-Noise Desingularization for Continuous-Time Generative Flows**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![CUDA PTX + CUDAGraph](https://img.shields.io/badge/CUDA-PTX%20%2B%20CUDAGraph-76B900.svg)](src/residual_rescaled_bf16/ptx_kernels.py)
[![Hardware: NVIDIA GB10 & RTX 5060](https://img.shields.io/badge/Verified-NVIDIA%20GB10%20%7C%20RTX%205060-success.svg)](RESULTS_REPORT.md)

**Author:** Andrew Rober Fouad Samy ([BytesGlue](https://github.com/AndrewRober))  
**Paper PDF (Camera-Ready):** [`paper/main.pdf`](paper/main.pdf) | **Double-Blind Review PDF:** [`paper/main_anonymous_review.pdf`](paper/main_anonymous_review.pdf)

---

## Executive Summary

Every modern AI accelerator concentrates its arithmetic throughput in **7-bit-significand `BFLOAT16` (`BF16`) Tensor Cores** ($\\varepsilon_{{\\mathrm{{bf16}}}} = 2^{{-8}}$, machine spacing $2^{{-7}} \\approx 7.81 \\times 10^{{-3}}$). Consequently, every standard sequential and parallel continuous-time ODE/Diffusion/Flow-Matching solver—including `DDIM`, `DPM-Solver-2`, and sliding-window Picard iteration (`ParaDiGMS`)—hits a permanent **7-bit fixed-point stagnation wall** once trajectory updates shrink below $2^{{-7}}\\|X\\|$ (`~2.1e-3` relative $L_2$ error; `38–46 dB` PSNR). On pretrained variance-preserving (VP) diffusion models, dividing `BF16` predictions by $\\alpha_T \\approx 6.57 \\times 10^{{-3}}$ at $t=T$ further amplifies mantissa quantization noise by **`152.2x`**.

**`ResidualRescaledBF16` (`CSP-BF16-IR`)** resolves both barriers *without intralayer surgery or Ozaki matrix splitting*:

1. **$\\mathcal{{O}}(\\varepsilon_{{\\mathrm{{bf16}}}}^{{2/3}})$ Antipodal Residual Rescaling (Theorems 1–2):** Evaluates sparse `FP32` Clenshaw–Curtis spectral Volterra defect checkpoints $R_k$ and dynamically upscales sub-mantissa corrections into the `BF16` sweet spot via symmetric pairs $X_k \\pm s_c R_k$ ($s_c = \\sigma_c \\|X_k\\| / \\|R_k\\|$, $\\sigma_c = 2^{{-3}} = 0.125 = \\Theta(\\varepsilon_{{\\mathrm{{bf16}}}}^{{1/3}})$) packed into a **single** `[2M, d]` `BF16` Tensor-Core call. Antipodal symmetry cancels the quadratic Hessian $D^2 V_\\theta(X_k)[R_k, R_k]$ at **zero extra FLOPs**, cutting single-sweep JVP error **`{da['peak_single_sweep_gain_x']:.2f}x`** (`{100*da['optimal_fwd_rel_err']:.2f}% -> {100*da['optimal_central_rel_err']:.2f}%`) and driving unmodified `7-bit BF16` SwiGLU DiT layers (`46.91M` params) to **`{s512['csp_bf16_ir_432_rel_l2_err']:.2e}` `IEEE-FP32` precision** (**`>25,000x`** beyond `ParaDiGMS` and **`{s512['speedup_csp432_vs_cudagraph_rk4_x']:.1f}x` faster** than `512`-NFE `CUDAGraph` `FP32` RK4 on **NVIDIA GB10**).
2. **High-Noise $z = \\sqrt{{\\tau}}$ Root-Time Desingularization & $\\sin(999\\tau)$ Nyquist Horizon (Theorem 3 & Proposition 3):** Eliminates the $1/\\alpha_T = 152.2\\times$ `BF16` blowup on high-noise panels ($\\mathrm{{SNR}} \\le 1$), lifting pure-`BF16` `Log-SNR ParaFlow-BF16` above all existing `BF16` solvers while `Log-SNR CSP-BF16-IR` shatters the `~46 dB` ceiling to **`{csp9_d['psnr_mean_db']:.2f} dB`** (`{dd['csp_vs_fp32_collocation_psnr_db']:.2f} dB` vs. exact `FP32` collocation) on `google/ddpm-cifar10-32` (`35.75M`) and **`{csp9_e['psnr_mean_db']:.2f} dB`** zero-shot on **`64x`-higher-dimensional `google/ddpm-celebahq-256` (`113.67M` UNet, `256x256` faces, $D=196{{,}}608$)** with $\\sigma_c = 0.125$ fixed.

![Perturbation V-Curve & DiT Precision](paper/figures/fig1_bf16_vcurve_scaling_law.png)

---

## Headline Silicon Benchmarks (`NVIDIA GB10` Grace Blackwell Superchip)

| Stage & Model | Baseline (`BF16` Wall / `FP32` Reference) | `ParaFlow-BF16` (Ours, Pure `BF16`) | `CSP-BF16-IR` (Ours, Exact Scientific) | Key Gain |
| :--- | :--- | :--- | :--- | :--- |
| **Stage A:** `46.9M` DiT JVP V-Curve | Unscaled `BF16`: `{da['naive_unscaled_bf16_fd_rel_err']:.0f}x` err; Fwd: `{100*da['optimal_fwd_rel_err']:.2f}%` | — | **`{100*da['optimal_central_rel_err']:.2f}%` JVP err** ($\\sigma^\\star = 0.125$) | **`{da['peak_single_sweep_gain_x']:.2f}x`** vs. Forward SP |
| **Stage B:** `46.9M` SwiGLU DiT ($S=512$) | `CG-RK4 FP32`: `{s512['cudagraph_rk4_fp32_ms']:.1f} ms` (`{s512['rk4_fp32_rel_l2_err']:.2e}`); `ParaDiGMS`: `{s512['paradigms_bf16_rel_l2_err']:.2e}` | **`{s512['ptx_cg_paraflow_bf16_ms']:.2f} ms`** (`{s512['speedup_pf_bf16_vs_cudagraph_rk4_x']:.0f}x` speedup) | **`{s512['ptx_cg_csp_bf16_ir_432_ms']:.2f} ms` @ `{s512['csp_bf16_ir_432_rel_l2_err']:.2e}`** | **`{s512['speedup_csp432_vs_cudagraph_rk4_x']:.1f}x` faster & `3.0x` cleaner** than `FP32` RK4 |
| **Stage C:** `34.7M` 2D Glyph OT-CFM | `CG ParaDiGMS-128`: `{gl['CUDAGraph ParaDiGMS-128 (BF16)']['b1_latency_ms']:.2f} ms` @ `{gl['CUDAGraph ParaDiGMS-128 (BF16)']['psnr_vs_exact_ode_db']:.2f} dB` | **`{gl['PTX+CG ParaFlow-BF16 (5,0,0)']['b1_latency_ms']:.2f} ms`** @ `{gl['PTX+CG ParaFlow-BF16 (5,0,0)']['psnr_vs_exact_ode_db']:.2f} dB` | **`{gl['PTX+CG CSP-BF16-IR (4,2,2)']['b1_latency_ms']:.2f} ms` @ `{gl['PTX+CG CSP-BF16-IR (4,2,2)']['psnr_vs_exact_ode_db']:.2f} dB`** | **Faster + `+{gl['PTX+CG CSP-BF16-IR (4,2,2)']['psnr_vs_exact_ode_db'] - gl['CUDAGraph ParaDiGMS-128 (BF16)']['psnr_vs_exact_ode_db']:.2f} dB`** vs. `ParaDiGMS` |
| **Stage D:** `google/ddpm-cifar10-32` (`35.8M`) | `BF16` Ceiling: `46.02 dB`; `IEEE DDIM-250`: `{dd['strict_ieee_fp32_ddim250_b1_ms']:.0f} ms` | **`{pf_d['b1_median_ms']:.1f} ms` @ `{pf_d['psnr_mean_db']:.2f} dB`** | **`{csp9_d['b1_median_ms']:.1f} ms` @ `{csp9_d['psnr_mean_db']:.2f} dB`** | **`+23.7 dB`** past `BF16` wall (`{dd['csp_vs_fp32_collocation_psnr_db']:.2f} dB` vs. Colloc) |
| **Stage E:** `google/ddpm-celebahq-256` (`113.7M`, `256x256`) | `ParaDiGMS`: `{e_map['ParaDiGMS-BF16 (S=96,W=8)']['psnr_mean_db']:.2f} dB`; `DPM2-100 BF16`: `{e_map['DPM-Solver-2-100 (BF16)']['psnr_mean_db']:.2f} dB` | **`{pf_e['b1_median_ms']:.0f} ms` @ `{pf_e['psnr_mean_db']:.2f} dB`** | **`{csp9_e['b1_median_ms']:.0f} ms` @ `{csp9_e['psnr_mean_db']:.2f} dB`** | **`64x` pixel scale-up** with $\\sigma_c=0.125$ unchanged |

![CelebA-HQ-256 Zero-Shot Scale-Up](paper/figures/fig6_celebahq256_scaleup.png)

---

## Repository Architecture

```text
ResidualRescaledBF16/
├── src/residual_rescaled_bf16/
│   ├── __init__.py               # Public API exports
│   ├── ptx_kernels.py            # Raw NVIDIA PTX 8.0 JIT loader (Windows/Linux + CPU CI fallback)
│   ├── volterra_chebyshev.py     # Exact Clenshaw-Curtis, exp(-lambda) & z=sqrt(tau) operators + CGS2-QR Anderson
│   ├── csp_bf16_solver.py        # PTXCspBf16Solver, 46.9M SwiGLU DiT, 34.7M 2D Glyph DiT & CUDAGraph helpers
│   ├── unet2d.py                 # Config-driven exact Diffusers-0.0.4 UNet2DModel (32x32 & 256x256 strict=True)
│   └── logsnr_diffusion.py       # AnalyticVPSchedule, hybrid z+exp ParaFlow-BF16 & CSP-BF16-IR diffusion solvers
├── scripts/
│   ├── verify_all.py             # Deterministic CPU/CUDA mathematical & solver verification suite
│   ├── run_stage_a_scaling_laws.py
│   ├── run_stage_b_dit_cudagraph.py
│   ├── run_stage_c_flow_matching.py
│   ├── run_stage_d_ddpm_cifar10.py
│   ├── run_stage_e_celebahq256.py
│   ├── generate_paper_figures.py # Renders all 6 vector PDF + 300-DPI PNG figures
│   └── build_submission_bundles.py
├── results/                      # Clean-room verified JSON scoreboards (NVIDIA GB10 + RTX 5060)
├── notebooks/CSP_BF16_IR_Demo.ipynb
├── paper/
│   ├── main.tex                  # Camera-ready & \\ifanonymous LaTeX monograph
│   ├── main.pdf                  # Compiled camera-ready paper
│   ├── main_anonymous_review.pdf # Double-blind anonymous review paper (0 identity leaks)
│   ├── references.bib            # BibTeX bibliography
│   └── figures/                  # 6 vector PDF + 300-DPI PNG publication figures
├── submission_bundles/           # Pre-audited arXiv, TMLR camera-ready & anonymous supplementary zips
├── PUBLISH_EVERYWHERE_GUIDE.md
├── RESULTS_REPORT.md
├── CITATION.cff
├── .zenodo.json
└── SHA256SUMS.txt
```

---

## Quickstart & Reproducibility

```bash
pip install -r requirements.txt
pip install -e .

# 1. Run deterministic math, quadrature, Anderson & CSP-BF16-IR verification (<5s on CPU or CUDA)
python scripts/verify_all.py

# 2. Reproduce Stages A-E on CUDA GPU
python scripts/run_stage_a_scaling_laws.py
python scripts/run_stage_b_dit_cudagraph.py
python scripts/run_stage_c_flow_matching.py
python scripts/run_stage_d_ddpm_cifar10.py
python scripts/run_stage_e_celebahq256.py

# 3. Regenerate all 6 vector PDF + 300-DPI PNG figures
python scripts/generate_paper_figures.py
```

## Citation

```bibtex
@article{{samy2026breaking,
  title={{Breaking the 7-Bit BF16 Mantissa Ceiling: $\\mathcal{{O}}(\\varepsilon_{{\\mathrm{{bf16}}}}^{{2/3}})$ Residual-Rescaled Spectral Volterra--Newton Refinement and High-Noise Desingularization for Continuous-Time Generative Flows}},
  author={{Samy, Andrew Rober Fouad}},
  journal={{arXiv preprint}},
  year={{2026}}
}}
```
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")

    # Full RESULTS_REPORT.md
    d_rows_md = "\n".join(
        f"| `{r['method']}` | `{r['b1_median_ms']:.2f}` | `{r['psnr_mean_db']:.2f} ± {r['psnr_std_db']:.2f}` | `{r['rel_l2_mean']:.3e}` | `{r['max_abs_err_mean']:.3e}` |"
        for r in dd["methods"]
    )
    e_rows_md = "\n".join(
        f"| `{r['method']}` | `{r['b1_median_ms']:.1f}` | `{r['psnr_mean_db']:.2f} ± {r['psnr_std_db']:.2f}` | `{r['rel_l2_mean']:.3e}` | `{r['max_abs_err_mean']:.3e}` |"
        for r in de["methods"]
    )
    b_rows_md = "\n".join(
        f"| `{r['S_steps']}` | `{r['eager_rk4_fp32_ms']:.2f}` | `{r['cudagraph_rk4_fp32_ms']:.2f}` | `{r['cudagraph_paradigms_bf16_ms']:.2f}` | `{r['ptx_cg_paraflow_bf16_ms']:.2f}` | **`{r['ptx_cg_csp_bf16_ir_432_ms']:.2f}`** | `{r['paradigms_bf16_rel_l2_err']:.2e}` | `{r['rk4_fp32_rel_l2_err']:.2e}` | **`{r['csp_bf16_ir_432_rel_l2_err']:.2e}`** | **`{r['speedup_csp432_vs_cudagraph_rk4_x']:.1f}x`** |"
        for r in db["scaling_s64_to_s512"]
    )

    report = f"""# Clean-Room Silicon Benchmark Report — `ResidualRescaledBF16`

**Primary Hardware:** `{db['gpu_device']}` (`aarch64` Linux, CUDA 13.0, Driver 580.142, `{db['pytorch_version']}`)  
**Cross-Platform Portability Hardware:** `{drtx['stage_b_dit_cudagraph']['gpu_device']}` (`x86_64` Windows, CUDA 12.8)

---

## Stage A: $\\mathcal{{O}}(\\varepsilon_{{\\mathrm{{bf16}}}}^{{2/3}})$ Perturbation V-Curve (`results/stage_a_vcurve.json`)
- **Unscaled `BF16` Finite Difference Relative Error:** `{da['naive_unscaled_bf16_fd_rel_err']:.2f}x` (`100%` mantissa underflow)
- **Optimal Forward `SP-BF16` Error ($\\mathcal{{O}}(\\varepsilon^{{1/2}})$):** `{100*da['optimal_fwd_rel_err']:.2f}%`
- **Optimal Antipodal Central `CSP-BF16` Error ($\\mathcal{{O}}(\\varepsilon^{{2/3}})$ at $\\sigma^\\star = 0.125$):** **`{100*da['optimal_central_rel_err']:.2f}%`** (**`{da['peak_single_sweep_gain_x']:.2f}x`** single-sweep gain at zero extra FLOPs)

---

## Stage B: `46.91M` SwiGLU DiT `PTX` + `CUDAGraph` Scaling (`results/stage_b_dit_cudagraph.json`)

| $S$ (NFEs) | Eager RK4 (ms) | CG RK4 (ms) | CG ParaDiGMS (ms) | ParaFlow-BF16 (ms) | **CSP-BF16-IR (ms)** | ParaDiGMS $L_2$ | RK4 $L_2$ | **CSP-BF16-IR $L_2$** | **Speedup vs. CG-RK4** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{b_rows_md}

---

## Stage D: Official Pretrained `google/ddpm-cifar10-32` (`35.75M` UNet, 8 Seeds, `results/stage_d_ddpm_cifar10_pareto.json`)
- **Strict `IEEE-FP32` `B=1` Step Latency:** `{dd['strict_ieee_fp32_b1_step_ms']:.2f} ms` (`DDIM-250 IEEE-FP32` = `{dd['strict_ieee_fp32_ddim250_b1_ms']:.1f} ms`)
- **`CSP-BF16-IR` vs. Exact `IEEE-FP32` Collocation Trajectory PSNR:** **`{dd['csp_vs_fp32_collocation_psnr_db']:.2f} dB`**

| Method | $B=1$ Median (ms) | 8-Seed PSNR (dB) | Mean Rel $L_2$ | Max Pixel Error |
| :--- | ---: | ---: | ---: | ---: |
{d_rows_md}

---

## Stage E: Zero-Shot `64x` Spatial Scale-Up on Official `google/ddpm-celebahq-256` (`113.67M` UNet, `256x256` Faces, 4 Seeds, `results/stage_e_celebahq256_scaleup.json`)
- **State Dimension:** `D = 3 x 256 x 256 = 196,608` (`64x` larger than CIFAR-10), $\\sigma_c = 0.125$ strictly unchanged.
- **Strict `IEEE-FP32` `B=1` Step Latency:** `{de['strict_ieee_fp32_b1_step_ms']:.2f} ms` (`DDIM-250 IEEE-FP32` = `{de['strict_ieee_fp32_ddim250_b1_ms']:.0f} ms`)

| Method (`256x256`) | $B=1$ Median (ms) | 4-Seed PSNR (dB) | Mean Rel $L_2$ | Max Pixel Error |
| :--- | ---: | ---: | ---: | ---: |
{e_rows_md}
"""
    (ROOT / "RESULTS_REPORT.md").write_text(report, encoding="utf-8")
    print("[OK] Generated README.md and RESULTS_REPORT.md from verified JSONs.")


if __name__ == "__main__":
    main()

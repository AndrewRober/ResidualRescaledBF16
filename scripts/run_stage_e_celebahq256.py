"""Stage E Benchmark: Zero-Shot 64x Spatial Scale-Up on `google/ddpm-celebahq-256` (113.67M UNet, 256x256).

Validates dimension-independent precision and latency scaling of `Log-SNR ParaFlow-BF16` and
`Log-SNR CSP-BF16-IR` when scaling state dimension 64x (from 3,072 CIFAR-10 dims to 196,608
CelebA-HQ dims) and model size 3.18x (from 35.75M to 113.67M parameters) with zero hyperparameter
retuning (`sigma_c = 0.125 = 2^{-3}`).
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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from residual_rescaled_bf16 import (
    AnalyticVPSchedule,
    DiffusersUNet2D,
    build_ddim_schedule,
    build_dpm_solver2_schedule,
    precompute_csp_ir_panels,
    precompute_paradigms_windows,
    precompute_paraflow_panels,
    run_hybrid_csp_bf16_ir,
    run_paradigms_bf16,
)


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
    torch.cuda.set_per_process_memory_fraction(0.25 if tot_gb > 32 else 0.85)
    torch.backends.cudnn.benchmark = True

    ckpt_dir = ROOT / ".cache" / "ddpm_celebahq_256"
    if not (ckpt_dir / "config.json").exists():
        from huggingface_hub import hf_hub_download

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        for fn in ["config.json", "scheduler_config.json", "diffusion_pytorch_model.bin"]:
            hf_hub_download("google/ddpm-celebahq-256", fn, local_dir=str(ckpt_dir))

    gpu_name = torch.cuda.get_device_name(0)
    print(f"=== Stage E: google/ddpm-celebahq-256 (256x256, 113.67M) Scale-Up on {gpu_name} ===")

    unet_fp32 = (
        DiffusersUNet2D.from_pretrained_dir(str(ckpt_dir))
        .to(device="cuda", memory_format=torch.channels_last)
        .eval()
    )
    unet_bf16 = (
        copy.deepcopy(unet_fp32)
        .to(dtype=torch.bfloat16, memory_format=torch.channels_last)
        .eval()
    )
    n_params = sum(p.numel() for p in unet_fp32.parameters())
    print(f"[1] Loaded google/ddpm-celebahq-256 (strict=True, {n_params / 1e6:.2f}M params, D=196,608)")

    seeds = [0, 1, 2, 3]
    xT4 = torch.cat(
        [
            torch.randn(
                1,
                3,
                256,
                256,
                generator=torch.Generator(device="cuda").manual_seed(s),
                device="cuda",
            )
            for s in seeds
        ],
        dim=0,
    )
    vp = AnalyticVPSchedule()

    with torch.inference_mode():
        # --------------------------------------------------------------------
        # 1. Strict 24-bit IEEE-FP32 Richardson Exp-Heun(40, 20) Exact Reference
        # --------------------------------------------------------------------
        t_ref_0 = time.perf_counter()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        # Measure strict IEEE-FP32 B=1 single-step latency
        x1_probe = xT4[0:1].to(memory_format=torch.channels_last)
        t1_probe = torch.zeros(1, dtype=torch.float32, device="cuda")
        _ = unet_fp32(x1_probe, t1_probe)
        torch.cuda.synchronize()
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        ev0.record()
        for _ in range(4):
            _ = unet_fp32(x1_probe, t1_probe)
        ev1.record()
        torch.cuda.synchronize()
        ieee_b1_step_ms = float(ev0.elapsed_time(ev1)) / 4.0

        def run_heun_b4(N_steps: int) -> torch.Tensor:
            lg = np.linspace(vp.lam_min, vp.lam_max, N_steps + 1)
            _, tg, ag, _ = vp.lambda_to_tau_t_alpha_sigma(lg)
            xc = xT4.to(memory_format=torch.channels_last).clone()
            tb = torch.empty(4, dtype=torch.float32, device="cuda")
            for i in range(N_steps):
                la, lb = lg[i], lg[i + 1]
                dlam = lb - la
                ha, hb = math.exp(-la), math.exp(-lb)
                aa, ab = float(ag[i]), float(ag[i + 1])
                tb.fill_(float(tg[i]))
                e1 = unet_fp32(xc, tb)
                xp = ((ab / aa) * xc + ab * (hb - ha) * e1).to(memory_format=torch.channels_last)
                tb.fill_(float(tg[i + 1]))
                e2 = unet_fp32(xp, tb)
                wt = hb - ha
                w2 = hb + wt / dlam
                xc = ((ab / aa) * xc + ab * ((wt - w2) * e1 + w2 * e2)).to(
                    memory_format=torch.channels_last
                )
            return xc.contiguous()

        x_ref4 = (4.0 * run_heun_b4(40) - run_heun_b4(20)) / 3.0
        torch.cuda.synchronize()
        print(
            f"[2] Strict IEEE-FP32 4-seed 256x256 Exact Reference computed in {time.perf_counter() - t_ref_0:.2f}s "
            f"(IEEE-FP32 B=1 step = {ieee_b1_step_ms:.2f} ms)"
        )

        # --------------------------------------------------------------------
        # 2. Evaluate Strict IEEE-FP32 Checkpoint CSP-BF16-IR first (allow_tf32=False)
        # --------------------------------------------------------------------
        outputs_4: dict[str, torch.Tensor] = {}
        timings_ms: dict[str, list[float]] = {}

        def eval_b1_4seeds(name: str, fn_b1) -> None:
            torch.cuda.empty_cache()
            # 1 warmup on Seed 0 to prime cuDNN workspace & L2
            _ = fn_b1(xT4[0:1])
            torch.cuda.synchronize()
            outs = []
            ts = []
            for s_idx in range(4):
                xi = xT4[s_idx : s_idx + 1]
                ea, eb = torch.cuda.Event(True), torch.cuda.Event(True)
                ea.record()
                yi = fn_b1(xi)
                eb.record()
                torch.cuda.synchronize()
                ts.append(float(ea.elapsed_time(eb)))
                outs.append(yi.contiguous())
            timings_ms[name] = ts
            outputs_4[name] = torch.cat(outs, dim=0)

        csp9_panels = precompute_csp_ir_panels(vp, P=9, M=8, hybrid_z=False, device="cuda")
        paraflow4 = precompute_paraflow_panels(vp, M=8, device="cuda")
        paradigms_wins = precompute_paradigms_windows(vp, S=96, W=8, device="cuda")

        eval_b1_4seeds(
            "Log-SNR CSP-BF16-IR (P=9,M=8,2-IR,IEEE)",
            lambda x: run_hybrid_csp_bf16_ir(
                x, csp9_panels, unet_bf16, unet_fp32, bsw=2, oir=2, inner_jvp=1
            ),
        )

        # --------------------------------------------------------------------
        # 3. Enable Tensor-Core TF32 for Steelmanned Sequential FP32 & Fast CSP-IR
        # --------------------------------------------------------------------
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        eval_b1_4seeds(
            "Log-SNR CSP-BF16-IR (P=9,M=8,1-IR,TF32)",
            lambda x: run_hybrid_csp_bf16_ir(
                x, csp9_panels, unet_bf16, unet_fp32, bsw=1, oir=1, inner_jvp=1
            ),
        )
        eval_b1_4seeds(
            "Log-SNR ParaFlow-BF16 (P=4,M=8)",
            lambda x: run_hybrid_csp_bf16_ir(x, paraflow4, unet_bf16, None, bsw=5, oir=0),
        )
        eval_b1_4seeds(
            "ParaDiGMS-BF16 (S=96,W=8)",
            lambda x: run_paradigms_bf16(x, paradigms_wins, unet_bf16, sweeps=3),
        )

        # Sequential DDIM & DPM-Solver-2 baselines
        ddim100 = build_ddim_schedule(vp, 100)
        dpm50 = build_dpm_solver2_schedule(vp, 50)
        dpm100 = build_dpm_solver2_schedule(vp, 100)

        def run_ddim_b1(x1: torch.Tensor, sched, use_bf16: bool) -> torch.Tensor:
            xc = x1.to(memory_format=torch.channels_last).clone()
            tb = torch.empty(1, dtype=torch.float32, device="cuda")
            for t_val, c1, c2 in sched:
                tb.fill_(t_val)
                if use_bf16:
                    eo = unet_bf16(xc.to(torch.bfloat16), tb).float()
                else:
                    eo = unet_fp32(xc, tb)
                xc = (c1 * xc + c2 * eo).to(memory_format=torch.channels_last)
            return xc.contiguous()

        def run_dpm2_b1(x1: torch.Tensor, sched, use_bf16: bool) -> torch.Tensor:
            xc = x1.to(memory_format=torch.channels_last).clone()
            tb = torch.empty(1, dtype=torch.float32, device="cuda")
            for t0, tm, cm1, cm2, cb1, cb2 in sched:
                tb.fill_(t0)
                e0 = unet_bf16(xc.to(torch.bfloat16), tb).float() if use_bf16 else unet_fp32(xc, tb)
                xm = (cm1 * xc + cm2 * e0).to(memory_format=torch.channels_last)
                tb.fill_(tm)
                em = unet_bf16(xm.to(torch.bfloat16), tb).float() if use_bf16 else unet_fp32(xm, tb)
                xc = (cb1 * xc + cb2 * em).to(memory_format=torch.channels_last)
            return xc.contiguous()

        eval_b1_4seeds("DDIM-100 (BF16)", lambda x: run_ddim_b1(x, ddim100, True))
        eval_b1_4seeds("DDIM-100 (FP32)", lambda x: run_ddim_b1(x, ddim100, False))
        eval_b1_4seeds("DPM-Solver-2-50 (BF16)", lambda x: run_dpm2_b1(x, dpm50, True))
        eval_b1_4seeds("DPM-Solver-2-50 (FP32)", lambda x: run_dpm2_b1(x, dpm50, False))
        eval_b1_4seeds("DPM-Solver-2-100 (BF16)", lambda x: run_dpm2_b1(x, dpm100, True))
        eval_b1_4seeds("DPM-Solver-2-100 (FP32)", lambda x: run_dpm2_b1(x, dpm100, False))

        method_order = [
            "DDIM-100 (FP32)",
            "DDIM-100 (BF16)",
            "DPM-Solver-2-50 (FP32)",
            "DPM-Solver-2-50 (BF16)",
            "DPM-Solver-2-100 (FP32)",
            "DPM-Solver-2-100 (BF16)",
            "ParaDiGMS-BF16 (S=96,W=8)",
            "Log-SNR ParaFlow-BF16 (P=4,M=8)",
            "Log-SNR CSP-BF16-IR (P=9,M=8,1-IR,TF32)",
            "Log-SNR CSP-BF16-IR (P=9,M=8,2-IR,IEEE)",
        ]

        results_rows = []
        print(f"\n{'Method':42s} | {'B=1 ms':>9s} | {'4-Seed PSNR (dB)':>18s} | {'Rel L2':>10s}")
        print("-" * 89)
        for name in method_order:
            t_list = sorted(timings_ms[name])
            med_ms = 0.5 * (t_list[1] + t_list[2])
            met = compute_metrics(outputs_4[name], x_ref4)
            row = {"method": name, "b1_median_ms": round(med_ms, 2), **met}
            results_rows.append(row)
            print(
                f"{name:42s} | {med_ms:9.2f} | {met['psnr_mean_db']:7.2f} +- {met['psnr_std_db']:4.2f} dB | {met['rel_l2_mean']:10.3e}"
            )

        # Save cached 256x256 visual arrays for Figure 6
        vis_save = {
            "x_ref4": x_ref4.cpu().numpy(),
            "DDIM-100 (BF16)": outputs_4["DDIM-100 (BF16)"].cpu().numpy(),
            "DPM-Solver-2-100 (BF16)": outputs_4["DPM-Solver-2-100 (BF16)"].cpu().numpy(),
            "ParaDiGMS-BF16 (S=96,W=8)": outputs_4["ParaDiGMS-BF16 (S=96,W=8)"].cpu().numpy(),
            "Log-SNR ParaFlow-BF16 (P=4,M=8)": outputs_4["Log-SNR ParaFlow-BF16 (P=4,M=8)"].cpu().numpy(),
            "Log-SNR CSP-BF16-IR (P=9,M=8,1-IR,TF32)": outputs_4["Log-SNR CSP-BF16-IR (P=9,M=8,1-IR,TF32)"].cpu().numpy(),
            "Log-SNR CSP-BF16-IR (P=9,M=8,2-IR,IEEE)": outputs_4["Log-SNR CSP-BF16-IR (P=9,M=8,2-IR,IEEE)"].cpu().numpy(),
        }
        np.savez_compressed(ckpt_dir / "celebahq256_vis_arrays.npz", **vis_save)

        peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024**2))
        total_wall_s = time.perf_counter() - t_wall_0
        out_json = ROOT / "results" / "stage_e_celebahq256_scaleup.json"
        summary = {
            "stage": "Stage E — Zero-Shot 64x Spatial Scale-Up on google/ddpm-celebahq-256 (113.67M UNet, 256x256)",
            "gpu": gpu_name,
            "pytorch_version": torch.__version__,
            "model": "google/ddpm-celebahq-256 (113.67M UNet2DModel, strict=True, D=196,608)",
            "num_seeds": 4,
            "seeds": seeds,
            "strict_ieee_fp32_b1_step_ms": round(ieee_b1_step_ms, 2),
            "strict_ieee_fp32_ddim250_b1_ms": round(250.0 * ieee_b1_step_ms, 2),
            "strict_ieee_fp32_dpm100_b1_ms": round(100.0 * ieee_b1_step_ms, 2),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "total_benchmark_wall_s": round(total_wall_s, 2),
            "methods": results_rows,
        }
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            f"\n[Done] Saved {out_json} | Peak VRAM: {peak_vram_mb:.1f} MB | Wall: {total_wall_s:.2f}s"
        )


if __name__ == "__main__":
    main()

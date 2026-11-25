"""Stage B Benchmark: PTX+CUDAGraph CSP-BF16-IR on 46.91M SwiGLU DiT Across S in {64, 128, 256, 512}."""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from residual_rescaled_bf16 import (
    PTXCspBf16Solver,
    PTXDiTVelocityOperator,
    capture_cudagraph,
    gpu_paradigms_body,
    gpu_seq_euler_body,
    gpu_seq_rk4_body,
)


def time_graph(g: torch.cuda.CUDAGraph, iters: int = 15) -> float:
    times = []
    for _ in range(iters):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        times.append(float(e0.elapsed_time(e1)))
    times.sort()
    return times[len(times) // 2]


def time_eager(fn, iters: int = 5) -> float:
    fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        times.append(float(e0.elapsed_time(e1)))
    times.sort()
    return times[len(times) // 2]


def main() -> None:
    t0 = time.perf_counter()
    tot_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    torch.cuda.set_per_process_memory_fraction(0.25 if tot_gb > 32 else 0.40)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    gpu_name = torch.cuda.get_device_name(0)
    print(f"=== Stage B: 46.91M SwiGLU DiT PTX+CUDAGraph Scaling on {gpu_name} ===")

    net_fp32 = PTXDiTVelocityOperator(d=1280, h=3072, n_blocks=3, seed=42).to("cuda").eval()
    net_fp64 = copy.deepcopy(net_fp32).to(torch.float64).eval()
    net_bf16 = copy.deepcopy(net_fp32).to(torch.bfloat16).eval()

    torch.manual_seed(7)
    x0 = torch.randn(1, 1280, device="cuda", dtype=torch.float32)

    x_ref = torch.empty(1, 1280, device="cuda", dtype=torch.float64)
    gpu_seq_rk4_body(net_fp64, x0.double(), x_ref, S_nfe=1024, use_bf16=False)
    x_ref32 = x_ref.float()
    ref_norm = float(torch.linalg.norm(x_ref32).cpu())

    solver = PTXCspBf16Solver(net_bf16, net_fp32, M=16, d_state=1280, h_ff=3072, device="cuda")
    x_in = x0.clone()
    x_out = torch.empty_like(x0)

    fwd_vs_central_ablation = {}
    for label, cfg in [
        ("PTX+CUDAGraph ParaFlow-BF16 (4,0,0)", (4, 0, 0, True)),
        ("PTX+CUDAGraph Fwd SP-BF16-IR (4,3,1) [O(eps^1/2)]", (4, 3, 1, False)),
        ("PTX+CUDAGraph Central CSP-BF16-IR (4,2,2) [O(eps^2/3)]", (4, 2, 2, True)),
        ("PTX+CUDAGraph Central CSP-BF16-IR (4,3,1) [O(eps^2/3)]", (4, 3, 1, True)),
        ("PTX+CUDAGraph Central CSP-BF16-IR (4,3,2) [O(eps^2/3)]", (4, 3, 2, True)),
    ]:
        bf_i, oir, ijvp, cen = cfg
        g = capture_cudagraph(
            lambda bf_i=bf_i, oir=oir, ijvp=ijvp, cen=cen: solver.run_b1_into(
                x_in, x_out, bf_init=bf_i, outer_ir=oir, inner_jvp=ijvp, sigma=0.125, central=cen
            )
        )
        ms = time_graph(g)
        err = float(torch.linalg.norm(x_out - x_ref32).cpu()) / ref_norm
        fwd_vs_central_ablation[label] = {"cuda_ms": round(ms, 4), "rel_l2_err_vs_fp64": err}
        print(f"{label:56s} | {ms:7.3f} ms | Rel L2 = {err:.3e}")
        del g

    pf_ms = fwd_vs_central_ablation["PTX+CUDAGraph ParaFlow-BF16 (4,0,0)"]["cuda_ms"]
    csp431_ms = fwd_vs_central_ablation["PTX+CUDAGraph Central CSP-BF16-IR (4,3,1) [O(eps^2/3)]"]["cuda_ms"]
    csp432_ms = fwd_vs_central_ablation["PTX+CUDAGraph Central CSP-BF16-IR (4,3,2) [O(eps^2/3)]"]["cuda_ms"]
    csp431_err = fwd_vs_central_ablation["PTX+CUDAGraph Central CSP-BF16-IR (4,3,1) [O(eps^2/3)]"]["rel_l2_err_vs_fp64"]
    csp432_err = fwd_vs_central_ablation["PTX+CUDAGraph Central CSP-BF16-IR (4,3,2) [O(eps^2/3)]"]["rel_l2_err_vs_fp64"]

    scaling_s64_to_s512 = []
    for S in [64, 128, 256, 512]:
        eager_rk4_ms = time_eager(lambda S=S: gpu_seq_rk4_body(net_fp32, x_in, x_out, S_nfe=S), iters=3)
        eager_eu_ms = time_eager(lambda S=S: gpu_seq_euler_body(net_fp32, x_in, x_out, S=S), iters=3)
        err_eu = float(torch.linalg.norm(x_out - x_ref32).cpu()) / ref_norm

        g_rk4 = capture_cudagraph(lambda S=S: gpu_seq_rk4_body(net_fp32, x_in, x_out, S_nfe=S))
        ms_rk4 = time_graph(g_rk4, iters=7)
        err_rk4 = float(torch.linalg.norm(x_out - x_ref32).cpu()) / ref_norm
        del g_rk4

        g_pd = capture_cudagraph(
            lambda S=S: gpu_paradigms_body(net_bf16, x_in, x_out, S=S, W=32, K_sweeps=4)
        )
        ms_pd = time_graph(g_pd, iters=9)
        err_pd = float(torch.linalg.norm(x_out - x_ref32).cpu()) / ref_norm
        del g_pd

        row = {
            "S_steps": S,
            "eager_rk4_fp32_ms": round(eager_rk4_ms, 3),
            "eager_euler_fp32_ms": round(eager_eu_ms, 3),
            "cudagraph_rk4_fp32_ms": round(ms_rk4, 3),
            "cudagraph_paradigms_bf16_ms": round(ms_pd, 3),
            "ptx_cg_paraflow_bf16_ms": round(pf_ms, 3),
            "ptx_cg_csp_bf16_ir_431_ms": round(csp431_ms, 3),
            "ptx_cg_csp_bf16_ir_432_ms": round(csp432_ms, 3),
            "speedup_csp432_vs_cudagraph_rk4_x": round(ms_rk4 / csp432_ms, 3),
            "speedup_csp432_vs_eager_rk4_x": round(eager_rk4_ms / csp432_ms, 3),
            "speedup_pf_bf16_vs_cudagraph_rk4_x": round(ms_rk4 / pf_ms, 3),
            "rk4_fp32_rel_l2_err": err_rk4,
            "euler_fp32_rel_l2_err": err_eu,
            "paradigms_bf16_rel_l2_err": err_pd,
            "csp_bf16_ir_431_rel_l2_err": csp431_err,
            "csp_bf16_ir_432_rel_l2_err": csp432_err,
        }
        scaling_s64_to_s512.append(row)
        print(
            f"  S={S:3d} | CG-RK4={ms_rk4:7.2f} ms ({err_rk4:.2e}) | "
            f"CSP-432={csp432_ms:5.2f} ms ({csp432_err:.2e}, {ms_rk4/csp432_ms:.1f}x)"
        )

    out_path = ROOT / "results" / "stage_b_dit_cudagraph.json"
    out_path.write_text(
        json.dumps(
            {
                "stage": "Stage B — 46.91M SwiGLU DiT PTX + CUDAGraph Trajectory Scaling",
                "gpu_device": gpu_name,
                "pytorch_version": torch.__version__,
                "reference": "FP64 1024-NFE Classical RK4 Exact Trajectory",
                "peak_vram_mb": round(float(torch.cuda.max_memory_allocated() / (1024**2)), 2),
                "fwd_vs_central_ablation": fwd_vs_central_ablation,
                "scaling_s64_to_s512": scaling_s64_to_s512,
                "wall_s": round(time.perf_counter() - t0, 2),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[Done] Saved {out_path}")


if __name__ == "__main__":
    main()

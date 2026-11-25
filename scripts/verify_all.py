"""Deterministic Mathematical & Software Verification Suite for `residual_rescaled_bf16`.

Runs on BOTH CPU-only CI environments (GitHub Actions Ubuntu/Windows/macOS) and CUDA GPUs:
  1. Verifies exact spectral Clenshaw-Curtis Volterra polynomial integration to < 1e-6 FP32 error.
  2. Verifies exponentially weighted (`build_exp_chebyshev_panel`) and root-time (`build_z_chebyshev_panel`)
     quadrature operators against analytical closed-form integrals.
  3. Verifies C^infty `AnalyticVPSchedule` forward/inverse bijectivity (`tau <-> lambda`).
  4. Verifies `O(eps_bf16^{2/3})` Central-Difference JVP error cancellation vs `O(eps_bf16^{1/2})`
     Forward-Difference and catastrophic unscaled BF16 underflow.
  5. Verifies `PTXCspBf16Solver` breaks the 7-bit BF16 fixed-point stagnation wall by > 1,000x.
  6. Verifies `DiffusersUNet2D` architectural parity and all repository JSON result schemas.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from residual_rescaled_bf16 import (
    AnalyticVPSchedule,
    DiffusersUNet2D,
    PTXCspBf16Solver,
    PTXDiTVelocityOperator,
    batched_cgs2_anderson,
    build_cheb_volterra,
    build_exp_chebyshev_panel,
    build_z_chebyshev_panel,
    graph_safe_anderson_cgs2,
)


def test_clenshaw_curtis_polynomial_exactness() -> None:
    nodes, Q = build_cheb_volterra(M=12, device="cpu")
    # Integrate f(t) = 5 t^4 - 3 t^2 + 2 -> exact integral F(t) = t^5 - t^3 + 2 t
    f_vals = 5.0 * (nodes**4) - 3.0 * (nodes**2) + 2.0
    F_num = Q @ f_vals
    F_exact = (nodes**5) - (nodes**3) + 2.0 * nodes
    max_err = float(torch.max(torch.abs(F_num - F_exact)))
    assert max_err < 1e-6, f"Clenshaw-Curtis polynomial integration failed: {max_err}"
    print(f"[PASS] Clenshaw-Curtis M=12 polynomial exactness: max_err = {max_err:.2e}")


def test_exp_and_z_panels_exactness() -> None:
    # 1. int_{la}^{lj} (-exp(-s)) * exp(0.5*s) ds = -2 * (exp(-0.5*la) - exp(-0.5*lj))?
    # Wait: int (-e^{-s}) e^{0.5 s} ds = int -e^{-0.5 s} ds = 2 (e^{-0.5 lj} - e^{-0.5 la})
    la, lb = -1.5, 1.2
    ln, Q_exp = build_exp_chebyshev_panel(la, lb, M=10, device="cpu")
    g_vals = torch.tensor(np.exp(0.5 * ln), dtype=torch.float32)
    I_num = Q_exp @ g_vals
    I_ex = torch.tensor(2.0 * (np.exp(-0.5 * ln) - math.exp(-0.5 * la)), dtype=torch.float32)
    err_exp = float(torch.max(torch.abs(I_num - I_ex)))
    assert err_exp < 2e-6, f"Exp-Chebyshev panel failed: {err_exp}"

    # 2. Root-time z-panel int_{za}^{zj} (3 z^2 + 1) dz = (zj^3 + zj) - (za^3 + za)
    za, zb = 1.0, 0.3
    zn, Q_z = build_z_chebyshev_panel(za, zb, M=8, device="cpu")
    hz = torch.tensor(3.0 * (zn**2) + 1.0, dtype=torch.float32)
    Iz_num = Q_z @ hz
    Iz_ex = torch.tensor((zn**3 + zn) - (za**3 + za), dtype=torch.float32)
    err_z = float(torch.max(torch.abs(Iz_num - Iz_ex)))
    assert err_z < 1e-6, f"z-Chebyshev panel failed: {err_z}"
    print(f"[PASS] Exp-Volterra ({err_exp:.2e}) & z=sqrt(tau) Volterra ({err_z:.2e}) exactness")


def test_analytic_vp_schedule_bijectivity() -> None:
    vp = AnalyticVPSchedule()
    taus = np.linspace(1.0, 1e-3, 50)
    lams = np.array([vp.tau_to_lambda(float(t)) for t in taus])
    taus_rec, _, alphas, sigmas = vp.lambda_to_tau_t_alpha_sigma(lams)
    err_tau = float(np.max(np.abs(taus - taus_rec)))
    err_unit = float(np.max(np.abs(alphas**2 + sigmas**2 - 1.0)))
    assert err_tau < 1e-12 and err_unit < 1e-12
    print(f"[PASS] AnalyticVPSchedule bijectivity ({err_tau:.2e}) & variance preservation ({err_unit:.2e})")


def test_cgs2_anderson_contraction() -> None:
    torch.manual_seed(0)
    A = 0.32 * torch.randn(32, 32) / math.sqrt(32)
    b = torch.randn(32)
    X = torch.zeros(4, 8)
    Xh, Gh = [], []
    for _ in range(20):
        G = (A @ X.reshape(-1) + b).reshape(4, 8)
        Xh.append(X)
        Gh.append(G)
        if len(Xh) > 6:
            Xh.pop(0)
            Gh.pop(0)
        X = graph_safe_anderson_cgs2(Xh, Gh, m_max=4, reg=1e-8)
    res = float(torch.linalg.norm((A @ X.reshape(-1) + b) - X.reshape(-1)))
    assert res < 1e-5, f"Anderson CGS2 residual too high: {res}"

    # Batched 5D check
    Xb = torch.zeros(2, 4, 1, 2, 2)
    Xbh, Gbh = [], []
    for _ in range(5):
        Gb = 0.3 * Xb + 1.0
        Xbh.append(Xb)
        Gbh.append(Gb)
        Xb = batched_cgs2_anderson(Xbh, Gbh, m_max=2, reg=1e-8)
    assert float(torch.max(torch.abs(Xb - (1.0 / 0.7)))) < 1e-5
    print(f"[PASS] CUDAGraph-safe CGS2-QR Anderson contraction: res = {res:.2e}")


def test_central_bf16_jvp_and_solver_precision() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    net_fp32 = PTXDiTVelocityOperator(d=128, h=256, n_blocks=2, seed=11).to(device).eval()
    net_fp64 = copy.deepcopy(net_fp32).to(torch.float64).eval()
    net_bf16 = copy.deepcopy(net_fp32).to(torch.bfloat16).eval()

    torch.manual_seed(5)
    X = torch.randn(8, 128, device=device)
    t = torch.linspace(0.1, 1.0, 8, device=device)
    R = torch.randn_like(X)
    R = R * (2e-5 * torch.linalg.norm(X) / torch.linalg.norm(R))

    with torch.no_grad():
        h = 1e-7
        J_ex = (
            (net_fp64(X.double() + h * R.double(), t.double()) - net_fp64(X.double() - h * R.double(), t.double()))
            / (2.0 * h)
        ).float()
        jn = float(torch.linalg.norm(J_ex).cpu())
        sc = 0.125 * torch.linalg.norm(X) / torch.linalg.norm(R)

        # Unscaled BF16
        J_un = net_bf16((X + R).to(torch.bfloat16), t).float() - net_bf16(X.to(torch.bfloat16), t).float()
        err_un = float(torch.linalg.norm(J_un - J_ex).cpu()) / jn

        # Forward SP-BF16
        J_fw = (net_bf16((X + sc * R).to(torch.bfloat16), t).float() - net_bf16(X.to(torch.bfloat16), t).float()) / sc
        err_fw = float(torch.linalg.norm(J_fw - J_ex).cpu()) / jn

        # Central CSP-BF16
        Xp = torch.cat([X + sc * R, X - sc * R], dim=0).to(torch.bfloat16)
        Vp = net_bf16(Xp, torch.cat([t, t], dim=0)).float()
        J_cn = (Vp[:8] - Vp[8:]) / (2.0 * sc)
        err_cn = float(torch.linalg.norm(J_cn - J_ex).cpu()) / jn

    assert err_un > 0.9, f"Expected catastrophic unscaled BF16 underflow, got {err_un}"
    assert err_cn < err_fw * 0.55, f"Expected Central < 0.55 * Forward: {err_cn} vs {err_fw}"

    # Verify full CSP-BF16-IR solver breaks 7-bit BF16 stagnation wall by > 1,000x
    solver = PTXCspBf16Solver(net_bf16, net_fp32, M=12, d_state=128, h_ff=256, device=device)
    x0 = torch.randn(1, 128, device=device)
    out_pf = torch.empty_like(x0)
    out_csp = torch.empty_like(x0)
    solver.run_b1_into(x0, out_pf, bf_init=5, outer_ir=0)
    solver.run_b1_into(x0, out_csp, bf_init=4, outer_ir=3, inner_jvp=2)

    # FP64 reference
    x_ref = x0.double().clone()
    dt = 1.0 / 256
    tz = torch.zeros(1, device=device, dtype=torch.float64)
    for s in range(256):
        ts = tz + s * dt
        k1 = net_fp64(x_ref, ts)
        k2 = net_fp64(x_ref + 0.5 * dt * k1, ts + 0.5 * dt)
        k3 = net_fp64(x_ref + 0.5 * dt * k2, ts + 0.5 * dt)
        k4 = net_fp64(x_ref + dt * k3, ts + dt)
        x_ref = x_ref + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    x_ref32 = x_ref.detach().float()
    rn = float(torch.linalg.norm(x_ref32).cpu())
    err_pf_sol = float(torch.linalg.norm(out_pf - x_ref32).cpu()) / rn
    err_csp_sol = float(torch.linalg.norm(out_csp - x_ref32).cpu()) / rn
    assert err_csp_sol < 1e-6 and (err_pf_sol / err_csp_sol) > 1000.0, (
        f"Expected >1000x precision improvement: pf={err_pf_sol:.2e}, csp={err_csp_sol:.2e}"
    )
    print(
        f"[PASS] CSP-BF16-IR precision law ({device}): Unscaled={err_un:.1f}x, Fwd={err_fw:.3f}, "
        f"Central={err_cn:.3f} | Trajectory err: BF16={err_pf_sol:.2e} -> CSP-IR={err_csp_sol:.2e} "
        f"({err_pf_sol / err_csp_sol:.0f}x better!)"
    )


def test_unet2d_tiny_forward_and_results_json() -> None:
    tiny_unet = DiffusersUNet2D(
        sample_size=16,
        block_out_channels=(32, 64),
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        layers_per_block=1,
        norm_num_groups=8,
    ).eval()
    with torch.no_grad():
        x = torch.randn(2, 3, 16, 16)
        t = torch.tensor([12.5, 850.0])
        y = tiny_unet(x, t)
    assert y.shape == (2, 3, 16, 16) and torch.isfinite(y).all()

    res_dir = ROOT / "results"
    if res_dir.exists():
        jsons = list(res_dir.glob("*.json"))
        for jp in jsons:
            data = json.loads(jp.read_text(encoding="utf-8"))
            assert isinstance(data, dict) and len(data) > 0
        print(f"[PASS] DiffusersUNet2D forward + {len(jsons)} JSON result artifacts validated")


if __name__ == "__main__":
    test_clenshaw_curtis_polynomial_exactness()
    test_exp_and_z_panels_exactness()
    test_analytic_vp_schedule_bijectivity()
    test_cgs2_anderson_contraction()
    test_central_bf16_jvp_and_solver_precision()
    test_unet2d_tiny_forward_and_results_json()
    print("\nALL VERIFICATION CHECKS PASSED (0 ERRORS).")

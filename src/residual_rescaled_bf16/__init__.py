"""ResidualRescaledBF16: Breaking the 7-Bit BF16 Mantissa Ceiling via Dynamic Residual Rescaling."""

from .csp_bf16_solver import (
    PTXCspBf16Solver,
    PTXDiTBlock,
    PTXDiTVelocityOperator,
    Spatial2DGlyphDiT,
    capture_cudagraph,
    gpu_paradigms_body,
    gpu_seq_euler_body,
    gpu_seq_rk4_body,
)
from .logsnr_diffusion import (
    AnalyticVPSchedule,
    build_ddim_schedule,
    build_dpm_solver2_schedule,
    precompute_csp_ir_panels,
    precompute_hybrid_panels,
    precompute_paradigms_windows,
    precompute_paraflow_panels,
    run_hybrid_csp_bf16_ir,
    run_paradigms_bf16,
)
from .ptx_kernels import (
    PTX_SRC,
    PTXKernel,
    compile_ptx,
    get_csp_ptx_kernels,
    launch_ptx_csp_pack,
    launch_ptx_csp_unpack,
    launch_ptx_swiglu,
)
from .unet2d import DiffusersUNet2D, get_timestep_embedding
from .volterra_chebyshev import (
    batched_cgs2_anderson,
    build_cheb_volterra,
    build_exp_chebyshev_panel,
    build_z_chebyshev_panel,
    graph_safe_anderson_cgs2,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "AnalyticVPSchedule",
    "DiffusersUNet2D",
    "PTXCspBf16Solver",
    "PTXDiTBlock",
    "PTXDiTVelocityOperator",
    "PTXKernel",
    "PTX_SRC",
    "Spatial2DGlyphDiT",
    "batched_cgs2_anderson",
    "build_cheb_volterra",
    "build_ddim_schedule",
    "build_dpm_solver2_schedule",
    "build_exp_chebyshev_panel",
    "build_z_chebyshev_panel",
    "capture_cudagraph",
    "compile_ptx",
    "get_csp_ptx_kernels",
    "get_timestep_embedding",
    "gpu_paradigms_body",
    "gpu_seq_euler_body",
    "gpu_seq_rk4_body",
    "graph_safe_anderson_cgs2",
    "launch_ptx_csp_pack",
    "launch_ptx_csp_unpack",
    "launch_ptx_swiglu",
    "precompute_csp_ir_panels",
    "precompute_hybrid_panels",
    "precompute_paradigms_windows",
    "precompute_paraflow_panels",
    "run_hybrid_csp_bf16_ir",
    "run_paradigms_bf16",
]

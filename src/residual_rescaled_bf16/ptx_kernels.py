"""Raw NVIDIA PTX Driver API JIT loader and fused SM_80/SM_120 kernels for CSP-BF16-IR.

Provides zero-overhead Driver API kernel compilation (`cuModuleLoadData` / `cuLaunchKernel`)
on Windows (`nvcuda.dll`) and Linux (`libcuda.so.1`), with automatic IEEE-754 RNE PyTorch
fallbacks when executed on CPU-only continuous integration hosts.
"""

from __future__ import annotations

import ctypes
import sys
from typing import Sequence

import torch
import torch.nn.functional as F

_CUDA_DRIVER = None


def _get_cuda_driver():
    global _CUDA_DRIVER
    if _CUDA_DRIVER is not None:
        return _CUDA_DRIVER
    if not torch.cuda.is_available():
        return None
    lib_names = ["nvcuda.dll"] if sys.platform == "win32" else ["libcuda.so.1", "libcuda.so"]
    for name in lib_names:
        try:
            drv = ctypes.WinDLL(name) if sys.platform == "win32" else ctypes.CDLL(name)
            drv.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
            drv.cuModuleLoadData.restype = ctypes.c_int
            drv.cuModuleGetFunction.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_void_p,
                ctypes.c_char_p,
            ]
            drv.cuModuleGetFunction.restype = ctypes.c_int
            drv.cuLaunchKernel.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint,
                ctypes.c_uint,
                ctypes.c_uint,
                ctypes.c_uint,
                ctypes.c_uint,
                ctypes.c_uint,
                ctypes.c_uint,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
            ]
            drv.cuLaunchKernel.restype = ctypes.c_int
            _CUDA_DRIVER = drv
            return _CUDA_DRIVER
        except OSError:
            continue
    return None


class PTXKernel:
    """Thin zero-overhead wrapper around a loaded CUfunction handle."""

    def __init__(self, mod: ctypes.c_void_p, fn: ctypes.c_void_p, name: str):
        self._mod = mod
        self._fn = fn
        self.name = name

    def launch(
        self,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: Sequence,
        shared_mem_bytes: int = 0,
        stream: int | None = None,
    ) -> None:
        drv = _get_cuda_driver()
        if drv is None:
            raise RuntimeError("CUDA Driver API is unavailable on this host.")
        if stream is None:
            stream = torch.cuda.current_stream().cuda_stream
        c_args = (ctypes.c_void_p * len(args))(
            *[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in args]
        )
        err = drv.cuLaunchKernel(
            self._fn,
            int(grid[0]),
            int(grid[1]),
            int(grid[2]),
            int(block[0]),
            int(block[1]),
            int(block[2]),
            int(shared_mem_bytes),
            ctypes.c_void_p(stream),
            c_args,
            None,
        )
        if err != 0:
            raise RuntimeError(f"cuLaunchKernel({self.name}) failed with CUDA Driver error {err}")


def compile_ptx(ptx_src: str | bytes, entry_names: list[str]) -> dict[str, PTXKernel]:
    """JIT-compile PTX assembly via `cuModuleLoadData` on the active PyTorch CUDA context."""
    drv = _get_cuda_driver()
    if drv is None:
        return {}
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    if isinstance(ptx_src, str):
        clean = ptx_src.replace("—", "-").replace("\r", "").encode("ascii", errors="ignore")
    else:
        clean = ptx_src.replace(b"\r", b"")
    mod = ctypes.c_void_p()
    err = drv.cuModuleLoadData(ctypes.byref(mod), clean)
    if err != 0:
        raise RuntimeError(f"cuModuleLoadData failed with CUDA Driver error {err}")
    out: dict[str, PTXKernel] = {}
    for name in entry_names:
        fn = ctypes.c_void_p()
        err = drv.cuModuleGetFunction(ctypes.byref(fn), mod, name.encode("utf-8"))
        if err != 0:
            raise RuntimeError(f"cuModuleGetFunction({name}) failed with error {err}")
        out[name] = PTXKernel(mod, fn, name)
    return out


PTX_SRC = r"""
.version 8.0
.target sm_80
.address_size 64

// Kernel 1: Fused SwiGLU in BF16 - reads [B, 2*H] bf16, computes silu(u1)*u2 in fp32, writes RNE [B, H] bf16
.visible .entry ptx_swiglu_bf16(
    .param .u64 p_in,
    .param .u64 p_out,
    .param .u32 H,
    .param .u32 total_BH
) {
    .reg .pred %p;
    .reg .u16 %h1, %h2, %hout;
    .reg .u32 %r_tid, %r_ntid, %r_ctaid, %idx, %h, %tot, %b, %col, %in_idx1, %in_idx2;
    .reg .u32 %u1_bits, %u2_bits, %out_bits, %rnd, %lsb;
    .reg .u64 %base_in, %base_out, %addr1, %addr2, %addr_out, %off;
    .reg .f32 %f1, %f2, %neg_f1, %log2e, %e_val, %denom, %silu, %res;

    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_ntid, %ntid.x;
    mov.u32 %r_ctaid, %ctaid.x;
    mad.lo.u32 %idx, %r_ctaid, %r_ntid, %r_tid;

    ld.param.u32 %tot, [total_BH];
    setp.ge.u32 %p, %idx, %tot;
    @%p bra DONE1;

    ld.param.u32 %h, [H];
    ld.param.u64 %base_in, [p_in];
    ld.param.u64 %base_out, [p_out];

    div.u32 %b, %idx, %h;
    rem.u32 %col, %idx, %h;
    mul.lo.u32 %in_idx1, %b, %h;
    shl.b32 %in_idx1, %in_idx1, 1;
    add.u32 %in_idx1, %in_idx1, %col;
    add.u32 %in_idx2, %in_idx1, %h;

    mul.wide.u32 %off, %in_idx1, 2;
    add.u64 %addr1, %base_in, %off;
    mul.wide.u32 %off, %in_idx2, 2;
    add.u64 %addr2, %base_in, %off;

    ld.global.u16 %h1, [%addr1];
    ld.global.u16 %h2, [%addr2];

    cvt.u32.u16 %u1_bits, %h1;
    cvt.u32.u16 %u2_bits, %h2;
    shl.b32 %u1_bits, %u1_bits, 16;
    shl.b32 %u2_bits, %u2_bits, 16;
    mov.b32 %f1, %u1_bits;
    mov.b32 %f2, %u2_bits;

    neg.f32 %neg_f1, %f1;
    mov.f32 %log2e, 0f3FB8AA3B;
    mul.f32 %neg_f1, %neg_f1, %log2e;
    ex2.approx.ftz.f32 %e_val, %neg_f1;
    add.f32 %denom, %e_val, 0f3F800000;
    div.full.f32 %silu, %f1, %denom;
    mul.f32 %res, %silu, %f2;

    // IEEE-754 Round-to-Nearest-Even fp32 -> bf16
    mov.b32 %out_bits, %res;
    shr.u32 %lsb, %out_bits, 16;
    and.b32 %lsb, %lsb, 1;
    add.u32 %rnd, %lsb, 32767;
    add.u32 %out_bits, %out_bits, %rnd;
    shr.u32 %out_bits, %out_bits, 16;
    cvt.u16.u32 %hout, %out_bits;

    mul.wide.u32 %off, %idx, 2;
    add.u64 %addr_out, %base_out, %off;
    st.global.u16 [%addr_out], %hout;

DONE1:
    ret;
}

// Kernel 2: Fused SwiGLU in FP32 - reads [B, 2*H] fp32, writes [B, H] fp32
.visible .entry ptx_swiglu_fp32(
    .param .u64 p_in,
    .param .u64 p_out,
    .param .u32 H,
    .param .u32 total_BH
) {
    .reg .pred %p;
    .reg .u32 %r_tid, %r_ntid, %r_ctaid, %idx, %h, %tot, %b, %col, %in_idx1, %in_idx2;
    .reg .u64 %base_in, %base_out, %addr1, %addr2, %addr_out, %off;
    .reg .f32 %f1, %f2, %neg_f1, %log2e, %e_val, %denom, %silu, %res;

    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_ntid, %ntid.x;
    mov.u32 %r_ctaid, %ctaid.x;
    mad.lo.u32 %idx, %r_ctaid, %r_ntid, %r_tid;

    ld.param.u32 %tot, [total_BH];
    setp.ge.u32 %p, %idx, %tot;
    @%p bra DONE2;

    ld.param.u32 %h, [H];
    ld.param.u64 %base_in, [p_in];
    ld.param.u64 %base_out, [p_out];

    div.u32 %b, %idx, %h;
    rem.u32 %col, %idx, %h;
    mul.lo.u32 %in_idx1, %b, %h;
    shl.b32 %in_idx1, %in_idx1, 1;
    add.u32 %in_idx1, %in_idx1, %col;
    add.u32 %in_idx2, %in_idx1, %h;

    mul.wide.u32 %off, %in_idx1, 4;
    add.u64 %addr1, %base_in, %off;
    mul.wide.u32 %off, %in_idx2, 4;
    add.u64 %addr2, %base_in, %off;

    ld.global.f32 %f1, [%addr1];
    ld.global.f32 %f2, [%addr2];

    neg.f32 %neg_f1, %f1;
    mov.f32 %log2e, 0f3FB8AA3B;
    mul.f32 %neg_f1, %neg_f1, %log2e;
    ex2.approx.ftz.f32 %e_val, %neg_f1;
    add.f32 %denom, %e_val, 0f3F800000;
    div.full.f32 %silu, %f1, %denom;
    mul.f32 %res, %silu, %f2;

    mul.wide.u32 %off, %idx, 4;
    add.u64 %addr_out, %base_out, %off;
    st.global.f32 [%addr_out], %res;

DONE2:
    ret;
}

// Kernel 3: Fused Symmetric Central-Difference Pack [X + s*D ; X - s*D] -> BF16 in ONE pass
.visible .entry ptx_csp_pack_bf16(
    .param .u64 p_X,
    .param .u64 p_D,
    .param .u64 p_nx2,
    .param .u64 p_nd2,
    .param .u64 p_out_pair,
    .param .f32 sigma,
    .param .u32 N
) {
    .reg .pred %p;
    .reg .u16 %hp, %hm;
    .reg .u32 %r_tid, %r_ntid, %r_ctaid, %idx, %n, %idx2;
    .reg .u32 %bp, %bm, %lsb, %rnd;
    .reg .u64 %bX, %bD, %bnx, %bnd, %bOut, %off, %aX, %aD, %aOut1, %aOut2;
    .reg .f32 %nx2, %nd2, %sig, %ratio, %s, %x, %d, %xp, %xm;

    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_ntid, %ntid.x;
    mov.u32 %r_ctaid, %ctaid.x;
    mad.lo.u32 %idx, %r_ctaid, %r_ntid, %r_tid;

    ld.param.u32 %n, [N];
    setp.ge.u32 %p, %idx, %n;
    @%p bra DONE3;

    ld.param.u64 %bnx, [p_nx2];
    ld.param.u64 %bnd, [p_nd2];
    ld.global.f32 %nx2, [%bnx];
    ld.global.f32 %nd2, [%bnd];
    ld.param.f32 %sig, [sigma];

    add.f32 %nx2, %nx2, 0f17000000;
    add.f32 %nd2, %nd2, 0f0A000000;
    div.full.f32 %ratio, %nx2, %nd2;
    sqrt.rn.f32 %s, %ratio;
    mul.f32 %s, %s, %sig;

    ld.param.u64 %bX, [p_X];
    ld.param.u64 %bD, [p_D];
    ld.param.u64 %bOut, [p_out_pair];

    mul.wide.u32 %off, %idx, 4;
    add.u64 %aX, %bX, %off;
    add.u64 %aD, %bD, %off;
    ld.global.f32 %x, [%aX];
    ld.global.f32 %d, [%aD];

    fma.rn.f32 %xp, %s, %d, %x;
    neg.f32 %d, %d;
    fma.rn.f32 %xm, %s, %d, %x;

    mov.b32 %bp, %xp;
    shr.u32 %lsb, %bp, 16;
    and.b32 %lsb, %lsb, 1;
    add.u32 %rnd, %lsb, 32767;
    add.u32 %bp, %bp, %rnd;
    shr.u32 %bp, %bp, 16;
    cvt.u16.u32 %hp, %bp;

    mov.b32 %bm, %xm;
    shr.u32 %lsb, %bm, 16;
    and.b32 %lsb, %lsb, 1;
    add.u32 %rnd, %lsb, 32767;
    add.u32 %bm, %bm, %rnd;
    shr.u32 %bm, %bm, 16;
    cvt.u16.u32 %hm, %bm;

    mul.wide.u32 %off, %idx, 2;
    add.u64 %aOut1, %bOut, %off;
    add.u32 %idx2, %idx, %n;
    mul.wide.u32 %off, %idx2, 2;
    add.u64 %aOut2, %bOut, %off;

    st.global.u16 [%aOut1], %hp;
    st.global.u16 [%aOut2], %hm;

DONE3:
    ret;
}

// Kernel 4: Fused Central-Difference Unpack [V+ ; V-] BF16 -> J_c = (V+ - V-) / (2*s) FP32
.visible .entry ptx_csp_unpack_fp32(
    .param .u64 p_vpair,
    .param .u64 p_nx2,
    .param .u64 p_nd2,
    .param .u64 p_jout,
    .param .f32 sigma,
    .param .u32 N
) {
    .reg .pred %p;
    .reg .u16 %hp, %hm;
    .reg .u32 %r_tid, %r_ntid, %r_ctaid, %idx, %n, %idx2, %vp32, %vm32;
    .reg .u64 %bV, %bnx, %bnd, %bJ, %off, %aV1, %aV2, %aJ;
    .reg .f32 %nx2, %nd2, %sig, %ratio, %two_s, %vp, %vm, %diff, %jval;

    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_ntid, %ntid.x;
    mov.u32 %r_ctaid, %ctaid.x;
    mad.lo.u32 %idx, %r_ctaid, %r_ntid, %r_tid;

    ld.param.u32 %n, [N];
    setp.ge.u32 %p, %idx, %n;
    @%p bra DONE4;

    ld.param.u64 %bnx, [p_nx2];
    ld.param.u64 %bnd, [p_nd2];
    ld.global.f32 %nx2, [%bnx];
    ld.global.f32 %nd2, [%bnd];
    ld.param.f32 %sig, [sigma];

    add.f32 %nx2, %nx2, 0f17000000;
    add.f32 %nd2, %nd2, 0f0A000000;
    div.full.f32 %ratio, %nx2, %nd2;
    sqrt.rn.f32 %two_s, %ratio;
    mul.f32 %two_s, %two_s, %sig;
    add.f32 %two_s, %two_s, %two_s;

    ld.param.u64 %bV, [p_vpair];
    ld.param.u64 %bJ, [p_jout];

    mul.wide.u32 %off, %idx, 2;
    add.u64 %aV1, %bV, %off;
    add.u32 %idx2, %idx, %n;
    mul.wide.u32 %off, %idx2, 2;
    add.u64 %aV2, %bV, %off;

    ld.global.u16 %hp, [%aV1];
    ld.global.u16 %hm, [%aV2];
    cvt.u32.u16 %vp32, %hp;
    cvt.u32.u16 %vm32, %hm;
    shl.b32 %vp32, %vp32, 16;
    shl.b32 %vm32, %vm32, 16;
    mov.b32 %vp, %vp32;
    mov.b32 %vm, %vm32;

    sub.f32 %diff, %vp, %vm;
    div.full.f32 %jval, %diff, %two_s;

    mul.wide.u32 %off, %idx, 4;
    add.u64 %aJ, %bJ, %off;
    st.global.f32 [%aJ], %jval;

DONE4:
    ret;
}
"""

_COMPILED_KERNELS: dict[str, PTXKernel] | None = None


def get_csp_ptx_kernels() -> dict[str, PTXKernel]:
    """Lazily compile and cache the four CSP-BF16-IR PTX kernels on first GPU use."""
    global _COMPILED_KERNELS
    if _COMPILED_KERNELS is None:
        _COMPILED_KERNELS = compile_ptx(
            PTX_SRC,
            ["ptx_swiglu_bf16", "ptx_swiglu_fp32", "ptx_csp_pack_bf16", "ptx_csp_unpack_fp32"],
        )
    return _COMPILED_KERNELS


def launch_ptx_swiglu(x_2h: torch.Tensor, out_h: torch.Tensor) -> torch.Tensor:
    """Execute fused SwiGLU via raw PTX on CUDA or exact PyTorch fallback on CPU."""
    kernels = get_csp_ptx_kernels() if x_2h.is_cuda else {}
    if x_2h.is_cuda and kernels:
        B = x_2h.shape[0]
        H = out_h.shape[1]
        tot = B * H
        kname = "ptx_swiglu_bf16" if x_2h.dtype == torch.bfloat16 else "ptx_swiglu_fp32"
        kernels[kname].launch(
            ((tot + 255) // 256, 1, 1),
            (256, 1, 1),
            [
                ctypes.c_uint64(x_2h.data_ptr()),
                ctypes.c_uint64(out_h.data_ptr()),
                ctypes.c_uint32(H),
                ctypes.c_uint32(tot),
            ],
        )
        return out_h
    u1, u2 = x_2h.float().chunk(2, dim=-1)
    out_h.copy_((F.silu(u1) * u2).to(out_h.dtype))
    return out_h


def launch_ptx_csp_pack(
    X: torch.Tensor,
    delta: torch.Tensor,
    nx2: torch.Tensor,
    nd2: torch.Tensor,
    pair_buf: torch.Tensor,
    sigma: float,
    n_tot: int,
) -> None:
    """Pack `[X + s*delta ; X - s*delta]` into BF16 `pair_buf` in a single memory pass."""
    kernels = get_csp_ptx_kernels() if X.is_cuda else {}
    if X.is_cuda and kernels:
        kernels["ptx_csp_pack_bf16"].launch(
            ((n_tot + 255) // 256, 1, 1),
            (256, 1, 1),
            [
                ctypes.c_uint64(X.data_ptr()),
                ctypes.c_uint64(delta.data_ptr()),
                ctypes.c_uint64(nx2.data_ptr()),
                ctypes.c_uint64(nd2.data_ptr()),
                ctypes.c_uint64(pair_buf.data_ptr()),
                ctypes.c_float(sigma),
                ctypes.c_uint32(n_tot),
            ],
        )
        return
    s = sigma * torch.sqrt((nx2 + 4.1359e-25) / (nd2 + 6.1630e-33))
    M = X.shape[0]
    pair_buf[:M].copy_((X + s * delta).to(torch.bfloat16))
    pair_buf[M:].copy_((X - s * delta).to(torch.bfloat16))


def launch_ptx_csp_unpack(
    Vp_pair: torch.Tensor,
    nx2: torch.Tensor,
    nd2: torch.Tensor,
    j_buf: torch.Tensor,
    sigma: float,
    n_tot: int,
) -> None:
    """Unpack symmetric BF16 evaluations `[V+ ; V-]` into FP32 `J_c = (V+ - V-) / (2*s)`."""
    kernels = get_csp_ptx_kernels() if Vp_pair.is_cuda else {}
    if Vp_pair.is_cuda and kernels:
        kernels["ptx_csp_unpack_fp32"].launch(
            ((n_tot + 255) // 256, 1, 1),
            (256, 1, 1),
            [
                ctypes.c_uint64(Vp_pair.data_ptr()),
                ctypes.c_uint64(nx2.data_ptr()),
                ctypes.c_uint64(nd2.data_ptr()),
                ctypes.c_uint64(j_buf.data_ptr()),
                ctypes.c_float(sigma),
                ctypes.c_uint32(n_tot),
            ],
        )
        return
    two_s = 2.0 * sigma * torch.sqrt((nx2 + 4.1359e-25) / (nd2 + 6.1630e-33))
    M = j_buf.shape[0]
    j_buf.copy_((Vp_pair[:M].float() - Vp_pair[M:].float()) / two_s)

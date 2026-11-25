#!/usr/bin/env bash
set -euo pipefail

ROOT="/run/user/1000/rr_bf16_gx10"
PY="/home/andrew/bibleproject-imagegen/.venv/bin/python"
export HF_HOME="${ROOT}/.hf_cache"
export TORCH_HOME="${ROOT}/.torch_cache"
mkdir -p "${ROOT}/results" "${HF_HOME}" "${TORCH_HOME}"

cd "${ROOT}"
echo "=== [GX10 GB10] 0. Verification Suite ==="
"${PY}" -u scripts/verify_all.py

echo "=== [GX10 GB10] 1. Stage A: Perturbation V-Curve ==="
"${PY}" -u scripts/run_stage_a_scaling_laws.py

echo "=== [GX10 GB10] 2. Stage B: 46.91M SwiGLU DiT PTX+CUDAGraph Scaling ==="
"${PY}" -u scripts/run_stage_b_dit_cudagraph.py

echo "=== [GX10 GB10] 3. Stage C: Trained Flow-Matching Generative Models ==="
"${PY}" -u scripts/run_stage_c_flow_matching.py

echo "=== [GX10 GB10] 4. Stage D: google/ddpm-cifar10-32 Full Pareto & Nyquist ==="
"${PY}" -u scripts/run_stage_d_ddpm_cifar10.py
rm -f "${ROOT}/.cache/ddpm_cifar10_32/"*.safetensors 2>/dev/null || true
rm -rf "${HF_HOME}"/* 2>/dev/null || true

echo "=== [GX10 GB10] 5. Stage E: google/ddpm-celebahq-256 (256x256, 113.67M) Scale-Up ==="
"${PY}" -u scripts/run_stage_e_celebahq256.py
rm -f "${ROOT}/.cache/ddpm_celebahq_256/"*.bin 2>/dev/null || true
rm -rf "${HF_HOME}"/* 2>/dev/null || true

echo "=== [GX10 GB10] ALL STAGES COMPLETE ==="
ls -lh "${ROOT}/results"

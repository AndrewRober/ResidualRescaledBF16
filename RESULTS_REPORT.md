# Clean-Room Silicon Benchmark Report — `ResidualRescaledBF16`

**Primary Hardware:** `NVIDIA GB10` (`aarch64` Linux, CUDA 13.0, Driver 580.142, `2.11.0+cu130`)  
**Cross-Platform Portability Hardware:** `NVIDIA GeForce RTX 5060 Laptop GPU (Blackwell SM_120)` (`x86_64` Windows, CUDA 12.8)

---

## Stage A: $\mathcal{O}(\varepsilon_{\mathrm{bf16}}^{2/3})$ Perturbation V-Curve (`results/stage_a_vcurve.json`)
- **Unscaled `BF16` Finite Difference Relative Error:** `162.16x` (`100%` mantissa underflow)
- **Optimal Forward `SP-BF16` Error ($\mathcal{O}(\varepsilon^{1/2})$):** `14.12%`
- **Optimal Antipodal Central `CSP-BF16` Error ($\mathcal{O}(\varepsilon^{2/3})$ at $\sigma^\star = 0.125$):** **`4.46%`** (**`3.68x`** single-sweep gain at zero extra FLOPs)

---

## Stage B: `46.91M` SwiGLU DiT `PTX` + `CUDAGraph` Scaling (`results/stage_b_dit_cudagraph.json`)

| $S$ (NFEs) | Eager RK4 (ms) | CG RK4 (ms) | CG ParaDiGMS (ms) | ParaFlow-BF16 (ms) | **CSP-BF16-IR (ms)** | ParaDiGMS $L_2$ | RK4 $L_2$ | **CSP-BF16-IR $L_2$** | **Speedup vs. CG-RK4** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `64` | `66.76` | `59.24` | `4.96` | `2.76` | **`10.59`** | `2.17e-03` | `1.59e-07` | **`9.37e-08`** | **`5.6x`** |
| `128` | `133.74` | `119.62` | `9.55` | `2.76` | **`10.59`** | `2.12e-03` | `1.60e-07` | **`9.37e-08`** | **`11.3x`** |
| `256` | `268.30` | `233.53` | `17.64` | `2.76` | **`10.59`** | `2.11e-03` | `2.08e-07` | **`9.37e-08`** | **`22.1x`** |
| `512` | `537.21` | `468.35` | `34.65` | `2.76` | **`10.59`** | `2.10e-03` | `2.85e-07` | **`9.37e-08`** | **`44.2x`** |

---

## Stage D: Official Pretrained `google/ddpm-cifar10-32` (`35.75M` UNet, 8 Seeds, `results/stage_d_ddpm_cifar10_pareto.json`)
- **Strict `IEEE-FP32` `B=1` Step Latency:** `12.08 ms` (`DDIM-250 IEEE-FP32` = `3020.3 ms`)
- **`CSP-BF16-IR` vs. Exact `IEEE-FP32` Collocation Trajectory PSNR:** **`86.08 dB`**

| Method | $B=1$ Median (ms) | 8-Seed PSNR (dB) | Mean Rel $L_2$ | Max Pixel Error |
| :--- | ---: | ---: | ---: | ---: |
| `DDIM-20 (FP32)` | `80.04` | `28.35 ± 3.32` | `1.765e-01` | `2.942e-01` |
| `DDIM-20 (BF16)` | `49.81` | `27.82 ± 3.19` | `1.864e-01` | `3.085e-01` |
| `DDIM-50 (FP32)` | `199.48` | `35.12 ± 3.75` | `8.546e-02` | `1.467e-01` |
| `DDIM-50 (BF16)` | `123.75` | `33.90 ± 3.24` | `9.521e-02` | `1.631e-01` |
| `DDIM-100 (FP32)` | `398.16` | `40.50 ± 3.99` | `4.758e-02` | `8.455e-02` |
| `DDIM-100 (BF16)` | `245.81` | `38.00 ± 2.83` | `5.868e-02` | `1.061e-01` |
| `DDIM-250 (FP32)` | `992.98` | `47.86 ± 4.18` | `2.078e-02` | `4.411e-02` |
| `DDIM-250 (BF16)` | `613.17` | `41.55 ± 1.59` | `3.710e-02` | `7.573e-02` |
| `DPM-Solver-2-20 (FP32)` | `79.27` | `31.85 ± 2.56` | `1.200e-01` | `2.195e-01` |
| `DPM-Solver-2-20 (BF16)` | `48.82` | `32.22 ± 2.11` | `1.140e-01` | `1.980e-01` |
| `DPM-Solver-2-50 (FP32)` | `199.29` | `49.70 ± 1.93` | `1.479e-02` | `2.688e-02` |
| `DPM-Solver-2-50 (BF16)` | `122.07` | `45.55 ± 2.11` | `2.383e-02` | `5.614e-02` |
| `DPM-Solver-2-100 (FP32)` | `398.22` | `61.43 ± 2.57` | `3.778e-03` | `7.914e-03` |
| `DPM-Solver-2-100 (BF16)` | `244.84` | `46.02 ± 1.79` | `2.198e-02` | `4.770e-02` |
| `ParaDiGMS-BF16 (S=96,W=8)` | `259.90` | `37.87 ± 2.96` | `6.006e-02` | `1.077e-01` |
| `Log-SNR ParaFlow-BF16 (P=4,M=8)` | `146.36` | `46.97 ± 2.26` | `1.974e-02` | `4.857e-02` |
| `Log-SNR CSP-BF16-IR (P=7,M=8,1-IR,TF32)` | `264.72` | `52.16 ± 1.84` | `1.098e-02` | `2.825e-02` |
| `Log-SNR CSP-BF16-IR (P=9,M=8,1-IR,TF32)` | `272.51` | `54.82 ± 3.89` | `9.391e-03` | `2.960e-02` |
| `Log-SNR CSP-BF16-IR (P=7,M=8,1-IR,IEEE)` | `346.42` | `52.96 ± 2.07` | `9.897e-03` | `2.565e-02` |
| `Log-SNR CSP-BF16-IR (P=9,M=8,1-IR,IEEE)` | `374.55` | `55.42 ± 3.39` | `8.454e-03` | `2.778e-02` |
| `Log-SNR CSP-BF16-IR (P=9,M=8,2-IR,IEEE)` | `729.60` | `69.69 ± 2.20` | `1.487e-03` | `2.384e-03` |

---

## Stage E: Zero-Shot `64x` Spatial Scale-Up on Official `google/ddpm-celebahq-256` (`113.67M` UNet, `256x256` Faces, 4 Seeds, `results/stage_e_celebahq256_scaleup.json`)
- **State Dimension:** `D = 3 x 256 x 256 = 196,608` (`64x` larger than CIFAR-10), $\sigma_c = 0.125$ strictly unchanged.
- **Strict `IEEE-FP32` `B=1` Step Latency:** `146.01 ms` (`DDIM-250 IEEE-FP32` = `36502 ms`)

| Method (`256x256`) | $B=1$ Median (ms) | 4-Seed PSNR (dB) | Mean Rel $L_2$ | Max Pixel Error |
| :--- | ---: | ---: | ---: | ---: |
| `DDIM-100 (FP32)` | `7544.6` | `38.91 ± 2.04` | `4.136e-02` | `3.503e-01` |
| `DDIM-100 (BF16)` | `5243.5` | `26.11 ± 1.42` | `1.776e-01` | `7.439e-01` |
| `DPM-Solver-2-50 (FP32)` | `3779.4` | `42.34 ± 3.98` | `3.014e-02` | `2.879e-01` |
| `DPM-Solver-2-50 (BF16)` | `2624.9` | `25.57 ± 1.52` | `1.896e-01` | `8.390e-01` |
| `DPM-Solver-2-100 (FP32)` | `7527.1` | `46.20 ± 2.00` | `1.798e-02` | `1.596e-01` |
| `DPM-Solver-2-100 (BF16)` | `5259.3` | `25.65 ± 1.19` | `1.869e-01` | `8.134e-01` |
| `ParaDiGMS-BF16 (S=96,W=8)` | `15342.9` | `26.46 ± 1.36` | `1.705e-01` | `6.960e-01` |
| `Log-SNR ParaFlow-BF16 (P=4,M=8)` | `8569.2` | `25.60 ± 1.90` | `1.903e-01` | `8.691e-01` |
| `Log-SNR CSP-BF16-IR (P=9,M=8,1-IR,TF32)` | `16739.9` | `44.20 ± 1.19` | `2.210e-02` | `1.961e-01` |
| `Log-SNR CSP-BF16-IR (P=9,M=8,2-IR,IEEE)` | `40323.3` | `50.10 ± 2.46` | `1.161e-02` | `1.103e-01` |

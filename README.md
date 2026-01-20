# Fast CUDA Kernels

High-performance matrix multiplication kernels written to push CUDA limits and learn deeply how modern GPU optimizations work.

**Focus**: GEMM implementations that exploit **Tensor Cores** and optimized classical GEMM patterns, aiming for maximum TFLOPS on NVIDIA GPUs.

Performance has been tested on **NVIDIA Ampere** RTX 3070 GPUs, with a maximum of 0.5x cuBLAS performance while maintaining perfect accuracy.

## Key Implementations

### `tc_matmul.cu` — Tensor Core GEMM

High-throughput matrix multiplication using **WMMA** / Tensor Core instructions.

**Highlights**
- Direct use of Tensor Cores (fp16/bf16/tf32)
- Tiled/shared memory loading optimized with 128-bit vectorization
- Double buffering to hide memory latency
- Asynchronous [CUDA Pipelining](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html) for overlapping compute & data transfers between global/shared memory.
- Very high TFLOPS utilization, with 20 TFLOPS on Ampere

Currently, the fastest single-file GEMM kernel in this repo.

### `blas_gemm.cu` — Classical CUDA Core GEMM

Heavily tuned **non-Tensor-Core** matrix multiply.

Inspired by [this](https://siboehm.com/articles/22/CUDA-MMM) famous blog.

**Highlights**
- Multi-level tiling (block → warp → thread)
- Vectorized global → shared memory loads (float4 / float2)
- Bank-conflict-free shared memory layouts
- Aggressive unrolling + register blocking
- FP32 & FP16 variants
- Designed to reach close to peak non-TC performance on Ampere

## Getting Started

To compile `tc_matmul.cu`, run (for NVIDIA Ampere GPUs):

`nvcc -O3 -lcublas -use_fast_math -arch=sm_86 tc_matmul.cu -o tc_matmul`

To compile `blas_gemm.cu`, run the regular `nvcc` command.

## Other Files

`sparse_gemm.cu` is for my personal experimentation with cuSPARSELt, a library for sparse matrix multiplications.

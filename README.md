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

### `flash_attention.cu` — Flash Attention

Flash Attention is a high-performance implementation in CUDA of the Attention Algorithm, optimizing memory usage without sacrificing accuracy. 

Verified accuracy against a naive attention kernel, demonstrating over **7x speedup**.

**Highlights**
- Implements the Flash Attention algorithm using tiled computation and online softmax
- Uses shared memory to load Q, K, V tiles (size: BLOCK_SIZE × HEAD_DIM)
- Processes attention in blocks to reduce memory I/O
- Maintains running statistics (max m_i and sum l_i) for numerically stable softmax
- Rescales outputs incrementally as new blocks are processed
- Still trying to incorporate Tensor Core usage to squeeze out maximum TFLOPS from matrix multiplications

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

To compile `flash_attention.cu`, run (for NVIDIA Ampere GPUs):

`nvcc -o flash_attn flash_attention.cu -std=c++11 -arch=sm_86`

To compile `blas_gemm.cu`, run the regular `nvcc` command.

## Other Files

`sparse_gemm.cu` is for my personal experimentation with cuSPARSELt, a library for sparse matrix multiplications.

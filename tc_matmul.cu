#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cuda/pipeline>

using namespace nvcuda;

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d\n", __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 64;
constexpr int WARPS_M = 4;
constexpr int WARPS_N = 2;
constexpr int WARPS_PER_BLOCK = WARPS_M * WARPS_N;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
constexpr int STAGES = 2;

__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* gmem_ptr, bool guard) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    int src_size = guard ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(smem_addr), "l"(gmem_ptr), "r"(src_size));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template<int N>
__device__ __forceinline__ void cp_async_wait() {
    if constexpr (N == 0) asm volatile("cp.async.wait_group 0;\n" ::);
    else if constexpr (N == 1) asm volatile("cp.async.wait_group 1;\n" ::);
    else if constexpr (N == 2) asm volatile("cp.async.wait_group 2;\n" ::);
    else asm volatile("cp.async.wait_all;\n" ::);
}

__global__ void tensorMatMul(
    const half* __restrict__ A, 
    const half* __restrict__ B, 
    float* __restrict__ C, 
    int M, int N, int K) 
{
    const int stride_A = BK + 8; 
    const int stride_B = BN + 8;
    const int A_tile_size = BM * stride_A;
    const int B_tile_size = BK * stride_B;

    extern __shared__ half smem[];
    half* smem_A[STAGES];
    half* smem_B[STAGES];
    for (int i = 0; i < STAGES; ++i) {
        smem_A[i] = smem + i * A_tile_size;
        smem_B[i] = smem + STAGES * A_tile_size + i * B_tile_size;
    }

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_m = warp_id / WARPS_N; 
    const int warp_n = warp_id % WARPS_N;

    const int block_m = blockIdx.y * BM;
    const int block_n = blockIdx.x * BN;

    // Warp tile: 32x64 (2x4 fragments)
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j)
            wmma::fill_fragment(acc[i][j], 0.0f);

    // Prologue
    int kt = 0;
    for (int s = 0; s < STAGES - 1; ++s) {
        if (kt < K) {
            #pragma unroll
            for (int i = tid; i < (BM * BK) / 8; i += THREADS_PER_BLOCK) {
                int r = i / (BK / 8); int c = i % (BK / 8);
                cp_async_16(&smem_A[s][r * stride_A + c * 8], &A[(block_m + r) * K + kt + c * 8], (block_m + r < M && kt + c * 8 < K));
            }
            #pragma unroll
            for (int i = tid; i < (BK * BN) / 8; i += THREADS_PER_BLOCK) {
                int r = i / (BN / 8); int c = i % (BN / 8);
                cp_async_16(&smem_B[s][r * stride_B + c * 8], &B[(kt + r) * N + block_n + c * 8], (kt + r < K && block_n + c * 8 < N));
            }
        }
        cp_async_commit();
        kt += BK;
    }

    // Main Loop
    int compute_stage = 0;
    for (int k_idx = 0; k_idx < K; k_idx += BK) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();

        // Prefetch next
        int fetch_stage = (compute_stage + STAGES - 1) % STAGES;
        if (kt < K) {
            #pragma unroll
            for (int i = tid; i < (BM * BK) / 8; i += THREADS_PER_BLOCK) {
                int r = i / (BK / 8); int c = i % (BK / 8);
                cp_async_16(&smem_A[fetch_stage][r * stride_A + c * 8], &A[(block_m + r) * K + kt + c * 8], (block_m + r < M && kt + c * 8 < K));
            }
            #pragma unroll
            for (int i = tid; i < (BK * BN) / 8; i += THREADS_PER_BLOCK) {
                int r = i / (BN / 8); int c = i % (BN / 8);
                cp_async_16(&smem_B[fetch_stage][r * stride_B + c * 8], &B[(kt + r) * N + block_n + c * 8], (kt + r < K && block_n + c * 8 < N));
            }
        }
        cp_async_commit();

        // Compute current stage
        int warp_row_off = warp_m * 32;
        int warp_col_off = warp_n * 64;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[4];

        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                wmma::load_matrix_sync(a_frag[i], &smem_A[compute_stage][(warp_row_off + i * 16) * stride_A + kk], stride_A);
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                wmma::load_matrix_sync(b_frag[j], &smem_B[compute_stage][kk * stride_B + warp_col_off + j * 16], stride_B);

            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }

        compute_stage = (compute_stage + 1) % STAGES;
        kt += BK;
    }

    // Store results
    int warp_row_off = warp_m * 32;
    int warp_col_off = warp_n * 64;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int out_r = block_m + warp_row_off + i * 16;
            int out_c = block_n + warp_col_off + j * 16;
            if (out_r < M && out_c < N) {
                wmma::store_matrix_sync(&C[out_r * N + out_c], acc[i][j], N, wmma::mem_row_major);
            }
        }
    }
}

void initMatrix(float* mat, int N) {
    for (int i = 0; i < N * N; ++i)
        mat[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
}

void floatToHalf(const float* f, half* h, int n) {
    for (int i = 0; i < n; ++i) h[i] = __float2half(f[i]);
}

void halfToFloat(const half* h, float* f, int n) {
    for (int i = 0; i < n; ++i) f[i] = __half2float(h[i]);
}

bool checkAccuracy(const float* gpu, const float* ref, int N) {
    float max_err = 0.0f;
    float avg_err = 0.0f;
    int bad_count = 0;
    
    for (int i = 0; i < N * N; ++i) {
        float err = fabsf(gpu[i] - ref[i]);
        
        if (err > max_err) max_err = err;
        avg_err += err;
        
        float threshold = fmaxf(fabsf(ref[i]) * 0.02f, 0.02f);
        if (err > threshold) bad_count++;
    }
    
    avg_err /= (N * N);
    printf("Max error: %.6f  Avg error: %.6f\n", max_err, avg_err);
    printf("Errors above threshold: %d / %d\n", bad_count, N * N);
    
    return bad_count == 0;
}

int main(int argc, char** argv) {
    int N = 4096;
    if (argc > 1) N = atoi(argv[1]);

    size_t float_bytes = N * N * sizeof(float);
    size_t half_bytes  = N * N * sizeof(half);

    float *h_A = (float*)malloc(float_bytes);
    float *h_B = (float*)malloc(float_bytes);
    float *h_C = (float*)malloc(float_bytes);
    float *h_C_cublas = (float*)malloc(float_bytes);
    half *h_A_h = (half*)malloc(half_bytes);
    half *h_B_h = (half*)malloc(half_bytes);
    half *h_C_cublas_h = (half*)malloc(half_bytes);

    srand(42);
    initMatrix(h_A, N);
    initMatrix(h_B, N);
    floatToHalf(h_A, h_A_h, N * N);
    floatToHalf(h_B, h_B_h, N * N);

    half *d_A, *d_B, *d_C_cublas;
    float *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, half_bytes));
    CUDA_CHECK(cudaMalloc(&d_B, half_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, float_bytes));
    CUDA_CHECK(cudaMalloc(&d_C_cublas, half_bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A_h, half_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B_h, half_bytes, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(N, BM));
    int stride_A = BK + 8;
    int stride_B = BN + 8;
    size_t smem_size = STAGES * (BM * stride_A + BK * stride_B) * sizeof(half);

    // Kernel Call
    CUDA_CHECK(cudaFuncSetAttribute(tensorMatMul, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    tensorMatMul<<<grid, block, smem_size>>>(d_A, d_B, d_C, N, N, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Benchmark custom kernel
    int iters = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        tensorMatMul<<<grid, block, smem_size>>>(d_A, d_B, d_C, N, N, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_custom;
    CUDA_CHECK(cudaEventElapsedTime(&ms_custom, start, stop));
    ms_custom /= iters;

    double gflops_custom = 2.0 * N * N * N / (ms_custom * 1e6);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, float_bytes, cudaMemcpyDeviceToHost));

    printf("=== Custom Kernel Performance ===\n");
    printf("Time: %.3f ms\n", ms_custom);
    printf("Throughput: %.2f GFLOPS\n\n", gflops_custom);

    // Benchmark cuBLAS
    printf("=== Computing cuBLAS reference ===\n");
    
    const float alpha_f = 1.0f;
    const float beta_f = 0.0f;
    
    // Warmup - using FP32 compute for accurate reference
    CUBLAS_CHECK(cublasGemmEx(handle,
                          CUBLAS_OP_T,  // A^T → gives us row-major A as intended
                          CUBLAS_OP_T,  // B^T → gives us row-major B as intended
                          N, N, N,
                          &alpha_f,
                          d_A, CUDA_R_16F, N,   // ld = N (row stride)
                          d_B, CUDA_R_16F, N,
                          &beta_f,
                          d_C_cublas, CUDA_R_16F, N,
                          CUBLAS_COMPUTE_32F,
                          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  N, N, N,
                                  &alpha_f,
                                  d_B, CUDA_R_16F, N,
                                  d_A, CUDA_R_16F, N,
                                  &beta_f,
                                  d_C_cublas, CUDA_R_16F, N,
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_cublas;
    CUDA_CHECK(cudaEventElapsedTime(&ms_cublas, start, stop));
    ms_cublas /= iters;

    double gflops_cublas = 2.0 * N * N * N / (ms_cublas * 1e6);

    CUDA_CHECK(cudaMemcpy(h_C_cublas_h, d_C_cublas, half_bytes, cudaMemcpyDeviceToHost));
    halfToFloat(h_C_cublas_h, h_C_cublas, N * N);

    printf("cuBLAS Time: %.3f ms\n", ms_cublas);
    printf("cuBLAS Throughput: %.2f GFLOPS\n", gflops_cublas);
    printf("Custom vs cuBLAS: %.2fx %s\n\n", 
           ms_cublas / ms_custom,
           ms_custom < ms_cublas ? "faster" : "slower");
    
    printf("=== Accuracy Check (vs cuBLAS) ===\n");
    if (checkAccuracy(h_C, h_C_cublas, N)) {
        printf("Result: PASS\n");
    } else {
        printf("Result: FAIL\n");
    }

    free(h_A); free(h_B); free(h_C); free(h_C_cublas); 
    free(h_A_h); free(h_B_h); free(h_C_cublas_h);
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); 
    CUDA_CHECK(cudaFree(d_C)); CUDA_CHECK(cudaFree(d_C_cublas));
    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}
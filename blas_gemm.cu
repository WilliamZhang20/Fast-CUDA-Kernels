#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

#define BM 128
#define BN 128
#define BK 16

#define TM 8
#define TN 8

// Padding to avoid bank conflicts
#define AS_PAD 8
#define BS_PAD 8

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Software pipelined matrix multiplication with maximum overlap
__global__ void tiledMatMul(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    // Double buffers with padding
    __shared__ float As[2][BM][BK + AS_PAD];
    __shared__ float Bs[2][BK][BN + BS_PAD];

    const int threadRow = threadIdx.y;
    const int threadCol = threadIdx.x;
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    const int rowBase = blockRow * BM + threadRow * TM;
    const int colBase = blockCol * BN + threadCol * TN;

    // Accumulator registers
    float threadResults[TM * TN] = {0.0f};
    
    // Fragment registers for computation
    float regM[2][TM];  // Double buffer for A fragments
    float regN[2][TN];  // Double buffer for B fragments

    const int threadIdx_linear = threadRow * blockDim.x + threadCol;
    const int numThreads = blockDim.x * blockDim.y;

    const int totalFloat4sA = (BM * BK) / 4;
    const int float4sPerRowA = BK / 4;
    const int totalFloat4sB = (BK * BN) / 4;
    const int float4sPerRowB = BN / 4;

    // Prefetch first tile
    int writeIdx = 0;
    {
        for (int loadIdx = threadIdx_linear; loadIdx < totalFloat4sA; loadIdx += numThreads) {
            const int row = loadIdx / float4sPerRowA;
            const int col4 = loadIdx % float4sPerRowA;
            const int col = col4 * 4;
            const int globalRow = blockRow * BM + row;
            const int globalCol = col;
            
            if (globalRow < N && globalCol + 3 < N) {
                float4 tmp = reinterpret_cast<const float4*>(&A[globalRow * N + globalCol])[0];
                As[writeIdx][row][col + 0] = tmp.x;
                As[writeIdx][row][col + 1] = tmp.y;
                As[writeIdx][row][col + 2] = tmp.z;
                As[writeIdx][row][col + 3] = tmp.w;
            } else {
                for (int j = 0; j < 4; j++) {
                    As[writeIdx][row][col + j] = (globalRow < N && globalCol + j < N) 
                        ? A[globalRow * N + globalCol + j] : 0.0f;
                }
            }
        }

        for (int loadIdx = threadIdx_linear; loadIdx < totalFloat4sB; loadIdx += numThreads) {
            const int row = loadIdx / float4sPerRowB;
            const int col4 = loadIdx % float4sPerRowB;
            const int col = col4 * 4;
            const int globalRow = row;
            const int globalCol = blockCol * BN + col;
            
            if (globalRow < N && globalCol + 3 < N) {
                float4 tmp = reinterpret_cast<const float4*>(&B[globalRow * N + globalCol])[0];
                Bs[writeIdx][row][col + 0] = tmp.x;
                Bs[writeIdx][row][col + 1] = tmp.y;
                Bs[writeIdx][row][col + 2] = tmp.z;
                Bs[writeIdx][row][col + 3] = tmp.w;
            } else {
                for (int j = 0; j < 4; j++) {
                    Bs[writeIdx][row][col + j] = (globalRow < N && globalCol + j < N) 
                        ? B[globalRow * N + globalCol + j] : 0.0f;
                }
            }
        }
    }
    __syncthreads();

    // Prefetch first fragment into regM[0] and regN[0]
    int readIdx = writeIdx;
    int regIdx = 0;
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        regM[regIdx][i] = As[readIdx][threadRow * TM + i][0];
    #pragma unroll
    for (int j = 0; j < TN; ++j)
        regN[regIdx][j] = Bs[readIdx][0][threadCol * TN + j];

    // Main loop: software pipeline with fine-grained overlap
    int numTiles = (N + BK - 1) / BK;
    for (int tileIdx = 0; tileIdx < numTiles - 1; ++tileIdx) {
        int bk = (tileIdx + 1) * BK;
        writeIdx = 1 - writeIdx;
        
        // Inner K loop with pipelined loads
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            int nextK = k + 1;
            int nextRegIdx = 1 - regIdx;
            
            // Prefetch next fragments while computing current
            if (nextK < BK) {
                #pragma unroll
                for (int i = 0; i < TM; ++i)
                    regM[nextRegIdx][i] = As[readIdx][threadRow * TM + i][nextK];
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    regN[nextRegIdx][j] = Bs[readIdx][nextK][threadCol * TN + j];
            }
            
            // Load next tile asynchronously (only on first K iteration)
            if (k == 0) {
                for (int loadIdx = threadIdx_linear; loadIdx < totalFloat4sA; loadIdx += numThreads) {
                    const int row = loadIdx / float4sPerRowA;
                    const int col4 = loadIdx % float4sPerRowA;
                    const int col = col4 * 4;
                    const int globalRow = blockRow * BM + row;
                    const int globalCol = bk + col;
                    
                    if (globalRow < N && globalCol + 3 < N) {
                        float4 tmp = reinterpret_cast<const float4*>(&A[globalRow * N + globalCol])[0];
                        As[writeIdx][row][col + 0] = tmp.x;
                        As[writeIdx][row][col + 1] = tmp.y;
                        As[writeIdx][row][col + 2] = tmp.z;
                        As[writeIdx][row][col + 3] = tmp.w;
                    } else {
                        for (int j = 0; j < 4; j++) {
                            As[writeIdx][row][col + j] = (globalRow < N && globalCol + j < N) 
                                ? A[globalRow * N + globalCol + j] : 0.0f;
                        }
                    }
                }

                for (int loadIdx = threadIdx_linear; loadIdx < totalFloat4sB; loadIdx += numThreads) {
                    const int row = loadIdx / float4sPerRowB;
                    const int col4 = loadIdx % float4sPerRowB;
                    const int col = col4 * 4;
                    const int globalRow = bk + row;
                    const int globalCol = blockCol * BN + col;
                    
                    if (globalRow < N && globalCol + 3 < N) {
                        float4 tmp = reinterpret_cast<const float4*>(&B[globalRow * N + globalCol])[0];
                        Bs[writeIdx][row][col + 0] = tmp.x;
                        Bs[writeIdx][row][col + 1] = tmp.y;
                        Bs[writeIdx][row][col + 2] = tmp.z;
                        Bs[writeIdx][row][col + 3] = tmp.w;
                    } else {
                        for (int j = 0; j < 4; j++) {
                            Bs[writeIdx][row][col + j] = (globalRow < N && globalCol + j < N) 
                                ? B[globalRow * N + globalCol + j] : 0.0f;
                        }
                    }
                }
            }
            
            // Compute using current fragments
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    threadResults[i * TN + j] += regM[regIdx][i] * regN[regIdx][j];
                }
            }
            
            regIdx = nextRegIdx;
        }
        
        __syncthreads();
        readIdx = writeIdx;
        
        // Prefetch first fragment of next tile
        #pragma unroll
        for (int i = 0; i < TM; ++i)
            regM[regIdx][i] = As[readIdx][threadRow * TM + i][0];
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            regN[regIdx][j] = Bs[readIdx][0][threadCol * TN + j];
    }

    // Final tile computation (no prefetch needed)
    #pragma unroll
    for (int k = 0; k < BK; ++k) {
        int nextK = k + 1;
        int nextRegIdx = 1 - regIdx;
        
        if (nextK < BK) {
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                regM[nextRegIdx][i] = As[readIdx][threadRow * TM + i][nextK];
            #pragma unroll
            for (int j = 0; j < TN; ++j)
                regN[nextRegIdx][j] = Bs[readIdx][nextK][threadCol * TN + j];
        }
        
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                threadResults[i * TN + j] += regM[regIdx][i] * regN[regIdx][j];
            }
        }
        
        regIdx = nextRegIdx;
    }

    // Vectorized write back
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int row = rowBase + i;
        if (row < N) {
            int col = colBase;
            if (col + TN <= N && (col % 4) == 0) {
                float4 result1;
                result1.x = threadResults[i * TN + 0];
                result1.y = threadResults[i * TN + 1];
                result1.z = threadResults[i * TN + 2];
                result1.w = threadResults[i * TN + 3];
                reinterpret_cast<float4*>(&C[row * N + col])[0] = result1;
                
                float4 result2;
                result2.x = threadResults[i * TN + 4];
                result2.y = threadResults[i * TN + 5];
                result2.z = threadResults[i * TN + 6];
                result2.w = threadResults[i * TN + 7];
                reinterpret_cast<float4*>(&C[row * N + col + 4])[0] = result2;
            } else {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    int col = colBase + j;
                    if (col < N) {
                        C[row * N + col] = threadResults[i * TN + j];
                    }
                }
            }
        }
    }
}

// CPU reference implementation
void cpuMatMul(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void initMatrix(float* mat, int N) {
    for (int i = 0; i < N * N; i++) {
        mat[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
}

bool checkAccuracy(const float* C_gpu, const float* C_cpu, int N, float threshold = 1e-3) {
    float maxError = 0.0f;
    int errorCount = 0;
    
    for (int i = 0; i < N * N; i++) {
        float error = fabs(C_gpu[i] - C_cpu[i]);
        if (error > maxError) maxError = error;
        if (error > threshold) errorCount++;
    }
    
    printf("Max error: %.6f\n", maxError);
    printf("Errors above threshold: %d / %d\n", errorCount, N * N);
    
    return errorCount == 0;
}

int main(int argc, char** argv) {
    int N = 1024;
    if (argc > 1) N = atoi(argv[1]);
    
    printf("Matrix size: %d x %d\n", N, N);
    printf("Block tile: BM=%d, BN=%d, BK=%d\n", BM, BN, BK);
    printf("Thread tile: TM=%d, TN=%d\n", TM, TN);
    printf("Optimization: SOFTWARE PIPELINING + Double Buffering + Swizzling\n");
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    
    size_t bytes = N * N * sizeof(float);
    
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    float *h_C_ref = (float*)malloc(bytes);
    
    srand(42);
    initMatrix(h_A, N);
    initMatrix(h_B, N);
    
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    
    dim3 blockDimConfig(BN / TN, BM / TM);
    dim3 gridDimConfig(CEIL_DIV(N, BN), CEIL_DIV(N, BM));
    
    printf("Grid: (%d, %d), Block: (%d, %d)\n", 
           gridDimConfig.x, gridDimConfig.y, blockDimConfig.x, blockDimConfig.y);
    
    tiledMatMul<<<gridDimConfig, blockDimConfig>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    int numIters = 10;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < numIters; i++) {
        tiledMatMul<<<gridDimConfig, blockDimConfig>>>(d_A, d_B, d_C, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float gpuTime;
    CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));
    gpuTime /= numIters;
    
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    
    float gflops = (2.0f * N * N * N) / (gpuTime * 1e6);
    float peakFP32 = 2.0f * prop.clockRate * 1e-6f * prop.multiProcessorCount * 128 / 1000.0f;
    
    printf("\n=== GPU Performance ===\n");
    printf("Time: %.3f ms\n", gpuTime);
    printf("Throughput: %.2f GFLOPS (%.1f%% of peak)\n", 
           gflops, 100.0f * gflops / (peakFP32 * 1000.0f));
    
    if (N <= 1024) {
        printf("\n=== Computing CPU reference ===\n");
        clock_t cpuStart = clock();
        cpuMatMul(h_A, h_B, h_C_ref, N);
        clock_t cpuEnd = clock();
        
        double cpuTime = ((double)(cpuEnd - cpuStart)) / CLOCKS_PER_SEC * 1000.0;
        printf("CPU Time: %.3f ms\n", cpuTime);
        printf("Speedup: %.2fx\n", cpuTime / gpuTime);
        
        printf("\n=== Accuracy Check ===\n");
        bool correct = checkAccuracy(h_C, h_C_ref, N);
        printf("Result: %s\n", correct ? "PASS" : "FAIL");
    }
    
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return 0;
}
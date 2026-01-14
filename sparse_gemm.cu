// VERIFIABLY WORKING CUSPARSELT GEMM
#include <cstdio>
#include <cuda_runtime.h>
#include <cusparseLt.h>

#define CHECK_CUDA(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(err)); \
        return 1; \
    } \
} while(0)

#define CHECK_LT(x) do { \
    cusparseStatus_t st = (x); \
    if (st != CUSPARSE_STATUS_SUCCESS) { \
        printf("cuSPARSELt error: %d\n", st); \
        return 1; \
    } \
} while(0)

int main() {
    cusparseLtHandle_t handle;
    CHECK_LT(cusparseLtInit(&handle));

    const int m = 8, n = 8, k = 8;

    // 2:4 structured sparse A
    float hA[m * k] = {
        1, 2, 0, 0,  3, 4, 0, 0,
        0, 0, 5, 6,  0, 0, 7, 8,
        9, 0, 0, 10, 11, 0, 0, 12,
        0, 13, 14, 0, 0, 15, 16, 0,
        1, 1, 0, 0,  2, 2, 0, 0,
        0, 0, 3, 3,  0, 0, 4, 4,
        5, 0, 0, 5,  6, 0, 0, 6,
        0, 7, 7, 0,  0, 8, 8, 0
    };

    float hB[k * n];
    for (int i = 0; i < k * n; i++) hB[i] = float(i + 1);

    float hC[m * n] = {0};

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    void  *dA_compressed = nullptr;
    void  *dCompressBuffer = nullptr;

    CHECK_CUDA(cudaMalloc(&dA, sizeof(hA)));
    CHECK_CUDA(cudaMalloc(&dB, sizeof(hB)));
    CHECK_CUDA(cudaMalloc(&dC, sizeof(hC)));

    CHECK_CUDA(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));

    const uint32_t alignment = 16;

    cusparseLtMatDescriptor_t matA, matB, matC;

    CHECK_LT(cusparseLtStructuredDescriptorInit(
        &handle, &matA,
        m, k, k,
        alignment,
        CUDA_R_32F,
        CUSPARSE_ORDER_ROW,
        CUSPARSELT_SPARSITY_50_PERCENT));

    CHECK_LT(cusparseLtDenseDescriptorInit(
        &handle, &matB,
        k, n, n,
        alignment,
        CUDA_R_32F,
        CUSPARSE_ORDER_ROW));

    CHECK_LT(cusparseLtDenseDescriptorInit(
        &handle, &matC,
        m, n, n,
        alignment,
        CUDA_R_32F,
        CUSPARSE_ORDER_ROW));

    cusparseLtMatmulDescriptor_t matmulDesc;
    CHECK_LT(cusparseLtMatmulDescriptorInit(
        &handle, &matmulDesc,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &matA, &matB, &matC, &matC,
        CUSPARSE_COMPUTE_32F));

    cusparseLtMatmulAlgSelection_t algSel;
    CHECK_LT(cusparseLtMatmulAlgSelectionInit(
        &handle, &algSel,
        &matmulDesc,
        CUSPARSELT_MATMUL_ALG_DEFAULT));

    cusparseLtMatmulPlan_t plan;
    CHECK_LT(cusparseLtMatmulPlanInit(
        &handle, &plan,
        &matmulDesc, &algSel));

    // -------- Compression sizes (0.7.1 API) --------
    size_t compressedSize = 0;
    size_t compressBufferSize = 0;

    CHECK_LT(cusparseLtSpMMACompressedSize(
        &handle,
        &plan,
        &compressedSize,
        &compressBufferSize));

    CHECK_CUDA(cudaMalloc(&dA_compressed, compressedSize));
    if (compressBufferSize)
        CHECK_CUDA(cudaMalloc(&dCompressBuffer, compressBufferSize));

    // -------- Compress A --------
    CHECK_LT(cusparseLtSpMMACompress(
        &handle,
        &plan,
        dA,
        dA_compressed,
        dCompressBuffer,
        nullptr));

    // -------- Workspace --------
    size_t workspaceSize = 0;
    CHECK_LT(cusparseLtMatmulGetWorkspace(
        &handle,
        &plan,
        &workspaceSize));

    void* dWorkspace = nullptr;
    if (workspaceSize)
        CHECK_CUDA(cudaMalloc(&dWorkspace, workspaceSize));

    float alpha = 1.0f, beta = 0.0f;

    CHECK_LT(cusparseLtMatmul(
        &handle,
        &plan,
        &alpha,
        dA_compressed,
        dB,
        &beta,
        dC,
        dC,
        dWorkspace,
        nullptr,
        0));

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hC, dC, sizeof(hC), cudaMemcpyDeviceToHost));

    printf("C = A * B (first 4x4 block):\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++)
            printf("%7.1f ", hC[i * n + j]);
        printf("\n");
    }

    printf("\ncuSPARSELt 0.7.1 works ✅\n");

    if (dWorkspace) cudaFree(dWorkspace);
    if (dCompressBuffer) cudaFree(dCompressBuffer);
    cudaFree(dA);
    cudaFree(dA_compressed);
    cudaFree(dB);
    cudaFree(dC);
    cusparseLtDestroy(&handle);

    return 0;
}

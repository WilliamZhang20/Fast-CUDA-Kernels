#include <iostream>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

using namespace cutlass;

int main() {
  // Problem size (tuned for P4: small tiles for quick test; scale up to 4096x4096 for perf)
  int M = 512;
  int N = 256;
  int K = 128;
  float alpha = 1.25f;
  float beta = 0.0f;

  // Define GEMM (float, RowMajor A, ColumnMajor B, RowMajor C; Simt for Pascal)
  using Gemm = gemm::device::Gemm<
      float, layout::RowMajor,    // A: float, row-major
      float, layout::ColumnMajor, // B: float, col-major
      float, layout::RowMajor>;   // C: float, row-major

  Gemm gemm_op;

  // Allocate host tensors
  HostTensor<float, layout::RowMajor> A({M, K});
  HostTensor<float, layout::ColumnMajor> B({K, N});
  HostTensor<float, layout::RowMajor> C({M, N});
  HostTensor<float, layout::RowMajor> D({M, N});
  HostTensor<float, layout::RowMajor> Reference({M, N});

  // Fill with uniform random data [-2, 2]
  reference::host::TensorFillRandomUniform(A.host_view(), 42, -2.f, 2.f, 0);
  reference::host::TensorFillRandomUniform(B.host_view(), 43, -2.f, 2.f, 0);
  reference::host::TensorFillRandomUniform(C.host_view(), 44, -2.f, 2.f, 0);

  // Copy to device
  A.sync_device();
  B.sync_device();
  C.sync_device();

  // Launch GEMM: D = alpha * A * B + beta * C
  cutlass::Status status = gemm_op(
      {M, N, K},                             // Shape
      {A.device_data(), K},                  // A ptr + lda
      {B.device_data(), N},                  // B ptr + ldb
      {C.device_data(), N},                  // C ptr + ldc
      {D.device_data(), N},                  // D ptr + ldd
      alpha, beta);                          // Scalars

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "GEMM failed: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    return -1;
  }

  // Copy D back to host
  D.sync_host();

  // Compute CPU reference
  reference::host::Gemm<float, layout::RowMajor, float, layout::ColumnMajor,
                        float, layout::RowMajor, float, float>(
      Reference.host_view(), A.host_view(), B.host_view(), C.host_view(), alpha, beta);

  // Verify: count mismatches (tolerance for FP errors)
  bool passed = reference::host::TensorEquals(D.host_view(), Reference.host_view());
  std::cout << (passed ? "Passed" : "Failed") << " CUTLASS verification!" << std::endl;

  if (!passed) {
    std::cout << "Error - CUTLASS results mismatch CPU reference." << std::endl;
    // Optional: Print first 8x8 for debug
    std::cout << "D (CUTLASS):\n" << D.host_view().slice(layout::PitchLinearCoord(0, 0), layout::PitchLinearCoord(8, 8)) << std::endl;
    std::cout << "Reference:\n" << Reference.host_view().slice(layout::PitchLinearCoord(0, 0), layout::PitchLinearCoord(8, 8)) << std::endl;
    return -1;
  }

  std::cout << "Success! CUTLASS 2.x GEMM on Tesla P4 (SM61) for M=" << M << ", N=" << N << ", K=" << K << std::endl;
  return 0;
}

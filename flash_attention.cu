#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cmath>
#include <random>

// Optimized Flash Attention kernel
// Implements the Flash Attention algorithm with correct online softmax
template<int BLOCK_SIZE, int HEAD_DIM>
__global__ void flash_attention_kernel(
    const float* Q,      // [batch, seq_len, num_heads, head_dim]
    const float* K,      // [batch, seq_len, num_heads, head_dim]
    const float* V,      // [batch, seq_len, num_heads, head_dim]
    float* O,            // [batch, seq_len, num_heads, head_dim]
    int batch_size,
    int seq_len,
    int num_heads,
    float scale
) {
    // Shared memory for tiles
    __shared__ float Q_tile[BLOCK_SIZE][HEAD_DIM];
    __shared__ float K_tile[BLOCK_SIZE][HEAD_DIM];
    __shared__ float V_tile[BLOCK_SIZE][HEAD_DIM];
    
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int q_block_idx = blockIdx.x;
    
    int tid = threadIdx.x;
    int q_start = q_block_idx * BLOCK_SIZE;
    int q_local = tid;
    int q_idx = q_start + q_local;
    
    // Early exit for out of bounds
    if (q_idx >= seq_len) return;
    
    // Initialize output and statistics
    float O_local[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) {
        O_local[d] = 0.0f;
    }
    
    float m_i = -INFINITY;  // Row max
    float l_i = 0.0f;       // Row sum of exp
    
    // Load Q tile into shared memory
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) {
        int offset = batch_idx * seq_len * num_heads * HEAD_DIM +
                    q_idx * num_heads * HEAD_DIM +
                    head_idx * HEAD_DIM + d;
        Q_tile[q_local][d] = Q[offset];
    }
    
    // Process each K/V block
    int num_kv_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * BLOCK_SIZE;
        
        // Load K and V tiles cooperatively
        // Using tid as kv_local
        int kv_local = tid;
        int kv_idx = kv_start + kv_local;
        
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            if (kv_idx < seq_len) {
                int offset = batch_idx * seq_len * num_heads * HEAD_DIM +
                           kv_idx * num_heads * HEAD_DIM +
                           head_idx * HEAD_DIM + d;
                K_tile[kv_local][d] = K[offset];
                V_tile[kv_local][d] = V[offset];
            } else {
                K_tile[kv_local][d] = 0.0f;
                V_tile[kv_local][d] = 0.0f;
            }
        }
        __syncthreads();
        
        // Compute new max and exp sums for this Q row in this block
        float m_j = -INFINITY;
        float S_local[BLOCK_SIZE];
        
        for (int j = 0; j < BLOCK_SIZE; j++) {
            int current_kv_idx = kv_start + j;
            float score = 0.0f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                score += Q_tile[q_local][d] * K_tile[j][d];
            }
            score *= scale;
            
            if (current_kv_idx < seq_len) {
                S_local[j] = score;
                m_j = fmaxf(m_j, score);
            } else {
                S_local[j] = -INFINITY;
            }
        }
        
        float m_i_new = fmaxf(m_i, m_j);
        
        float exp_diff_old = expf(m_i - m_i_new);
        float l_j = 0.0f;
        for (int j = 0; j < BLOCK_SIZE; j++) {
            S_local[j] = expf(S_local[j] - m_i_new);
            l_j += S_local[j];
        }
        
        float l_i_new = exp_diff_old * l_i + l_j;
        
        // Update O_local
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            // Rescale previous O_local
            O_local[d] *= (exp_diff_old * l_i);
            
            // Add contribution from current block
            float v_sum = 0.0f;
            for (int j = 0; j < BLOCK_SIZE; j++) {
                v_sum += S_local[j] * V_tile[j][d];
            }
            O_local[d] = (O_local[d] + v_sum) / l_i_new;
        }
        
        // Update statistics
        m_i = m_i_new;
        l_i = l_i_new;
        
        __syncthreads();
    }
    
    // Write output
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) {
        int offset = batch_idx * seq_len * num_heads * HEAD_DIM +
                    q_idx * num_heads * HEAD_DIM +
                    head_idx * HEAD_DIM + d;
        O[offset] = O_local[d];
    }
}

// Naive attention for verification
__global__ void naive_attention_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    float scale
) {
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (q_idx >= seq_len) return;
    
    // Use local memory for scores to avoid race conditions
    // SeqLen is 512, which resides in local memory/registers
    float scores[512]; 
    
    // Compute scores
    for (int kv_idx = 0; kv_idx < seq_len; kv_idx++) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            int q_offset = batch_idx * seq_len * num_heads * head_dim +
                          q_idx * num_heads * head_dim +
                          head_idx * head_dim + d;
            int k_offset = batch_idx * seq_len * num_heads * head_dim +
                          kv_idx * num_heads * head_dim +
                          head_idx * head_dim + d;
            score += Q[q_offset] * K[k_offset];
        }
        scores[kv_idx] = score * scale;
    }
    
    // Softmax
    float max_score = -INFINITY;
    for (int i = 0; i < seq_len; i++) {
        max_score = fmaxf(max_score, scores[i]);
    }
    
    float sum_exp = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        scores[i] = expf(scores[i] - max_score);
        sum_exp += scores[i];
    }
    
    for (int i = 0; i < seq_len; i++) {
        scores[i] /= sum_exp;
    }
    
    // Compute output
    for (int d = 0; d < head_dim; d++) {
        float out = 0.0f;
        for (int kv_idx = 0; kv_idx < seq_len; kv_idx++) {
            int v_offset = batch_idx * seq_len * num_heads * head_dim +
                          kv_idx * num_heads * head_dim +
                          head_idx * head_dim + d;
            out += scores[kv_idx] * V[v_offset];
        }
        
        int o_offset = batch_idx * seq_len * num_heads * head_dim +
                      q_idx * num_heads * head_dim +
                      head_idx * head_dim + d;
        O[o_offset] = out;
    }
}

// Verification and benchmarking
int main() {
    const int BATCH = 2;
    const int SEQ_LEN = 512;
    const int NUM_HEADS = 8;
    const int HEAD_DIM = 64;
    const int BLOCK_SIZE = 32;
    
    float scale = 1.0f / sqrtf(HEAD_DIM);
    
    size_t size = BATCH * SEQ_LEN * NUM_HEADS * HEAD_DIM * sizeof(float);
    
    // Allocate host memory
    float *h_Q = new float[BATCH * SEQ_LEN * NUM_HEADS * HEAD_DIM];
    float *h_K = new float[BATCH * SEQ_LEN * NUM_HEADS * HEAD_DIM];
    float *h_V = new float[BATCH * SEQ_LEN * NUM_HEADS * HEAD_DIM];
    float *h_O_flash = new float[BATCH * SEQ_LEN * NUM_HEADS * HEAD_DIM];
    float *h_O_naive = new float[BATCH * SEQ_LEN * NUM_HEADS * HEAD_DIM];
    
    // Initialize with random data
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (int i = 0; i < BATCH * SEQ_LEN * NUM_HEADS * HEAD_DIM; i++) {
        h_Q[i] = dist(gen);
        h_K[i] = dist(gen);
        h_V[i] = dist(gen);
    }
    
    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_O_flash, *d_O_naive;
    cudaMalloc(&d_Q, size);
    cudaMalloc(&d_K, size);
    cudaMalloc(&d_V, size);
    cudaMalloc(&d_O_flash, size);
    cudaMalloc(&d_O_naive, size);
    
    cudaMemcpy(d_Q, h_Q, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, size, cudaMemcpyHostToDevice);
    
    // Launch kernels
    dim3 flash_grid((SEQ_LEN + BLOCK_SIZE - 1) / BLOCK_SIZE, NUM_HEADS, BATCH);
    dim3 flash_block(BLOCK_SIZE);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    flash_attention_kernel<BLOCK_SIZE, HEAD_DIM><<<flash_grid, flash_block>>>(
        d_Q, d_K, d_V, d_O_flash, BATCH, SEQ_LEN, NUM_HEADS, scale
    );
    cudaDeviceSynchronize();
    
    // Flash attention
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        flash_attention_kernel<BLOCK_SIZE, HEAD_DIM><<<flash_grid, flash_block>>>(
            d_Q, d_K, d_V, d_O_flash, BATCH, SEQ_LEN, NUM_HEADS, scale
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float flash_time;
    cudaEventElapsedTime(&flash_time, start, stop);
    flash_time /= 10.0f;
    
    // Naive attention
    dim3 naive_grid((SEQ_LEN + 255) / 256, NUM_HEADS, BATCH);
    dim3 naive_block(256);
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        naive_attention_kernel<<<naive_grid, naive_block>>>(
            d_Q, d_K, d_V, d_O_naive, BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM, scale
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float naive_time;
    cudaEventElapsedTime(&naive_time, start, stop);
    naive_time /= 10.0f;
    
    // Copy results back
    cudaMemcpy(h_O_flash, d_O_flash, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_O_naive, d_O_naive, size, cudaMemcpyDeviceToHost);
    
    // Verify correctness
    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    int total = BATCH * SEQ_LEN * NUM_HEADS * HEAD_DIM;
    
    for (int i = 0; i < total; i++) {
        float diff = fabsf(h_O_flash[i] - h_O_naive[i]);
        max_diff = fmaxf(max_diff, diff);
        avg_diff += diff;
    }
    avg_diff /= total;
    
    std::cout << "=== Flash Attention Verification ===" << std::endl;
    std::cout << "Configuration: " << std::endl;
    std::cout << "  Batch: " << BATCH << ", SeqLen: " << SEQ_LEN 
              << ", Heads: " << NUM_HEADS << ", HeadDim: " << HEAD_DIM << std::endl;
    std::cout << "\nTiming (averaged over 10 runs):" << std::endl;
    std::cout << "  Flash Attention: " << flash_time << " ms" << std::endl;
    std::cout << "  Naive Attention: " << naive_time << " ms" << std::endl;
    std::cout << "  Speedup: " << naive_time / flash_time << "x" << std::endl;
    std::cout << "\nAccuracy:" << std::endl;
    std::cout << "  Max difference: " << max_diff << std::endl;
    std::cout << "  Avg difference: " << avg_diff << std::endl;
    std::cout << "  Status: " << (max_diff < 1e-3 ? "PASSED ✓" : "FAILED ✗") << std::endl;
    
    // Cleanup
    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_O_flash;
    delete[] h_O_naive;
    
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O_flash);
    cudaFree(d_O_naive);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#include "dtype_bfloat16.cuh"
#include "reduce_kernels.cuh"

inline int ceilDiv(int N, int D) {
    return (N + D - 1) / D;
}

template <typename Precision, int BLOCK_SIZE, int NUM_WARPS, int THREAD_GROUP_SIZE, int HEAD_SIZE, int STEP_SIZE, int QKV_HEAD_RATIO>
__global__ void custom_flash_attention_splitkv_kernel(int n_seq, int n_step, int q_num_heads, int kv_num_heads, int q_stride, int kv_batch_stride, int kv_head_stride, int kv_tok_stride, float scale, Precision* q, Precision* k, Precision* v, Precision* tmp_out, float* score_maxes, float* softmax_sums){
    const int tid = threadIdx.x;
    const int batch = blockIdx.x; 
    const int head = blockIdx.y;
    const int step = blockIdx.z;

    // store queries, keys, and values using default dtype (blfloat16 for llama3) to reduce smem usage
    __shared__ Precision sq[HEAD_SIZE * QKV_HEAD_RATIO];
    __shared__ Precision sk[HEAD_SIZE * STEP_SIZE];
    __shared__ Precision sv[HEAD_SIZE * STEP_SIZE];
    // and other flash attention intermidiate results using fp32 for better accuracy
    __shared__ float reduce_smem[NUM_WARPS];
    __shared__ float softmaxes[STEP_SIZE * QKV_HEAD_RATIO];
    __shared__ float scores[STEP_SIZE * QKV_HEAD_RATIO];

    int skv_size = HEAD_SIZE * STEP_SIZE;
    int n_tok = STEP_SIZE;

    // for each thread block, we execute in parallel flash attention 2 algorithm for STEP_SIZE tokens
    if (step == n_step - 1){
        n_tok = n_seq - STEP_SIZE * step;
        skv_size = HEAD_SIZE * n_tok;
    }

    // divide threads into groups of size MEM_THREAD_GROUP_SIZE, where each thread group loads HEAD_SIZE values from gmem to smem for k and v
    int MEM_THREAD_GROUP_SIZE = HEAD_SIZE / VEC_SIZE;
    int mem_group = tid / MEM_THREAD_GROUP_SIZE;
    int mem_lane = tid % MEM_THREAD_GROUP_SIZE;
    for (int idx = 0; idx < n_tok; idx += (BLOCK_SIZE * VEC_SIZE) / HEAD_SIZE){
        int sidx = mem_lane * VEC_SIZE + (mem_group + idx) * HEAD_SIZE;
        int gidx = batch * kv_batch_stride + head * kv_head_stride + (step * STEP_SIZE + idx + mem_group) * kv_tok_stride + mem_lane * VEC_SIZE;
        if (sidx < skv_size){
            reinterpret_cast<Precision_Vec *>(&sk[sidx])[0]  = reinterpret_cast<Precision_Vec *>(&k[gidx])[0];
            reinterpret_cast<Precision_Vec *>(&sv[sidx])[0]  = reinterpret_cast<Precision_Vec *>(&v[gidx])[0];        
        }
    }

    // load queries into smem
    // for each key and value pair load QKV_HEAD_RATIO queries (grouped attention) to reuse kv cache, and avoid reloading it multiple times 
    #pragma unroll
    for (int query_idx = 0; query_idx < QKV_HEAD_RATIO; query_idx++){
        sq[tid + query_idx * HEAD_SIZE] = q[tid + batch * q_stride + (head * QKV_HEAD_RATIO + query_idx) * HEAD_SIZE];
    }

    __syncthreads();


    // divide threads into groups of size THREAD_GROUP_SIZE, and compute query * key dot product (scores) from the flash attention 2 algorithm
    // Q @ K = S
    const int THREAD_VEC_SIZE = HEAD_SIZE / THREAD_GROUP_SIZE;
    int group = tid / THREAD_GROUP_SIZE;
    int lane = tid % THREAD_GROUP_SIZE;

    float rq[THREAD_VEC_SIZE * QKV_HEAD_RATIO];
    #pragma unroll
    for (int idx = 0; idx < THREAD_VEC_SIZE; idx++){
        #pragma unroll
        for (int query_idx = 0; query_idx < QKV_HEAD_RATIO; query_idx++){
            rq[idx + query_idx * THREAD_VEC_SIZE] =  p2float(sq[lane + idx * THREAD_GROUP_SIZE + query_idx * HEAD_SIZE]);
        }
    }

    float rk[THREAD_VEC_SIZE];
    for (int idx = 0; idx < n_tok; idx += (BLOCK_SIZE * THREAD_VEC_SIZE) / HEAD_SIZE){
        #pragma unroll
        for (int jdx = 0; jdx < THREAD_VEC_SIZE; jdx++){
            if ((group + idx) < n_tok){
                rk[jdx] =  p2float(sk[lane + jdx * THREAD_GROUP_SIZE + (group + idx) * HEAD_SIZE]);
            }
            else
            {
                rk[jdx] = 0.0;
            }
        }

        float qk_dot[QKV_HEAD_RATIO];
        #pragma unroll
        for (int query_idx = 0; query_idx < QKV_HEAD_RATIO; query_idx++){
            qk_dot[query_idx] = 0.0;
            #pragma unroll
            for (int jdx = 0; jdx < THREAD_VEC_SIZE; jdx++){
                qk_dot[query_idx] += rq[jdx + query_idx * THREAD_VEC_SIZE] * rk[jdx];
            }
        }
        #pragma unroll
        for (int query_idx = 0; query_idx < QKV_HEAD_RATIO; query_idx++){
            #pragma unroll
            for (int mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
                qk_dot[query_idx] += __shfl_xor_sync(0xFFFFFFFF, qk_dot[query_idx], mask);
            }
            if (lane == 0){
                scores[group + idx + query_idx * STEP_SIZE] = scale * qk_dot[query_idx];
            }
        }

    }
    __syncthreads();

    // continue the flash attention 2 algorithm:
    // 1) get the max of scores:
    // m = max(S)
    // 2) compute the exp of the diff between the max and the scores:
    // P = exp(S - m)
    // 3) compute the "online softmax":
    // l = sum(P)
    float curr_score_max[QKV_HEAD_RATIO];
    float softmax_sum[QKV_HEAD_RATIO];
    float score;
    #pragma unroll
    for (int query_idx = 0; query_idx < QKV_HEAD_RATIO; query_idx++){
        if (tid < n_tok){
            score = scores[tid + query_idx * STEP_SIZE];
        }
        else{
            score = -FLT_MAX;
        }

        curr_score_max[query_idx] = block_max<NUM_WARPS>(reduce_smem, score);

        float softmax;
        if (tid < n_tok){
            softmax = __expf(score - curr_score_max[query_idx]);
            softmaxes[tid + query_idx * STEP_SIZE] = softmax;
        }
        else{
            softmax = 0.0;
        }
        softmax_sum[query_idx] = block_sum<NUM_WARPS>(reduce_smem, softmax);
    }

    // compute the softmax * value dot product from the flash attention 2 algorithm:
    // O = P @ V
    float sv_dot[QKV_HEAD_RATIO];
    #pragma unroll
    for (int query_idx = 0; query_idx < QKV_HEAD_RATIO; query_idx++){
            sv_dot[query_idx] = 0.0;
        }
    for (int jdx = 0; jdx < n_tok; jdx++){
        float sv_val = p2float(sv[tid + jdx * HEAD_SIZE]);
        #pragma unroll
        for (int query_idx = 0; query_idx < QKV_HEAD_RATIO; query_idx++){
            sv_dot[query_idx] += softmaxes[jdx + STEP_SIZE * query_idx] * sv_val;
        }
    }
    __syncthreads();

    #pragma unroll
    // multiply the output by the "online softmax"
    for (int query_idx = 0; query_idx < QKV_HEAD_RATIO; query_idx++){
        tmp_out[tid + batch * q_num_heads * n_step * HEAD_SIZE + (head * QKV_HEAD_RATIO + query_idx) * n_step * HEAD_SIZE + step * HEAD_SIZE] = float2p<Precision>(__fdividef(1.0f, softmax_sum[query_idx] + 1e-6f) * sv_dot[query_idx]);

    }

    // store the maxes and softmaxes for each step (thread block), to be used by the combine kernel
    if (tid == 0){
        #pragma unroll
        for (int query_idx = 0; query_idx < QKV_HEAD_RATIO; query_idx++){
            score_maxes[batch * q_num_heads * n_step + (head * QKV_HEAD_RATIO + query_idx) * n_step + step] = curr_score_max[query_idx];
            softmax_sums[batch * q_num_heads * n_step + (head * QKV_HEAD_RATIO + query_idx) * n_step + step] = softmax_sum[query_idx];
        }
    }


}

template <typename Precision, int BLOCK_SIZE, int NUM_WARPS, int THREAD_GROUP_SIZE, int HEAD_SIZE, int STEP_SIZE>
__global__ void custom_flash_attention_combine_kernel(int n_seq, int n_step, int q_stride, int kv_batch_stride, int kv_head_stride, int kv_tok_stride, Precision* out, Precision* tmp_out, float* score_maxes, float* softmax_sums){
    const int tid = threadIdx.x;
    const int batch = blockIdx.x; 
    const int head = blockIdx.y;
    const int step = blockIdx.z;
    const int kv_num_heads = gridDim.y;

    __shared__ float reduce_smem[NUM_WARPS];
    extern __shared__ float shared_softmax_sums[];

    // if total number of tokens < step size, just return the output from the splitkv kernel 
    if (n_step == 1){
        if (tid < HEAD_SIZE){
            out[tid + batch * q_stride + head * HEAD_SIZE] = tmp_out[tid + batch * kv_num_heads * HEAD_SIZE + head * HEAD_SIZE];
            return;
        }
    }

    // get the global scores max across all steps
    float score_max = -FLT_MAX;
    if (tid < n_step){
        score_max = score_maxes[batch * kv_num_heads * n_step + head * n_step + step];
    }
    float global_score_max = block_max<NUM_WARPS>(reduce_smem, score_max);
    __syncthreads();

    // compute the global softmax across all steps
    float softmax_sum = 0.0;
    if (tid < n_step){
        softmax_sum = softmax_sums[batch * kv_num_heads * n_step + head * n_step + step];
        softmax_sum *=  expf(score_max - global_score_max);
        shared_softmax_sums[tid] = softmax_sum;
    }
    float global_softmax_sum = block_sum<NUM_WARPS>(reduce_smem, softmax_sum);
    __syncthreads();
    
    // complete the flash attention 2 algorithm and return final result
    float ro = 0.0;
    for (int idx = 0; idx < n_step; idx++){
        float float_tmp_out = p2float(tmp_out[tid + batch * kv_num_heads * n_step * HEAD_SIZE + head * n_step * HEAD_SIZE + idx * HEAD_SIZE]);
        ro += float_tmp_out * shared_softmax_sums[idx] * __fdividef(1.0f, global_softmax_sum + 1e-6f);
    }
    out[tid + batch * q_stride + head * HEAD_SIZE] = float2p<Precision>(ro);
}
void cudaCheck(cudaError_t error, const char* file, int line){
    if (error != cudaSuccess){
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

torch::Tensor flash_attention_decode(torch::Tensor query, torch::Tensor keys, torch::Tensor values) {

    // kernel parameters
    const int HEAD_SIZE = query.size(3);
    const int BLOCK_SIZE = HEAD_SIZE;
    // tunable parameter, where 8 val provided the best performance
    const int THREAD_GROUP_SIZE = 8;
    const int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    const int BATCH_SIZE = keys.size(0);
    // parameter for kv cache to fit into 48kb static smem for each thread block, and avoid using dynamic smem
    const int STEP_SIZE = 64;
    const int kv_num_heads = keys.size(1);
    const int q_num_heads = query.size(1);
    const int n_seq = keys.size(2);
    const int q_stride = query.stride(0);
    const int kv_batch_stride = keys.stride(0);
    const int kv_head_stride = keys.stride(1);
    const int kv_tok_stride = keys.stride(2);
    const int n_step = ceilDiv(n_seq, STEP_SIZE);
    const int QKV_HEAD_RATIO = q_num_heads / kv_num_heads;
    float scale = 1.0 / sqrt(HEAD_SIZE);

    // splitkv kernel grid
    dim3 split_grid;
    split_grid.x = BATCH_SIZE;
    split_grid.y = kv_num_heads;
    split_grid.z = n_step;

    // combine kernel grid
    dim3 combine_grid;
    combine_grid.x = BATCH_SIZE;
    combine_grid.y = q_num_heads;
    combine_grid.z = 1;

    // tensors for storing intermediates and results
    auto result = torch::empty_like(query);
    auto tmp_result = torch::empty({BATCH_SIZE, q_num_heads, n_step, HEAD_SIZE}, torch::dtype(torch::CppTypeToScalarType<P2Torch>()).device(torch::kCUDA));
    auto score_maxes = torch::empty({BATCH_SIZE, q_num_heads, n_step}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto softmax_sums = torch::empty({BATCH_SIZE, q_num_heads, n_step}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    // macros for launching kernels
    #define FLASH_ATTENTION_SPLITKV(HEAD_SIZE, QKV_HEAD_RATIO) custom_flash_attention_splitkv_kernel<Precision, HEAD_SIZE, (HEAD_SIZE / WARP_SIZE), THREAD_GROUP_SIZE, HEAD_SIZE, STEP_SIZE, QKV_HEAD_RATIO><<<split_grid, BLOCK_SIZE>>>(n_seq, n_step, q_num_heads, kv_num_heads, q_stride, kv_batch_stride, kv_head_stride, kv_tok_stride, scale, (Precision*)(query.data_ptr<P2Torch>()), (Precision*)(keys.data_ptr<P2Torch>()), (Precision*)(values.data_ptr<P2Torch>()), (Precision*)(tmp_result.data_ptr<P2Torch>()), score_maxes.data_ptr<float>(), softmax_sums.data_ptr<float>());
    #define FLASH_ATTENTION_COMBINE(HEAD_SIZE) custom_flash_attention_combine_kernel<Precision, HEAD_SIZE, (HEAD_SIZE / WARP_SIZE), THREAD_GROUP_SIZE, HEAD_SIZE, STEP_SIZE><<<combine_grid, BLOCK_SIZE, n_step * sizeof(float)>>>(n_seq, n_step, q_stride, kv_batch_stride, kv_head_stride, kv_tok_stride, (Precision*)(result.data_ptr<P2Torch>()), (Precision*)(tmp_result.data_ptr<P2Torch>()), score_maxes.data_ptr<float>(), softmax_sums.data_ptr<float>());

    // launch kernel according to head size, and num query / keys, value head ratio
    switch (HEAD_SIZE){
        case 64:
            switch (QKV_HEAD_RATIO){
                case 1:
                    // general non grouped attention
                    FLASH_ATTENTION_SPLITKV(64, 1);
                    break;
                case 4:
                    // llama 3.2 1b
                    FLASH_ATTENTION_SPLITKV(64, 4);
                    break;
                default:
                    TORCH_CHECK(false, "Unsupported QKV head ratio for splitkv kernel: ", QKV_HEAD_RATIO);
                    break;
            }
            break;
        case 128:
            switch (QKV_HEAD_RATIO){
                case 1:
                    // general non grouped attention
                    FLASH_ATTENTION_SPLITKV(128, 1);
                    break;
                case 3:
                    // llama 3.2 3b
                    FLASH_ATTENTION_SPLITKV(128, 3);
                    break;
                case 4:
                    // llama 3.1 8b
                    FLASH_ATTENTION_SPLITKV(128, 4);
                    break;
                case 8:
                    // llama 3.3 70b
                    FLASH_ATTENTION_SPLITKV(128, 8);
                    break;
                default:
                    TORCH_CHECK(false, "Unsupported QKV head ratio for splitkv kernel: ", QKV_HEAD_RATIO);
                    break;
            }
            break;
        default:
            TORCH_CHECK(false, "Unsupported head size for splitkv kernel: ", HEAD_SIZE);
            break;
    }
    

    switch (HEAD_SIZE){
        case 64:
            // head size 64
            FLASH_ATTENTION_COMBINE(64);
            break;
        case 128:
            // head size 128
            FLASH_ATTENTION_COMBINE(128);
            break;
        default:
            TORCH_CHECK(false, "Unsupported head size for combine kernel: ", HEAD_SIZE);
            break;
    }
    
    return result;
}

template <typename Precision, int BLOCK_SIZE, int NUM_WARPS>
__global__ void custom_rms_norm_kernel(int BATCH_SIZE, int SEQ_N, int EMBED_DIM, float eps, Precision* input, Precision* weight, Precision* output){
    
    // output = (1 / rms(input)) * input * weight
    // rms(input) = sqrt(eps + (1 / EMBED_DIM) + sum(input(i)^2))

    const int tid = threadIdx.x;
    const int batch = blockIdx.x;
    const int tok = blockIdx.y;
    __shared__ float reduce_smem[NUM_WARPS];

    float val = 0.0;

    // compute sum in float32 for better accuracy
    for (int idx = tid; idx < EMBED_DIM; idx += BLOCK_SIZE){
        float inp = p2float(input[batch * SEQ_N * EMBED_DIM + tok * EMBED_DIM + idx]);
        inp *= inp;
        val += inp;
    }
    
    float sum_sqr = block_sum<NUM_WARPS>(reduce_smem, val);

    Precision inv_rms =  float2p<Precision>(__frsqrt_rn((sum_sqr / float(EMBED_DIM)) + eps));

    for (int idx = tid; idx < EMBED_DIM; idx += BLOCK_SIZE){
        output[batch * SEQ_N * EMBED_DIM + tok * EMBED_DIM + idx] = inv_rms * weight[idx] * input[batch * SEQ_N * EMBED_DIM + tok * EMBED_DIM + idx];
    }
}

torch::Tensor rms_norm(torch::Tensor input, torch::Tensor weight, float eps) {
    
    const int BATCH_SIZE = input.size(0);
    const int SEQ_N = input.size(1);
    const int EMBED_DIM =  input.size(2);

    const int BLOCK_SIZE = 1024;
    const int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

    dim3 grid;
    grid.x = BATCH_SIZE;
    grid.y = SEQ_N;
    grid.z = 1;

    auto output = torch::empty_like(input);
    
    // each thread block applies RMS Norm to token embdeddings across batch size and sequence length
    custom_rms_norm_kernel<Precision, BLOCK_SIZE, NUM_WARPS><<<grid, BLOCK_SIZE>>>(BATCH_SIZE, SEQ_N, EMBED_DIM, eps, (Precision*)(input.data_ptr<P2Torch>()), (Precision*)(weight.data_ptr<P2Torch>()), (Precision*)(output.data_ptr<P2Torch>()));

    return output;
}

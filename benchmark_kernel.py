import torch
from torch.nn import functional as F
import argparse
from custom_kernels import flash_attention_decode_extension
from torch.nn.attention import SDPBackend, sdpa_kernel

'''
profile memory bandwidth of flash attention decode kernels: custom vs pytorch
two methods are available:
1) using the benchmark setup (default)
2) using nvidia nsight compute (better method, and recommended if your system is confidured to use ncu): skip bechmark setup with --use_ncu flag, and let ncu mesure mem bw
'''


torch.set_default_dtype(torch.bfloat16)

# clear GPU L2 cache after each iteration by filling it with a zeros tensor of size 128MB (for example H100 has 80MB of L2 cache) 
x = torch.empty(int(128 * (1024 ** 2)), dtype=torch.int8, device='cuda')
def flush_cache():
    x.zero_()

# select custom or pytorch kernel
def run_kernel(query, key, value, custom=True):
    if custom:
        _ = flash_attention_decode_extension.flash_attention_decode(query, key, value)
    else:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            _ = F.scaled_dot_product_attention(query, key, value, enable_gqa=True)

# warmup kernel before mesuring duration
def benchmark_kernel(query, key, value, custom):
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(RUN_STEPS)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(RUN_STEPS)]

    for _ in range(WARMUP_STEPS):
        run_kernel(query, key, value, custom)

    for step in range(RUN_STEPS):
        if flush_gpu_cache:
            flush_cache()
        start_events[step].record()
        run_kernel(query, key, value, custom)
        end_events[step].record()
    torch.cuda.synchronize()

    avgerage_elapsed_times = [s.elapsed_time(e) / 1000 for s, e in zip(start_events, end_events)]
    avgerage_elapsed_time =  sum(avgerage_elapsed_times) / RUN_STEPS
    data_size = 2 * 2 * KV_NUM_HEADS * HEAD_SIZE * SEQ_N * BATCH_SIZE / (1024  ** 3)
    bandwidth = data_size / avgerage_elapsed_time
    print(f"KV cache size: {data_size * 1024} MB | average bandwidth: {bandwidth} GB/s | average duration: {avgerage_elapsed_time * 1000} ms")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Benchmark custom attention decode kernel bandwidth")
    parser.add_argument(
        "--head_size",
        type=int,
        required=False,
        default = 128,
        help="model head size",
    )
    parser.add_argument(
        "--seq_n",
        type=int,
        default = 8192,
        required=False,
        help="sequence length",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default = 1,
        required=False,
        help="batch size",
    )
    parser.add_argument(
        "--q_num_heads",
        type=int,
        default = 32,
        required=False,
        help="number of query heads",
    )
    parser.add_argument(
        "--kv_num_heads",
        type=int,
        default = 8,
        required=False,
        help="number of key and value heads",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default = 100,
        required=False,
        help="number of kernel warmup steps",
    )
    parser.add_argument(
        "--run_steps",
        type=int,
        default = 1000,
        required=False,
        help="number of kernel benchmark steps",
    )
    parser.add_argument('--dont_flush_gpu_cache',
    dest='flush_gpu_cache',
    action='store_false',
    help="don't flush the GPU L2 cache"
    )
    parser.set_defaults(flush_gpu_cache=True)
    parser.add_argument('--use_pytorch_kernel',
    dest='custom',
    action='store_false',
    help="Benchmark Pytorch attention kernel instead"
    )
    parser.set_defaults(custom=True)
    parser.add_argument('--use_ncu',
    dest='use_ncu',
    action='store_true',
    help="Use Nvidia Nsight Compute for kernel profiling instead"
    )
    parser.set_defaults(use_ncu=False)
    parser.add_argument(
        "--random_seed",
        type=int,
        required=False,
        default = 42,
        help="random seed",
    )
    args = parser.parse_args()

    custom = args.custom
    HEAD_SIZE = args.head_size
    SEQ_N = args.seq_n
    BATCH_SIZE = args.batch_size
    Q_NUM_HEADS = args.q_num_heads
    KV_NUM_HEADS = args.kv_num_heads
    N_REP = Q_NUM_HEADS // KV_NUM_HEADS

    WARMUP_STEPS = args.warmup_steps
    RUN_STEPS = args.run_steps
    random_seed = args.random_seed
    flush_gpu_cache = args.flush_gpu_cache
    use_ncu = args.use_ncu

    if custom:
        print("Benchmarking custom attention decode kernel:")
        print("==================================")
    else:
        print("Benchmarking Pytorch flash attention kernel:")
        print("==================================")
    print(f"head size: {HEAD_SIZE}")
    print(f"sequence length: {SEQ_N}")
    print(f"batch size: {BATCH_SIZE}")
    print(f"number of query heads: {Q_NUM_HEADS}")
    print(f"number of key and value heads: {KV_NUM_HEADS}")
    print("==================================")

    torch.manual_seed(random_seed)

    # random query, key, and value tensor to simulate attention kernels input for llama3 models
    q = torch.rand(Q_NUM_HEADS * HEAD_SIZE * BATCH_SIZE)
    k = torch.rand(KV_NUM_HEADS * HEAD_SIZE * SEQ_N * BATCH_SIZE)
    v = torch.rand(KV_NUM_HEADS * HEAD_SIZE * SEQ_N * BATCH_SIZE)

    q = q.reshape((BATCH_SIZE, 1, Q_NUM_HEADS, HEAD_SIZE))
    k = k.reshape((BATCH_SIZE, SEQ_N, KV_NUM_HEADS, HEAD_SIZE))
    v = v.reshape((BATCH_SIZE, SEQ_N, KV_NUM_HEADS, HEAD_SIZE))

    q = q.to("cuda")
    k = k.to("cuda")
    v = v.to("cuda")

    q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))

    if use_ncu:
        run_kernel(q, k, v, custom)
    else:
        benchmark_kernel(q, k, v, custom)

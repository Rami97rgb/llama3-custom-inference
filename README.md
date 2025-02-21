# Llama3-Custom-Inference
This repo implements from scratch llama3 offline inference in Pytorch, and more importantly custom CUDA kernels for flash attention decode, rms norm, and rotary embdedding.
This implementation reaches higher throughput in tokens per second than vLLM in single user and batched scenarios.
The custom decode kernel reaches the same performance and memory bandwidth as the flash attention 2 inference kernel.

**v1.0:** custom flash attention decode kernel.

**v1.1:** custom rms norm kernel.

**v1.2:** custom rotary positional encoding kernel.

## Test llama3 inference
First install the requirements, then compile the custom kernels for llama3 and bind them as a Python module usable by Pytorch (llama 3.x models with head sizes 64 and 128 are supported as long as they fit into a single GPU VRAM, and don't require tensor or pipeline parallelism). You can select a llama3 model from the Hugging Face hub to test the inference implementation:
```
pip install -r requirements.txt
python3 setup.py
python3 test_llama3.py --model_name "unsloth/Llama-3.2-1B"
```

## Benchmark inference throughput
You can mesure inference throughput for the llama3 inference implementation across multiple scenarios (short vs long sequence length, prefill vs decode heavy), and compare with vLLM:
```
python3 benchmark_model.py --model_name "unsloth/Llama-3.2-1B" --max_gen_batch_size=1 --max_gen_len=128 --input_len=128
Benchmarking custom llama3 model inference implementation:
==================================
model name: unsloth/Llama-3.2-1B
batch size: 1
max output tokens: 128
input tokens: 128
max model sequence length: 256
==================================
using device: cuda
compiling the model...
model warmup...
running inference...
duration: 1.4909467697143555 Seconds | throughput: 85.85148886604651 Toks/sec
peak memory consumption: 2868 MiB
```
This uses the repos llama3 implementation by default, and you can benchmark vLLM instead with ```--use_vllm_impl```.
We mesure average inference throughput (prefill + decode) in Toks/Sec.

Here are the results I obtained by running llama 3.2 1B locally on my RTX 4060 GPU:

### Batched inference
We set the batch size to the max power of two that does not cause an OOM error (model weights + KV cache + activations).

| Batch Size    | Input / Ouput Toks | Custom Throughtput | vLLM Throughtput   |
| ------------- | ------------------ | ------------------ | ------------------ |
| 256           | 128, 128           | 2810.41            | 2535.55            |
| 64            | 128, 2048          | 2041.67            | 1769.46            |
| 16            | 2048, 128          | 415.58             | 403.66             |
| 16            | 2048, 2048         | 763.64             | 743.11             |

### Single user inference
Batch size is set to 1 for all scenarios, this puts more strain on memory bandwidth as the limiting factor for thoughtput.

| Batch Size    | Input / Ouput Toks | Custom Throughtput | vLLM Throughtput   |
| ------------- | ------------------ | ------------------ | ------------------ |
| 1             | 128, 128           | 91.56              | 88.93              |
| 1             | 128, 2048          | 90.47              | 89.59              |
| 1             | 2048, 128          | 79.68              | 78.10              |
| 1             | 2048, 2048         | 87.51              | 86.83              |

## Profile attention decode kernel
You can mesure memory bandwitdh of the custom decode kernel and compare it to [the flash attention 2 inference kernel](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#22-optimize-for-inference). This is done by simulating the attention input (query, key, value) for llama3 models.

For llama 3.2 1B, batch size 1, and sequence length 256:
```
python3 benchmark_kernel.py --head_size 64 --q_num_heads 32 --kv_num_heads 8 --batch_size 1 --seq_n 256
```
The custom kernel is used by default, but you can run the FA2 implementation instead (splitkv/combine kernels) by adding the ```--use_pytorch_kernel``` flag.
There are two methods for mesuring memory bandwidth:
1) using Nvidia Nsight Compute profiler (the recommended way for getting accurate results, but not supported by most cloud providers): run with ```ncu``` command and add ```--use_ncu``` flag.
2) using the benchmark setup (default, although not 100% accurate). The GPU's L2 cache is cleared after each iteration, but you can skip this with ```--dont_flush_gpu_cache```.

When looking at the actual inference trace (with Nvidia Nsight Systems), execution time is fairly close to the one reported by profiling the kernel with ```ncu```. So I am presenting these results instead.

I simulated running flash attention decode kernels for llama 3.2 1B on my RTX 4060 using batch size 1 (to make it more challenging to fully utilize the avaible memory bandwidth). Here, we are comparing the custom implementation of flash attention splitkv kernel with the one used in Pytorch/FA2:

![alt text](./benchmark_results.png)

We reach similar memory bandwidth utilization across a variety of sequence lengths.

## References
https://github.com/Dao-AILab/flash-attention

https://github.com/karpathy/llm.c

https://github.com/meta-llama/llama-models

https://github.com/NVIDIA/TensorRT-LLM

https://github.com/vllm-project/vllm

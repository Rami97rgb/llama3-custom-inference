# llama3-custom-inference
This repo implements from scratch llama3 offline inference in Pytorch, and more importantly a custom flash attention 2 decode kernel.
The llama3 implementation reaches the same throughput in tokens per second as vLLM in single user and batched scenarios.
The custom decode kernels reach the same speed and memory bandwidth as the flash attention 2 splitkv kernels.

## Test llama3 inference
First install the requirements, then compile the custom flash attention decode kernels for llama3 model (llama3.x models with head sizes 64 and 128 are supported as long as they fit into a single GPU VRAM, and don't require tensor or pipeline parallelism), and load them as a Python module usable by Pytorch. Then you can select llama3 model name from the Hugging Face hub, and test the inference implementation:
```
pip install -r requirements.txt
python3 load_custom_kernels.py
python3 test_llama3.py --model_name "unsloth/Llama-3.2-1B"
```

## Benchmark inference throughput
You can mesure inference throughput for the llama3 inference implementation across multiple scenarios(short vs long sequence length, and prefill vs decode heavy) and compare with vLLM, exemple:
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
This uses the repos llama3 implementation by default, and you benchmark vLLM instead with ```--use_vllm_impl```.
We mesure average inference throughput (prefill + decode),here are the results I obtained with llama3.2 1B running localy on my RTX 4060 GPU:

### Batched inference
We set the batch size to the max power of two that does not cause an OOM error (model weights + KV cache + activations).

| Batch Size    | Input / Ouput Toks | Custom Throughtput | vLLM Throughtput   |
| ------------- | ------------------ | ------------------ | ------------------ |
| 256           | 128, 128           | 2688.44            | 2535.55            |
| 64            | 128, 2048          | 1982.40            | 1769.46            |
| 16            | 2048, 128          | 384.71             | 403.66             |
| 16            | 2048, 2048         | 729.24             | 743.11             |

### Single user inference
batch size is set to 1 for all scenarios, this puts more strain on memory bandwidth as the limitor for thoughtput

| Batch Size    | Input / Ouput Toks | Custom Throughtput | vLLM Throughtput   |
| ------------- | ------------------ | ------------------ | ------------------ |
| 1             | 128, 128           | 85.92              | 88.93              |
| 1             | 128, 2048          | 85.08              | 89.59              |
| 1             | 2048, 128          | 74.90              | 78.10              |
| 1             | 2048, 2048         | 82.47              | 86.83              |

## Profile decode kernel
you can mesure memory bandwitdh of the custom decode kernel and compare to the original flash attention 2 inference kernel. This is a achieved by similating the attention input (query,key, and value) for the llama3 models.
Exemple, llama3 3.2 1B: batch size 1, squence length 256:
```
python3 benchmark_kernel.py --head_size 64 --q_num_heads 32 --kv_num_heads 8 --batch_size 1 --seq_n 256
```
the custom kernel is used by default, you can run the Pytorch implementation instead (splitkv/combine kernels) by adding the ```--use_pytorch_kernel``` flag.
there are two methods for mesuring memory bandwidth:
1) using Nvidia Nsight Compute profiler (more recommended for getting accurate results, but not supported by most cloud providers): run with ```ncu``` command and add ```--use_ncu``` flag.
2) using the benchmark setup (default, although not 100% accurate). L2 cache is cleared after each iteration, but you skip this with ```--dont_flush_gpu_cache```.
when looking at the actual inference trace (with Nvidia Nsight Systems), kernel execution time is fairly close to the one we get when we profiling the kernel with ```ncu```. So I am presenting these results instead.
I similated running flash attention decode kernels for llama3 3.2 1B on my RTX 4060 using batch size 1 (to make it more challenging to fully utilize the avaible memory bandwidth). Here, we are comparing the custom implementation of flash attention splitkv kernel with with the Pytorch implementation:


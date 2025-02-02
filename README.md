# llama3-custom-inference
This repo implements from scratch llama3 offline inference in Pytorch, and more importantly a custom flash attention 2 decode kernel.
This llama3 implementation reaches the same throughput in tokens per second as vLLM in single user and batched queries scenarios.
The custom decode kernels reach the same speed the flash attention 2 splitkv kernels.

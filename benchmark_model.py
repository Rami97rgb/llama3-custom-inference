import torch
import time
import random
from llama3_inference import llama3_inference_engine
from vllm import LLM, SamplingParams, TokensPrompt
import os
import argparse
os.environ['TOKENIZERS_PARALLELISM'] = "false"
import torch.distributed as dist

'''
mesure tokens per second throughput between llama3 implementations: vllm vs custom 
'''

# function to generate random tokens tensor of specific shape, copied from:
# https://github.com/NVIDIA/TensorRT-LLM/blob/main/benchmarks/cpp/utils/utils.py
def gen_random_tokens(ip_len, batch_size, tokenizer, random_seed):

    def get_sample_from_population(population_range, sample_size):
        # random.sample can not sample a value more than once. hence the check
        if sample_size < len(population_range):
            sample = random.sample(population_range, sample_size)
        else:
            sample = random.choices(population_range, k=sample_size)

        return sample

    input_ids = []
    random.seed(random_seed)
    for _ in range(batch_size):
        start_ids = get_sample_from_population(range(0, tokenizer.vocab_size),
                                               ip_len)
        # Make sure it does not contain EOS token
        eos_id = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)
        while set(eos_id).issubset(start_ids):
            tmp_id = (eos_id[0] + 1) % tokenizer.vocab_size
            start_ids = [
                tmp_id if element == eos_id[0] else element
                for element in start_ids
            ]
        input_ids.append(start_ids)

    return input_ids


# benchmark vllm llama3 implementation
def run_vllm_model(model_name, max_model_len, max_gen_batch_size, max_gen_len, random_seed):
    torch.cuda.reset_peak_memory_stats()

    sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=max_gen_len, ignore_eos=True)
    llm = LLM(model=model_name, max_model_len=max_model_len, gpu_memory_utilization=0.95)

    print("running inference...")
    prompt_tokens = [TokensPrompt(prompt_token_ids=prompt) for prompt in gen_random_tokens(input_len, max_gen_batch_size, llm.get_tokenizer(), random_seed)]

    t0 = time.time()
    torch.cuda.nvtx.range_push("generate")
    _ = llm.generate(prompt_tokens, sampling_params)
    torch.cuda.nvtx.range_pop()
    t1 = time.time()

    toks_sec = max_gen_len * max_gen_batch_size / (t1 - t0)
    print(f"duration: {t1 - t0} Seconds | throughput: {toks_sec} Toks/sec")
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
    dist.destroy_process_group()


# benchmark custom llama3 implementation
def run_custom_model(model_name, max_model_len, max_gen_batch_size, max_gen_len, random_seed):
    torch.cuda.reset_peak_memory_stats()

    model = llama3_inference_engine(model_name, max_model_len, max_gen_batch_size)
    print("running inference...")
    prompt_tokens = gen_random_tokens(input_len, max_gen_batch_size, model.tokenizer, random_seed)

    t0 = time.time()
    torch.cuda.nvtx.range_push("generate")
    generation_tokens = model.generate(prompt_tokens, max_gen_len=max_gen_len, temperature=0.6, top_p=0.9, echo=False, ignore_eos=True)
    torch.cuda.nvtx.range_pop()
    t1 = time.time()

    toks_sec = max_gen_len * model.config.max_gen_batch_size / (t1 - t0)
    print(f"duration: {t1 - t0} Seconds | throughput: {toks_sec} Toks/sec")
    _ = [{"generation": model.tokenizer.decode(t)} for t in generation_tokens]
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Benchmark llama3 inference implementations")
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default = "unsloth/Llama-3.2-1B",
        help="model name from the hugging face hub",
    )
    parser.add_argument(
        "--max_gen_batch_size",
        type=int,
        required=False,
        default = 256,
        help="maximum model batch size",
    )
    parser.add_argument(
        "--max_gen_len",
        type=int,
        required=False,
        default = 128,
        help="maximum output tokens",
    )
    parser.add_argument(
        "--input_len",
        type=int,
        required=False,
        default = 128,
        help="input tokens",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        required=False,
        default = 42,
        help="random seed",
    )
    parser.add_argument('--use_vllm_impl',
    dest='custom',
    action='store_false',
    help="Benchmark vllm llama3 implementation instead"
    )
    parser.set_defaults(custom=True)
    args = parser.parse_args()

    custom = args.custom
    model_name = args.model_name
    max_gen_batch_size = args.max_gen_batch_size
    max_gen_len = args.max_gen_len
    input_len = args.input_len
    random_seed = args.random_seed
    max_model_len = input_len + max_gen_len

    # run benchmark and print out results
    if custom:
        print("Benchmarking custom llama3 model inference implementation:")
        print("==================================")
    else:
        print("Benchmarking vllm llama3 model inference implementation:")
        print("==================================")
    print(f"model name: {model_name}")
    print(f"batch size: {max_gen_batch_size}")
    print(f"max output tokens: {max_gen_len}")
    print(f"input tokens: {input_len}")
    print(f"max model sequence length: {max_model_len}")
    print("==================================")

    if custom:
        run_custom_model(model_name, max_model_len, max_gen_batch_size, max_gen_len, random_seed)
    else:
        run_vllm_model(model_name, max_model_len, max_gen_batch_size, max_gen_len, random_seed)

    
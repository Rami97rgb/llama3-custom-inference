'''
adpated from:
https://github.com/karpathy/llm.c/blob/master/train_llama3.py
https://github.com/meta-llama/llama-models/blob/main/models/llama3/reference_impl/model.py
https://github.com/meta-llama/llama-models/blob/main/models/llama3/reference_impl/generation.py
'''

import math
from dataclasses import dataclass
from typing import (
    List,
    Optional,
    Tuple,
)

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from transformers import AutoModelForCausalLM, AutoTokenizer

from custom_kernels import custom_kernels_extension

# precompute frequency to be used by custom rotary embedding kernel

def apply_scaling(freqs: torch.Tensor):
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False
):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return torch.view_as_real(freqs_cis).flatten(1)

# custom kernel implementation for RMSNorm

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        output = custom_kernels_extension.rms_norm(x, self.weight, self.eps)
        return output

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_rep = self.n_head // self.n_kv_head
        self.hd = config.n_embd // config.n_head

        self.c_attn = nn.Linear(config.n_embd, (config.n_head + 2 * config.n_kv_head) * self.hd, bias=False)  # key, query, value projections
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)  # output projection

        # static KV cache - we could alternatively allocate it outside of the model and just pass it in when needed
        self.register_buffer('cache_k', torch.zeros((config.max_gen_batch_size, config.max_model_len, config.n_kv_head, self.hd)))
        self.register_buffer('cache_v', torch.zeros((config.max_gen_batch_size, config.max_model_len, config.n_kv_head, self.hd)))

    def forward(self, x, freqs_cis=None, start_pos=None, mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split([self.n_head * self.hd, self.n_kv_head * self.hd, self.n_kv_head * self.hd], dim=-1)
        q, k, v = map(lambda t: t.view(B, T, -1, self.hd), (q, k, v))  # (B, T, NH, HD)

        torch.cuda.nvtx.range_push("rot_embd")
        q = custom_kernels_extension.rot_embed(q, freqs_cis)
        k = custom_kernels_extension.rot_embed(k, freqs_cis)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("use_kv")
        self.cache_k[:B, start_pos : start_pos + T] = k
        self.cache_v[:B, start_pos : start_pos + T] = v
        k = self.cache_k[:B, : start_pos + T]
        v = self.cache_v[:B, : start_pos + T]
        torch.cuda.nvtx.range_pop()

        q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))  # (B, NH, T, HD)

        # for first token (prefill) call pytorch flash attention implementation, then (decode, when using kv cache) call our custom flash attention implemention
        if T > 1:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        else:
            y = custom_kernels_extension.flash_attention_decode(q, k, v)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        hidden_dim = config.hidden_dim
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_proj(F.silu(self.c_fc2(x)) * self.c_fc(x))
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, config.norm_eps)
        self.mlp = MLP(config)

    def forward(self, x, freqs_cis=None, start_pos=None, mask=None):
        x = x + self.attn(self.ln_1(x), freqs_cis, start_pos, mask)
        x = x + self.mlp(self.ln_2(x))
        return x

# holds llama3 model hyperparameters

@dataclass
class LlamaConfig:
    block_size: int = 8192
    vocab_size: int = 128256
    n_layer: int = 16
    n_head: int = 32
    n_kv_head: int = 8
    n_embd: int = 2048
    hidden_dim: int = 8192
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    use_scaled_rope: bool = True
    max_model_len: int = 256
    max_gen_batch_size: int = 256

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        assert self.n_kv_head <= self.n_head
        assert self.n_head % self.n_kv_head == 0
        assert self.n_embd % self.n_head == 0

# llama3 model definition

class LLaMA3(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd, config.norm_eps),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)

        self.register_buffer('freqs_cis', precompute_freqs_cis(
            config.n_embd // config.n_head,
            config.block_size * 2,
            config.rope_theta,
            config.use_scaled_rope,
        ))

    def forward(self, idx, targets=None, return_logits=True, start_pos=0):
        _, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the LLaMA model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        freqs_cis = self.freqs_cis[start_pos:start_pos+t]

        mask = torch.triu(torch.ones((t, t), device=next(self.parameters()).device, dtype=torch.bool), diagonal=1)

        for i, block in enumerate(self.transformer.h):
            torch.cuda.nvtx.range_push("layer")
            x = block(x, freqs_cis, start_pos, mask)
            torch.cuda.nvtx.range_pop()
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x).float()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]).float() # note: using list [-1] to preserve the time dim
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

    @staticmethod
    def adapt_llama_state_dict_keys_hf(checkpoint, config: LlamaConfig):
        # Modify key names from HuggingFace's LLaMA to our LLaMA
        # our key names are derived from GPT-2's key names
        checkpoint['transformer.wte.weight'] = checkpoint.pop('model.embed_tokens.weight')

        # We need to unpermute K and V because HF script permuted the original Meta-LLaMA weights
        # see: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py
        def unpermute(w, n_heads, dim1, dim2):
            return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

        for i in range(config.n_layer):
            for name in ['input_layernorm', 'post_attention_layernorm']:
                old_key = f'model.layers.{i}.{name}.weight'  # e.g. layers.x.attention_norm.weight -> transformer.h.x.ln_1.weight
                new_key = f'transformer.h.{i}.ln_{1 if name == "input_layernorm" else 2}.weight'
                checkpoint[new_key] = checkpoint.pop(old_key)

        for i in range(config.n_layer):
            for name in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']:
                old_key = f'model.layers.{i}.{name}.weight'
                new_key = f'transformer.h.{i}.attn.c_attn.weight'
                if name == 'self_attn.q_proj':
                    checkpoint[new_key] = unpermute(checkpoint.pop(old_key), config.n_head, config.n_embd, config.n_embd)
                else:  # merge 3 weights into transformer.h.x.attn.c_attn.weight
                    tensor = checkpoint.pop(old_key)
                    if name == 'self_attn.k_proj':
                        tensor = unpermute(tensor, config.n_kv_head, config.n_kv_head * (config.n_embd // config.n_head), config.n_embd)
                    checkpoint[new_key] = torch.cat((checkpoint[new_key], tensor), dim=0)
            old_key = f'model.layers.{i}.self_attn.o_proj.weight'
            new_key = f'transformer.h.{i}.attn.c_proj.weight'
            checkpoint[new_key] = checkpoint.pop(old_key)

        ffn_map = {'gate_proj': 'c_fc2', 'down_proj': 'c_proj', 'up_proj': 'c_fc'}
        for i in range(config.n_layer):
            for name in ['gate_proj', 'down_proj', 'up_proj']:
                old_key = f'model.layers.{i}.mlp.{name}.weight'
                new_key = f'transformer.h.{i}.mlp.{ffn_map[name]}.weight'
                checkpoint[new_key] = checkpoint.pop(old_key)

        checkpoint['transformer.ln_f.weight'] = checkpoint.pop('model.norm.weight')

        return checkpoint

    @classmethod
    def from_pretrained_llama3_hf(cls, model_id, max_model_len, max_gen_batch_size):
        """Loads pretrained LLaMA model weights from HuggingFace"""

        original_default_dtype = torch.get_default_dtype()  # save the default type
        torch.set_default_dtype(torch.bfloat16)
        model_args = LlamaConfig()

        model = AutoModelForCausalLM.from_pretrained(model_id)

        model_args.n_layer = model.config.num_hidden_layers
        model_args.n_head = model.config.num_attention_heads
        model_args.n_kv_head = model.config.num_key_value_heads
        model_args.n_embd = model.config.hidden_size
        model_args.hidden_dim = model.config.intermediate_size
        model_args.max_model_len = max_model_len
        model_args.max_gen_batch_size = max_gen_batch_size

        checkpoint = LLaMA3.adapt_llama_state_dict_keys_hf(model.state_dict(), model_args)
        original_default_device = torch.get_default_device()  # save the default type

        torch.set_default_device("cuda")

        model = LLaMA3(model_args)
        model.load_state_dict(checkpoint, strict=False)

        torch.set_default_dtype(original_default_dtype)
        torch.set_default_device(original_default_device)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_id = 128004  # this is the pad token id for LLaMA 3.x base, we need to set this explicitly as our generate func expects it
        tokenizer.stop_tokens = [tokenizer.eos_token_id]
        model.tokenizer = tokenizer
        
        return model

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        echo: bool = False,
        ignore_eos: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.

        """
        bsz = len(prompt_tokens)
        assert bsz <= self.config.max_gen_batch_size, f"Batch size {bsz} exceeds the maximum generation batch size {self.config.max_gen_batch_size}"
        device = next(self.parameters()).device

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= self.config.block_size, f"Prompt length {max_prompt_len} exceeds the maximum block size {self.config.block_size}"
        total_len = min(self.config.block_size, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
        for idx, t in enumerate(prompt_tokens):
            tokens[idx, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device=device)
        input_text_mask = tokens != pad_id

        if min_prompt_len == total_len:
            logits, _ = self.forward(tokens, start_pos=prev_pos)

        stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens)).to(device)

        for cur_pos in range(min_prompt_len, total_len):
            torch.cuda.nvtx.range_push("forward")
            logits, _ = self.forward(tokens[:, prev_pos:cur_pos], start_pos=prev_pos)
            torch.cuda.nvtx.range_pop()
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            eos_reached |= ~input_text_mask[:, cur_pos] & torch.isin(next_token, stop_tokens)
            prev_pos = cur_pos
            # ignore eos token useful when in benchmark mode
            if all(eos_reached) and not(ignore_eos):
                break

        out_tokens = []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            # cut to after eos tok if any
            for stop_token in self.tokenizer.stop_tokens:
                try:
                    eos_idx = toks.index(stop_token)
                    toks = toks[:eos_idx]
                except ValueError:
                    pass
            out_tokens.append(toks)
        return out_tokens

# util implementing top p sampling 

def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    
    return next_token
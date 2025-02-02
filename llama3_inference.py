from model import LLaMA3
import torch
import time

import os
os.environ['TOKENIZERS_PARALLELISM'] = "false"

# simple wrapper to load llama3 model from the hugging face hub, and warmup for inference

def llama3_inference_engine(model_name, max_model_len, max_gen_batch_size):

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    assert device_type in {'cuda'}, "GPU required to run LLaMA 3"
    print(f"using device: {device}")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    model = LLaMA3.from_pretrained_llama3_hf(model_name, max_model_len, max_gen_batch_size)

    print("compiling the model...")
    model = torch.compile(model)

    print("model warmup...")
    with torch.no_grad():
        torch.cuda.nvtx.range_push("warmup")
        warmup_prompt = "warmup"
        _ = model.generate([model.tokenizer(warmup_prompt).input_ids], max_gen_len=2, temperature=0.6, top_p=0.9, echo=False)
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
    time.sleep(1)

    return model
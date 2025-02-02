from llama3_inference import llama3_inference_engine
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test llama3 custom inference implementation")
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default = "unsloth/Llama-3.2-1B",
        help="model name from the hugging face hub",
    )
    parser.add_argument(
        "--max_gen_len",
        type=int,
        required=False,
        default = 64,
        help="maximum output tokens",
    )
    args = parser.parse_args()

    model_name = args.model_name
    max_gen_len = args.max_gen_len
    max_model_len = 4096
    max_gen_batch_size = 1

    print("Testing custom llama3 model inference implementation:")
    print("==================================")
    print(f"model name: {model_name}")
    print(f"max output tokens: {max_gen_len}")
    print("==================================")

    # prepare model for inference
    model = llama3_inference_engine(model_name, max_model_len, max_gen_batch_size)

    print("running inference...")
    
    # prompt example
    prompts = [
    "Clearly, the meaning of life is" for _ in range(model.config.max_gen_batch_size)
    ]
    prompt_tokens = [model.tokenizer(x).input_ids for x in prompts]
    generation_tokens = model.generate(prompt_tokens, max_gen_len=max_gen_len, temperature=0.6, top_p=0.9, echo=False)
    results = [{"generation": model.tokenizer.decode(t)} for t in generation_tokens]

    # print output tokens
    for prompt, result in zip(prompts, results):
        print("==================================")
        print(prompt, end="")
        print(f"{result['generation']}")
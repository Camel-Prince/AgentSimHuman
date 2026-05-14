"""Quick smoke test: load Qwen2.5-3B-Instruct from MODELSCOPE_CACHE and run one chat turn.

Usage:
    python scripts/test_load_qwen_chat.py
    python scripts/test_load_qwen_chat.py --prompt "你好，自我介绍一下"
"""

import argparse
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_model_path() -> str:
    cache = os.environ.get("MODELSCOPE_CACHE", "/data1/wangzixu/.cache/modelscope")
    return os.path.join(cache, "hub/models/Qwen/Qwen2___5-3B-Instruct")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="用一句话介绍你自己。")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model_path = resolve_model_path()
    print(f"[INFO] model_path = {model_path}")
    assert os.path.isdir(model_path), f"model path not found: {model_path}"

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"[INFO] loaded in {time.time() - t0:.1f}s on {args.device}")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": args.prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    t1 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    gen_ids = output_ids[0, input_len:]
    reply = tokenizer.decode(gen_ids, skip_special_tokens=True)
    print(f"[INFO] generated {gen_ids.numel()} tokens in {time.time() - t1:.1f}s")
    print("---- prompt ----")
    print(args.prompt)
    print("---- reply ----")
    print(reply)


if __name__ == "__main__":
    main()

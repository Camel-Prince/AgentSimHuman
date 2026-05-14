"""Batch smoke test for cached Qwen chat models under $MODELSCOPE_CACHE.

Loads each model (sequentially to avoid OOM), applies chat template, generates
one reply, then releases GPU memory before moving on.

Usage:
    # verify all five models with default prompt
    python scripts/test_load_qwen_models_batch.py

    # custom prompt / only a subset
    python scripts/test_load_qwen_models_batch.py --prompt "你好" \
        --models Qwen2___5-1___5B-Instruct Qwen2___5-3B-Instruct
"""

import argparse
import gc
import os
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODELS = [
    "Qwen2___5-1___5B-Instruct",
    "Qwen2___5-3B-Instruct",
    "Qwen2___5-7B-Instruct",
    "Qwen3-1___7B",
    "Qwen3-8B",
]


def model_dir(name: str) -> str:
    cache = os.environ.get("MODELSCOPE_CACHE", "/data1/wangzixu/.cache/modelscope")
    return os.path.join(cache, "hub/models/Qwen", name)


def test_one(name: str, prompt: str, max_new_tokens: int, device: str) -> dict:
    path = model_dir(name)
    print("\n" + "=" * 72)
    print(f"[TEST] {name}\n  path: {path}")
    if not os.path.isdir(path):
        print(f"[SKIP] path not found")
        return {"name": name, "status": "missing", "reply": ""}

    t0 = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device,
            trust_remote_code=True,
        )
        model.eval()
        load_s = time.time() - t0

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        t1 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        gen_ids = out[0, input_len:]
        reply = tokenizer.decode(gen_ids, skip_special_tokens=True)
        gen_s = time.time() - t1

        print(f"[OK] load={load_s:.1f}s  gen={gen_s:.1f}s  tokens={gen_ids.numel()}")
        print(f"---- reply ----\n{reply}")
        return {"name": name, "status": "ok", "reply": reply}
    except Exception as e:
        traceback.print_exc()
        return {"name": name, "status": f"error: {type(e).__name__}: {e}", "reply": ""}
    finally:
        # release GPU memory before next model
        for var in ("model", "tokenizer", "inputs", "out", "gen_ids"):
            if var in locals():
                del locals()[var]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="用一句话介绍你自己。")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS,
                        help="model directory names under $MODELSCOPE_CACHE/hub/models/Qwen/")
    args = parser.parse_args()

    print(f"[INFO] MODELSCOPE_CACHE={os.environ.get('MODELSCOPE_CACHE')}")
    print(f"[INFO] device={args.device}  prompt={args.prompt!r}")

    results = [test_one(m, args.prompt, args.max_new_tokens, args.device) for m in args.models]

    print("\n" + "#" * 72)
    print("SUMMARY")
    for r in results:
        print(f"  {r['status']:<40}  {r['name']}")
    failed = [r for r in results if r["status"] != "ok"]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import get_model_params


def setup_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_prompts(path: Path) -> list[dict[str, str]]:
    prompts = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def build_chat(tokenizer, prompt: str, open_thinking: bool = False) -> torch.Tensor:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=open_thinking,
    )
    return tokenizer(text, return_tensors="pt", truncation=True)


def load_official(path: Path, device: str):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
    get_model_params(model, model.config)
    return model.half().eval().to(device), tokenizer


def load_native(weight_path: Path, tokenizer_path: Path, device: str, hidden_size: int, num_hidden_layers: int):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(MiniMindConfig(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers))
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    get_model_params(model, model.config)
    return model.half().eval().to(device), tokenizer


@torch.inference_mode()
def generate_one(model, tokenizer, prompt: str, args, seed: int) -> tuple[str, float, int]:
    setup_seed(seed)
    inputs = build_chat(tokenizer, prompt, open_thinking=bool(args.open_thinking))
    inputs = {k: v.to(args.device) for k, v in inputs.items()}
    start = time.time()
    generated = model.generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
    )
    elapsed = max(time.time() - start, 1e-6)
    new_tokens = len(generated[0]) - len(inputs["input_ids"][0])
    response = tokenizer.decode(generated[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True).strip()
    return response, elapsed, new_tokens


def iter_results(model_name: str, model, tokenizer, prompts: Iterable[dict[str, str]], args):
    for index, item in enumerate(prompts):
        seed = args.seed + index
        response, elapsed, new_tokens = generate_one(model, tokenizer, item["prompt"], args, seed)
        yield {
            "model": model_name,
            "prompt_id": item.get("id", str(index)),
            "prompt": item["prompt"],
            "response": response,
            "seed": seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "new_tokens": new_tokens,
            "seconds": round(elapsed, 4),
            "tokens_per_second": round(new_tokens / elapsed, 4),
        }


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_markdown(rows: list[dict], path: Path) -> None:
    by_prompt: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_prompt.setdefault(row["prompt_id"], {})[row["model"]] = row

    lines = [
        "# MiniMind Mini Reproduction: Model Comparison",
        "",
        "说明：自训模型使用 mini 数据组合训练，官方 `minimind-3` 通常经过更充分的数据与流程训练；本对比重点是复现实验流程和同题输出差异，不假设自训模型应优于官方模型。",
        "",
        f"- temperature: `{rows[0]['temperature']}`",
        f"- top_p: `{rows[0]['top_p']}`",
        f"- max_new_tokens: `{rows[0]['max_new_tokens']}`",
        "",
    ]
    for prompt_id, grouped in by_prompt.items():
        sample = next(iter(grouped.values()))
        lines.extend([f"## {prompt_id}", "", f"**Prompt:** {sample['prompt']}", ""])
        for model_name in ["official", "self_trained"]:
            row = grouped.get(model_name)
            if row is None:
                continue
            lines.extend(
                [
                    f"### {model_name}",
                    "",
                    row["response"] or "(empty response)",
                    "",
                    f"`{row['new_tokens']} tokens, {row['tokens_per_second']} tokens/s`",
                    "",
                ]
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare official MiniMind and locally trained MiniMind.")
    parser.add_argument("--official-model", default="minimind-3")
    parser.add_argument("--self-weight", default="out/full_sft_mini_768.pth")
    parser.add_argument("--tokenizer-path", default="model")
    parser.add_argument("--prompts", default="experiments/prompts.jsonl")
    parser.add_argument("--results-dir", default="experiments/results")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--open-thinking", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    prompts = load_prompts(ROOT / args.prompts)
    official, official_tokenizer = load_official(ROOT / args.official_model, args.device)
    native, native_tokenizer = load_native(
        ROOT / args.self_weight,
        ROOT / args.tokenizer_path,
        args.device,
        args.hidden_size,
        args.num_hidden_layers,
    )

    rows = []
    rows.extend(iter_results("official", official, official_tokenizer, prompts, args))
    rows.extend(iter_results("self_trained", native, native_tokenizer, prompts, args))

    results_dir = ROOT / args.results_dir
    write_jsonl(rows, results_dir / "model_comparison.jsonl")
    write_markdown(rows, results_dir / "model_comparison.md")
    print(f"Wrote {len(rows)} generations to {results_dir}")


if __name__ == "__main__":
    main()

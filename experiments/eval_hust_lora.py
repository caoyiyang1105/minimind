import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.model_lora import apply_lora, load_lora
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import get_model_params


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_prompts(path: Path) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompts.append({"id": item.get("id", str(len(prompts))), "prompt": item["prompt"]})
    return prompts


def load_native_model(
    weight_path: Path,
    device: str,
    hidden_size: int,
    num_hidden_layers: int,
    use_moe: bool,
    lora_path: Path | None = None,
) -> MiniMindForCausalLM:
    config = MiniMindConfig(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, use_moe=use_moe)
    model = MiniMindForCausalLM(config)
    model.load_state_dict(torch.load(weight_path, map_location=device), strict=True)
    if lora_path is not None:
        apply_lora(model)
        load_lora(model, str(lora_path))
    get_model_params(model, model.config)
    return model.half().eval().to(device)


@torch.inference_mode()
def generate_one(
    model: MiniMindForCausalLM,
    tokenizer,
    prompt: str,
    device: str,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    open_thinking: bool,
    do_sample: bool,
) -> dict[str, object]:
    setup_seed(seed)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=open_thinking,
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    start = time.time()
    generated_ids = model.generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=1.0,
    )
    elapsed = max(time.time() - start, 1e-6)
    new_tokens = int(generated_ids.shape[-1] - inputs["input_ids"].shape[-1])
    response = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    return {
        "response": response.strip(),
        "tokens": new_tokens,
        "tokens_per_second": round(new_tokens / elapsed, 4),
    }


def evaluate_model(label: str, model: MiniMindForCausalLM, tokenizer, prompts, args) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, item in enumerate(prompts):
        generation = generate_one(
            model=model,
            tokenizer=tokenizer,
            prompt=item["prompt"],
            device=args.device,
            seed=args.seed + idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            open_thinking=bool(args.open_thinking),
            do_sample=bool(args.do_sample),
        )
        rows.append(
            {
                "prompt_id": item["id"],
                "prompt": item["prompt"],
                "model": label,
                "generation": generation,
            }
        )
    return rows


def write_markdown(path: Path, rows: list[dict[str, object]], args) -> None:
    grouped: dict[str, list[dict[str, object]]] = {}
    prompt_text: dict[str, str] = {}
    for row in rows:
        grouped.setdefault(row["prompt_id"], []).append(row)
        prompt_text[row["prompt_id"]] = row["prompt"]

    lines = [
        "# HUST 2025 Graduate Handbook Fine-tuning Comparison",
        "",
        "说明：`base_self` 是 mini 复现实验得到的 SFT 权重；其它模型是在同一基座上用《2025研究生手册》构造数据后微调得到的权重。",
        "",
        f"- temperature: `{args.temperature}`",
        f"- top_p: `{args.top_p}`",
        f"- max_new_tokens: `{args.max_new_tokens}`",
        "",
    ]
    for prompt_id, items in grouped.items():
        lines.append(f"## {prompt_id}")
        lines.append("")
        lines.append(f"**Prompt:** {prompt_text[prompt_id]}")
        lines.append("")
        for item in items:
            generation = item["generation"]
            lines.append(f"### {item['model']}")
            lines.append("")
            lines.append(generation["response"] or "(empty)")
            lines.append("")
            lines.append(f"`{generation['tokens']} tokens, {generation['tokens_per_second']} tokens/s`")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base MiniMind with HUST handbook LoRA.")
    parser.add_argument("--base-weight", type=Path, default=Path("out/full_sft_mini_768.pth"))
    parser.add_argument("--lora-weight", default="out/lora_hust_handbook_768.pth", help="LoRA weight path, or None to skip.")
    parser.add_argument("--full-weight", default=None, help="Full-SFT weight path to evaluate, optional.")
    parser.add_argument("--full-label", default="hust_full_sft")
    parser.add_argument("--tokenizer-path", type=Path, default=Path("model"))
    parser.add_argument("--prompts", type=Path, default=Path("experiments/hust_prompts.jsonl"))
    parser.add_argument("--results-dir", type=Path, default=Path("experiments/results"))
    parser.add_argument("--output-prefix", default="hust_lora_comparison")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--use-moe", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--open-thinking", type=int, default=0)
    parser.add_argument("--do-sample", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    prompts = load_prompts(args.prompts)
    all_rows: list[dict[str, object]] = []

    base_model = load_native_model(
        weight_path=args.base_weight,
        device=args.device,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    all_rows.extend(evaluate_model("base_self", base_model, tokenizer, prompts, args))
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.lora_weight and str(args.lora_weight).lower() != "none":
        lora_model = load_native_model(
            weight_path=args.base_weight,
            device=args.device,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            lora_path=Path(args.lora_weight),
        )
        all_rows.extend(evaluate_model("hust_lora", lora_model, tokenizer, prompts, args))
        del lora_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.full_weight and str(args.full_weight).lower() != "none":
        full_model = load_native_model(
            weight_path=Path(args.full_weight),
            device=args.device,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
        )
        all_rows.extend(evaluate_model(args.full_label, full_model, tokenizer, prompts, args))

    jsonl_path = args.results_dir / f"{args.output_prefix}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_markdown(args.results_dir / f"{args.output_prefix}.md", all_rows, args)
    print(f"Wrote {len(all_rows)} generations to {args.results_dir.resolve()}")


if __name__ == "__main__":
    main()

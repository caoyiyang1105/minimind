#!/usr/bin/env python
import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


LOG_PATTERN = re.compile(
    r"Epoch:\[(?P<epoch>\d+)/(?P<epochs>\d+)\]\((?P<step>\d+)/(?P<iters>\d+)\), "
    r"loss: (?P<loss>[-+0-9.eE]+), "
    r"logits_loss: (?P<logits_loss>[-+0-9.eE]+), "
    r"aux_loss: (?P<aux_loss>[-+0-9.eE]+), "
    r"lr: (?P<lr>[-+0-9.eE]+)"
)


def parse_log(path: Path, stage: str) -> list[dict[str, float | int | str]]:
    rows = []
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = LOG_PATTERN.search(line)
        if not match:
            continue
        item = match.groupdict()
        epoch = int(item["epoch"])
        iters = int(item["iters"])
        step = int(item["step"])
        rows.append(
            {
                "stage": stage,
                "epoch": epoch,
                "epochs": int(item["epochs"]),
                "step": step,
                "iters": iters,
                "global_step": (epoch - 1) * iters + step,
                "loss": float(item["loss"]),
                "logits_loss": float(item["logits_loss"]),
                "aux_loss": float(item["aux_loss"]),
                "lr": float(item["lr"]),
                "source_log": str(path),
            }
        )
    if not rows:
        raise ValueError(f"No loss records found in {path}")
    return rows


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_loss(pretrain_rows: list[dict], sft_rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, title, rows in [
        (axes[0], "Pretrain loss", pretrain_rows),
        (axes[1], "SFT loss", sft_rows),
    ]:
        ax.plot([r["global_step"] for r in rows], [r["loss"] for r in rows], label="loss")
        ax.plot([r["global_step"] for r in rows], [r["logits_loss"] for r in rows], label="logits_loss")
        if any(abs(r["aux_loss"]) > 1e-12 for r in rows):
            ax.plot([r["global_step"] for r in rows], [r["aux_loss"] for r in rows], label="aux_loss")
        ax.set_title(title)
        ax.set_xlabel("global step")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_single(rows: list[dict], title: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    ax.plot([r["global_step"] for r in rows], [r["loss"] for r in rows], label="loss")
    ax.plot([r["global_step"] for r in rows], [r["logits_loss"] for r in rows], label="logits_loss")
    if any(abs(r["aux_loss"]) > 1e-12 for r in rows):
        ax.plot([r["global_step"] for r in rows], [r["aux_loss"] for r in rows], label="aux_loss")
    ax.set_title(title)
    ax.set_xlabel("global step")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse MiniMind training logs and plot loss curves.")
    parser.add_argument("--pretrain-log", default="experiments/logs/pretrain_mini.log")
    parser.add_argument("--sft-log", default="experiments/logs/full_sft_mini.log")
    parser.add_argument("--hust-log", default=None, help="Optional HUST handbook fine-tuning log.")
    parser.add_argument("--hust-label", default="hust_lora", help="Stage label and filename prefix for --hust-log.")
    parser.add_argument("--results-dir", default="experiments/results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    pretrain_rows = parse_log(Path(args.pretrain_log), "pretrain")
    sft_rows = parse_log(Path(args.sft_log), "sft")
    hust_rows = parse_log(Path(args.hust_log), args.hust_label) if args.hust_log else []

    write_csv(pretrain_rows, results_dir / "pretrain_loss.csv")
    write_csv(sft_rows, results_dir / "sft_loss.csv")
    if hust_rows:
        write_csv(hust_rows, results_dir / f"{args.hust_label}_loss.csv")
        plot_single(hust_rows, f"{args.hust_label} loss", results_dir / f"{args.hust_label}_loss_curve.png")
    write_csv(pretrain_rows + sft_rows + hust_rows, results_dir / "loss_records.csv")
    plot_loss(pretrain_rows, sft_rows, results_dir / "loss_curve.png")

    print(f"Parsed {len(pretrain_rows)} pretrain records and {len(sft_rows)} SFT records.")
    if hust_rows:
        print(f"Parsed {len(hust_rows)} {args.hust_label} records.")
    print(f"Wrote results to {results_dir}")


if __name__ == "__main__":
    main()

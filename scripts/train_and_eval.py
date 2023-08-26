from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from eagle_draft.evaluate import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-steps", type=int, default=160)
    parser.add_argument("--draft-steps", type=int, default=180)
    parser.add_argument("--prompts", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--num-drafts", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("results/local_cpu.json"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.smoke:
        args.target_steps = 3
        args.draft_steps = 3
        args.prompts = 2
        args.max_new_tokens = 6
        args.output = Path("results/smoke.tmp")
    result = evaluate(
        target_steps=args.target_steps,
        draft_steps=args.draft_steps,
        prompts=args.prompts,
        max_new_tokens=args.max_new_tokens,
        num_drafts=args.num_drafts,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["training"], indent=2))
    for temperature, metrics in result["temperatures"].items():
        summary = {
            "temperature": temperature,
            "acceptance_length": metrics["speculative"]["acceptance_length"],
            "baseline_tokens_per_second": metrics["baseline"]["tokens_per_second"],
            "speculative_tokens_per_second": metrics["speculative"]["tokens_per_second"],
            "throughput_speedup": metrics["throughput_speedup"],
            "all_greedy_tokens_match": metrics["all_greedy_tokens_match"],
        }
        print(json.dumps(summary, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import platform
from datetime import datetime
from typing import Any

import torch

from eagle_draft.draft import EAGLEDraftHead
from eagle_draft.model import TinyTargetModel
from eagle_draft.speculative import DecodeResult, speculative_decode, target_decode
from eagle_draft.train import train_draft, train_target


def decode_metrics(result: DecodeResult) -> dict[str, float | int]:
    return {
        "generated_tokens": result.generated_tokens,
        "accepted_draft_tokens": result.accepted_draft_tokens,
        "verification_passes": result.verification_passes,
        "target_calls": result.target_calls,
        "acceptance_length": result.acceptance_length,
        "elapsed_seconds": result.elapsed_seconds,
        "tokens_per_second": result.tokens_per_second,
    }


def aggregate_metrics(results: list[DecodeResult]) -> dict[str, float | int]:
    generated = sum(result.generated_tokens for result in results)
    accepted = sum(result.accepted_draft_tokens for result in results)
    verifications = sum(result.verification_passes for result in results)
    target_calls = sum(result.target_calls for result in results)
    elapsed = sum(result.elapsed_seconds for result in results)
    acceptance_length = generated / (verifications or target_calls)
    return {
        "generated_tokens": generated,
        "accepted_draft_tokens": accepted,
        "verification_passes": verifications,
        "target_calls": sum(result.target_calls for result in results),
        "acceptance_length": acceptance_length,
        "elapsed_seconds": elapsed,
        "tokens_per_second": generated / elapsed,
    }


def evaluate(
    target_steps: int = 160,
    draft_steps: int = 180,
    prompts: int = 8,
    max_new_tokens: int = 32,
    num_drafts: int = 4,
    vocab_size: int = 31,
    hidden_size: int = 32,
) -> dict[str, Any]:
    if prompts <= 0:
        raise ValueError("prompts must be positive")
    torch.manual_seed(2026)
    target = TinyTargetModel(vocab_size=vocab_size, hidden_size=hidden_size)
    draft = EAGLEDraftHead(vocab_size=vocab_size, hidden_size=hidden_size)
    target_history = train_target(target, steps=target_steps)
    draft_history = train_draft(target, draft, steps=draft_steps)
    output: dict[str, Any] = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
        },
        "configuration": {
            "target_steps": target_steps,
            "draft_steps": draft_steps,
            "prompts": prompts,
            "max_new_tokens": max_new_tokens,
            "num_drafts": num_drafts,
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
        },
        "training": {
            "target_initial_loss": target_history.initial_loss,
            "target_final_loss": target_history.final_loss,
            "draft_initial_loss": draft_history.initial_loss,
            "draft_final_loss": draft_history.final_loss,
        },
        "temperatures": {},
    }
    for temperature in (0.0, 1.0):
        baselines = []
        speculative = []
        greedy_matches = []
        cases = []
        for index in range(prompts):
            prefix = [index % vocab_size, (index + 3) % vocab_size, (index + 7) % vocab_size]
            seed = 1000 + index
            baseline = target_decode(
                target, prefix, max_new_tokens, temperature=temperature, seed=seed
            )
            accelerated = speculative_decode(
                target,
                draft,
                prefix,
                max_new_tokens,
                num_drafts=num_drafts,
                temperature=temperature,
                seed=seed,
            )
            matches = baseline.tokens == accelerated.tokens
            if temperature == 0:
                greedy_matches.append(matches)
            baselines.append(baseline)
            speculative.append(accelerated)
            cases.append(
                {
                    "prompt_index": index,
                    "baseline": decode_metrics(baseline),
                    "speculative": decode_metrics(accelerated),
                    "greedy_tokens_match": matches if temperature == 0 else None,
                }
            )
        baseline_aggregate = aggregate_metrics(baselines)
        speculative_aggregate = aggregate_metrics(speculative)
        speedup = (
            speculative_aggregate["tokens_per_second"]
            / baseline_aggregate["tokens_per_second"]
        )
        output["temperatures"][str(temperature)] = {
            "baseline": baseline_aggregate,
            "speculative": speculative_aggregate,
            "throughput_speedup": speedup,
            "all_greedy_tokens_match": all(greedy_matches) if temperature == 0 else None,
            "cases": cases,
        }
    return output

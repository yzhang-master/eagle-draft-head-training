from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from eagle_draft.draft import EAGLEDraftHead
from eagle_draft.model import TinyTargetModel


@dataclass
class DecodeResult:
    tokens: list[int]
    elapsed_seconds: float
    target_calls: int
    verification_passes: int
    accepted_draft_tokens: int
    generated_tokens: int

    @property
    def acceptance_length(self) -> float:
        if self.verification_passes == 0:
            return float(self.generated_tokens)
        return self.generated_tokens / self.verification_passes

    @property
    def tokens_per_second(self) -> float:
        return self.generated_tokens / self.elapsed_seconds


def rejection_residual(target_probs: torch.Tensor, draft_probs: torch.Tensor) -> torch.Tensor:
    residual = (target_probs - draft_probs).clamp_min(0)
    mass = residual.sum()
    if mass <= 0:
        return target_probs
    return residual / mass


def acceptance_probability(
    target_probs: torch.Tensor, draft_probs: torch.Tensor, token: int
) -> float:
    draft_probability = float(draft_probs[token])
    if draft_probability <= 0:
        return 0.0
    return min(1.0, float(target_probs[token]) / draft_probability)


def _generator(device: torch.device, seed: int | None) -> torch.Generator:
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    return generator


def speculative_decode(
    target: TinyTargetModel,
    draft: EAGLEDraftHead,
    prefix: list[int],
    max_new_tokens: int,
    num_drafts: int = 4,
    temperature: float = 0.0,
    seed: int | None = None,
) -> DecodeResult:
    if not prefix or max_new_tokens <= 0 or num_drafts <= 0:
        raise ValueError("prefix, max_new_tokens, and num_drafts must be positive")
    if temperature < 0:
        raise ValueError("temperature must be nonnegative")
    device = next(target.parameters()).device
    generator = _generator(device, seed)
    tokens = list(prefix)
    target.eval()
    draft.eval()
    target_calls = 0
    verification_passes = 0
    accepted = 0
    started = time.perf_counter()
    with torch.inference_mode():
        _, current_hidden = target.forward_with_hidden(
            torch.tensor([tokens], dtype=torch.long, device=device)
        )
        target_calls += 1
        current_hidden = current_hidden[0, -1]
        while len(tokens) - len(prefix) < max_new_tokens:
            remaining = max_new_tokens - (len(tokens) - len(prefix))
            proposal = draft.propose(
                current_hidden, tokens[-1], min(num_drafts, remaining), temperature, generator
            )
            base_length = len(tokens)
            proposal_tokens = proposal.tokens.tolist()
            verification_input = torch.tensor(
                [tokens + proposal_tokens], dtype=torch.long, device=device
            )
            logits, hidden = target.forward_with_hidden(verification_input)
            target_calls += 1
            verification_passes += 1
            committed: list[int] = []
            rejected = False
            for index, draft_token in enumerate(proposal_tokens):
                target_logits = logits[0, base_length - 1 + index]
                if temperature == 0:
                    target_token = int(target_logits.argmax())
                    if target_token == draft_token:
                        committed.append(draft_token)
                        accepted += 1
                    else:
                        committed.append(target_token)
                        rejected = True
                        break
                else:
                    target_probs = torch.softmax(target_logits / temperature, dim=-1)
                    draft_probs = proposal.probabilities[index]
                    probability = acceptance_probability(target_probs, draft_probs, draft_token)
                    if torch.rand((), generator=generator, device=device) < probability:
                        committed.append(draft_token)
                        accepted += 1
                    else:
                        residual = rejection_residual(target_probs, draft_probs)
                        replacement = torch.multinomial(
                            residual, 1, generator=generator
                        ).item()
                        committed.append(int(replacement))
                        rejected = True
                        break
            if not rejected:
                bonus_logits = logits[0, base_length + len(proposal_tokens) - 1]
                if temperature == 0:
                    bonus = int(bonus_logits.argmax())
                else:
                    bonus_probs = torch.softmax(bonus_logits / temperature, dim=-1)
                    bonus = torch.multinomial(
                        bonus_probs, 1, generator=generator
                    ).item()
                committed.append(int(bonus))
            tokens.extend(committed[:remaining])
            if len(tokens) - len(prefix) < max_new_tokens:
                if rejected:
                    current_hidden = hidden[0, base_length + len(committed) - 1]
                else:
                    _, current_hidden = target.forward_with_hidden(
                        torch.tensor([tokens], dtype=torch.long, device=device)
                    )
                    target_calls += 1
                    current_hidden = current_hidden[0, -1]
    elapsed = time.perf_counter() - started
    generated = len(tokens) - len(prefix)
    return DecodeResult(
        tokens=tokens,
        elapsed_seconds=elapsed,
        target_calls=target_calls,
        verification_passes=verification_passes,
        accepted_draft_tokens=accepted,
        generated_tokens=generated,
    )


def target_decode(
    target: TinyTargetModel,
    prefix: list[int],
    max_new_tokens: int,
    temperature: float = 0.0,
    seed: int | None = None,
) -> DecodeResult:
    if not prefix or max_new_tokens <= 0:
        raise ValueError("prefix and max_new_tokens must be positive")
    if temperature < 0:
        raise ValueError("temperature must be nonnegative")
    device = next(target.parameters()).device
    generator = _generator(device, seed)
    tokens = list(prefix)
    target.eval()
    target_calls = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = target(input_ids)[0, -1]
            target_calls += 1
            if temperature == 0:
                next_token = int(logits.argmax())
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1, generator=generator).item()
            tokens.append(int(next_token))
    elapsed = time.perf_counter() - started
    return DecodeResult(
        tokens=tokens,
        elapsed_seconds=elapsed,
        target_calls=target_calls,
        verification_passes=0,
        accepted_draft_tokens=0,
        generated_tokens=max_new_tokens,
    )

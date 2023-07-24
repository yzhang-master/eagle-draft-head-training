from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from eagle_draft.data import cyclic_batch
from eagle_draft.draft import EAGLEDraftHead
from eagle_draft.model import TinyTargetModel


@dataclass
class TrainingHistory:
    losses: list[float]

    @property
    def initial_loss(self) -> float:
        return self.losses[0]

    @property
    def final_loss(self) -> float:
        return self.losses[-1]


def train_target(
    target: TinyTargetModel,
    steps: int = 160,
    batch_size: int = 32,
    sequence_length: int = 20,
    learning_rate: float = 0.03,
    seed: int = 11,
) -> TrainingHistory:
    if steps <= 0:
        raise ValueError("steps must be positive")
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(target.parameters(), lr=learning_rate)
    losses = []
    target.train()
    for _ in range(steps):
        tokens = cyclic_batch(batch_size, sequence_length, target.vocab_size, generator)
        logits = target(tokens)
        loss = F.cross_entropy(logits[:, :-1].flatten(0, 1), tokens[:, 1:].flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    target.eval()
    return TrainingHistory(losses)


def train_draft(
    target: TinyTargetModel,
    draft: EAGLEDraftHead,
    steps: int = 180,
    batch_size: int = 32,
    sequence_length: int = 20,
    learning_rate: float = 0.01,
    seed: int = 23,
) -> TrainingHistory:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if target.vocab_size != draft.vocab_size or target.hidden_size != draft.hidden_size:
        raise ValueError("target and draft configurations differ")
    target.eval()
    target.requires_grad_(False)
    draft.train()
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(draft.parameters(), lr=learning_rate)
    losses = []
    for _ in range(steps):
        tokens = cyclic_batch(batch_size, sequence_length, target.vocab_size, generator)
        with torch.no_grad():
            _, target_hidden = target.forward_with_hidden(tokens)
        state = None
        token_losses = []
        for position in range(sequence_length - 1):
            logits, state = draft.step(
                tokens[:, position], target_hidden[:, position], state
            )
            token_losses.append(F.cross_entropy(logits, tokens[:, position + 1]))
        loss = torch.stack(token_losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    draft.eval()
    return TrainingHistory(losses)


def target_is_frozen(target: nn.Module) -> bool:
    return all(not parameter.requires_grad for parameter in target.parameters())

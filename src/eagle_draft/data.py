from __future__ import annotations

import torch


def cyclic_batch(
    batch_size: int,
    sequence_length: int,
    vocab_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if batch_size <= 0 or sequence_length < 2 or vocab_size < 3:
        raise ValueError("batch_size > 0, sequence_length >= 2, and vocab_size >= 3 required")
    starts = torch.randint(0, vocab_size, (batch_size, 1), generator=generator)
    offsets = torch.arange(sequence_length).unsqueeze(0)
    return (starts + offsets) % vocab_size

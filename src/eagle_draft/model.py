from __future__ import annotations

import torch
from torch import nn


class TinyTargetModel(nn.Module):
    """A bigram target whose normalized token embeddings are its hidden states."""

    def __init__(self, vocab_size: int = 31, hidden_size: int = 32) -> None:
        super().__init__()
        if vocab_size < 3 or hidden_size < 2:
            raise ValueError("vocab_size >= 3 and hidden_size >= 2 are required")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward_with_hidden(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a rank-2 long tensor")
        if input_ids.shape[1] == 0:
            raise ValueError("input_ids must contain at least one token")
        hidden = self.norm(self.embedding(input_ids))
        return self.lm_head(hidden), hidden

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.forward_with_hidden(input_ids)[0]

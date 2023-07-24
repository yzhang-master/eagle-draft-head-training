from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class DraftProposal:
    tokens: torch.Tensor
    probabilities: torch.Tensor


class EAGLEDraftHead(nn.Module):
    """Recurrent draft conditioned on target features and previously drafted tokens."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.input_projection = nn.Linear(2 * hidden_size, hidden_size)
        self.cell = nn.GRUCell(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def step(
        self,
        token_ids: torch.Tensor,
        features: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if token_ids.ndim != 1 or features.ndim != 2:
            raise ValueError("token_ids must be [batch] and features must be [batch, hidden]")
        if features.shape != (token_ids.shape[0], self.hidden_size):
            raise ValueError("features have the wrong shape")
        if state is None:
            state = features.new_zeros(features.shape)
        embedded = self.token_embedding(token_ids)
        inputs = self.input_projection(torch.cat((features, embedded), dim=-1))
        next_state = self.cell(inputs, state)
        logits = self.lm_head(self.norm(next_state))
        return logits, next_state

    def propose(
        self,
        target_hidden: torch.Tensor,
        last_token: int,
        num_tokens: int,
        temperature: float,
        generator: torch.Generator,
    ) -> DraftProposal:
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if temperature < 0:
            raise ValueError("temperature must be nonnegative")
        features = target_hidden.unsqueeze(0)
        token = torch.tensor([last_token], device=features.device, dtype=torch.long)
        state = None
        tokens = []
        probabilities = []
        for _ in range(num_tokens):
            logits, state = self.step(token, features, state)
            if temperature == 0:
                next_token = logits.argmax(dim=-1)
                probs = torch.softmax(logits, dim=-1)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1, generator=generator).squeeze(1)
            tokens.append(next_token.squeeze(0))
            probabilities.append(probs.squeeze(0))
            token = next_token
            features = state
        return DraftProposal(torch.stack(tokens), torch.stack(probabilities))

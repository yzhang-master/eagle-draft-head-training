import pytest
import torch

from eagle_draft.draft import EAGLEDraftHead


def test_proposal_shape_and_autoregressive_fallback_features() -> None:
    torch.manual_seed(30)
    draft = EAGLEDraftHead(vocab_size=12, hidden_size=6)
    generator = torch.Generator().manual_seed(31)
    proposal = draft.propose(torch.randn(6), 3, 4, 0.0, generator)
    assert proposal.tokens.shape == (4,)
    assert proposal.probabilities.shape == (4, 12)
    torch.testing.assert_close(proposal.probabilities.sum(dim=-1), torch.ones(4))


def test_proposal_rejects_invalid_lengths() -> None:
    draft = EAGLEDraftHead(vocab_size=12, hidden_size=6)
    with pytest.raises(ValueError, match="positive"):
        draft.propose(torch.randn(6), 3, 0, 0.0, torch.Generator())

import torch

from eagle_draft.draft import DraftProposal, EAGLEDraftHead
from eagle_draft.model import TinyTargetModel
from eagle_draft.speculative import (
    acceptance_probability,
    rejection_residual,
    speculative_decode,
    target_decode,
)


def test_greedy_speculative_decoding_exactly_matches_target() -> None:
    torch.manual_seed(10)
    target = TinyTargetModel(vocab_size=13, hidden_size=8)
    draft = EAGLEDraftHead(vocab_size=13, hidden_size=8)
    for max_new_tokens, num_drafts in [(1, 1), (2, 3), (9, 4), (10, 7)]:
        expected = target_decode(target, [1, 5, 8], max_new_tokens)
        actual = speculative_decode(
            target, draft, [1, 5, 8], max_new_tokens, num_drafts=num_drafts
        )
        assert actual.tokens == expected.tokens
        assert actual.generated_tokens == max_new_tokens
        assert actual.target_calls == actual.verification_passes + 1


def test_rejection_and_resampling_preserve_target_distribution() -> None:
    draft_probs = torch.tensor([0.40, 0.35, 0.25])
    target_probs = torch.tensor([0.20, 0.50, 0.30])
    output_probs = torch.zeros(3)
    rejection_mass = 0.0
    for token in range(3):
        probability = acceptance_probability(target_probs, draft_probs, token)
        output_probs[token] += draft_probs[token] * probability
        rejection_mass += draft_probs[token] * (1.0 - probability)
    output_probs += rejection_mass * rejection_residual(target_probs, draft_probs)
    torch.testing.assert_close(output_probs, target_probs)
    torch.testing.assert_close(
        rejection_residual(target_probs, draft_probs),
        torch.tensor([0.0, 0.15, 0.05]) / 0.20,
    )


def test_acceptance_probability_uses_min_ratio() -> None:
    target_probs = torch.tensor([0.6, 0.1, 0.3])
    draft_probs = torch.tensor([0.3, 0.4, 0.0])
    assert acceptance_probability(target_probs, draft_probs, 0) == 1.0
    assert acceptance_probability(target_probs, draft_probs, 1) == 0.25
    assert acceptance_probability(target_probs, draft_probs, 2) == 0.0


class UniformDraft:
    def eval(self) -> None:
        pass

    def propose(
        self,
        target_hidden: torch.Tensor,
        last_token: int,
        num_tokens: int,
        temperature: float,
        generator: torch.Generator,
    ) -> DraftProposal:
        del target_hidden, last_token, temperature
        vocab_size = 9
        probabilities = torch.full((num_tokens, vocab_size), 1.0 / vocab_size)
        tokens = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
        return DraftProposal(tokens=tokens, probabilities=probabilities)


def test_matching_distributions_accept_every_draft() -> None:
    target = TinyTargetModel(vocab_size=9, hidden_size=4)
    with torch.no_grad():
        for parameter in target.parameters():
            parameter.zero_()
    result = speculative_decode(
        target,
        UniformDraft(),
        [0],
        max_new_tokens=5,
        num_drafts=2,
        temperature=1.0,
        seed=19,
    )
    assert result.accepted_draft_tokens == 4
    assert result.verification_passes == 2
    assert result.acceptance_length == 5 / 2


def test_stochastic_decoding_is_reproducible() -> None:
    torch.manual_seed(21)
    target = TinyTargetModel(vocab_size=11, hidden_size=8)
    draft = EAGLEDraftHead(vocab_size=11, hidden_size=8)
    first = speculative_decode(
        target, draft, [2, 4], 8, num_drafts=3, temperature=1.0, seed=22
    )
    second = speculative_decode(
        target, draft, [2, 4], 8, num_drafts=3, temperature=1.0, seed=22
    )
    assert first.tokens == second.tokens
    assert first.accepted_draft_tokens == second.accepted_draft_tokens

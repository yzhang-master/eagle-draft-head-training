import torch

from eagle_draft.data import cyclic_batch
from eagle_draft.draft import EAGLEDraftHead
from eagle_draft.model import TinyTargetModel
from eagle_draft.train import target_is_frozen, train_draft, train_target


def test_cyclic_batch_has_real_next_token_structure() -> None:
    tokens = cyclic_batch(4, 8, 11, torch.Generator().manual_seed(3))
    assert tokens.shape == (4, 8)
    torch.testing.assert_close((tokens[:, :-1] + 1) % 11, tokens[:, 1:])


def test_target_and_draft_training_reduce_loss_and_freeze_target() -> None:
    torch.manual_seed(4)
    target = TinyTargetModel(vocab_size=11, hidden_size=12)
    target_history = train_target(
        target, steps=70, batch_size=16, sequence_length=10, seed=5
    )
    assert sum(target_history.losses[-5:]) < sum(target_history.losses[:5])

    draft = EAGLEDraftHead(vocab_size=11, hidden_size=12)
    before = {name: value.clone() for name, value in target.state_dict().items()}
    draft_history = train_draft(
        target, draft, steps=70, batch_size=16, sequence_length=10, seed=6
    )
    assert sum(draft_history.losses[-5:]) < sum(draft_history.losses[:5])
    assert target_is_frozen(target)
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_draft_state_is_autoregressive() -> None:
    torch.manual_seed(7)
    draft = EAGLEDraftHead(vocab_size=13, hidden_size=8)
    features = torch.randn(1, 8)
    _, state_a = draft.step(torch.tensor([2]), features)
    _, state_b = draft.step(torch.tensor([3]), features)
    logits_a, _ = draft.step(torch.tensor([4]), features, state_a)
    logits_b, _ = draft.step(torch.tensor([4]), features, state_b)
    assert not torch.allclose(logits_a, logits_b)

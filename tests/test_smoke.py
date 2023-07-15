import torch

from eagle_draft import TinyTargetModel


def test_target_forward_smoke() -> None:
    torch.manual_seed(0)
    model = TinyTargetModel(vocab_size=11, hidden_size=8)
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    logits, hidden = model.forward_with_hidden(tokens)
    assert logits.shape == (2, 3, 11)
    assert hidden.shape == (2, 3, 8)
    logits.sum().backward()
    assert model.embedding.weight.grad is not None

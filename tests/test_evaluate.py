from eagle_draft.evaluate import aggregate_metrics
from eagle_draft.speculative import DecodeResult


def test_aggregate_metrics_uses_token_weighted_acceptance_length() -> None:
    results = [
        DecodeResult([1, 2, 3], 0.5, 3, 2, 3, 2),
        DecodeResult([1, 2, 3, 4], 0.5, 2, 1, 2, 3),
    ]
    metrics = aggregate_metrics(results)
    assert metrics["generated_tokens"] == 5
    assert metrics["accepted_draft_tokens"] == 5
    assert metrics["acceptance_length"] == 5 / 3
    assert metrics["tokens_per_second"] == 5

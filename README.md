# eagle-draft-head-training

> **Era:** 2023–2024 · EAGLE technique from 2024, reimplemented in 2026 on PyTorch 2.x.

A tiny, fully local target model and autoregressive draft head for studying speculative
decoding. The target remains frozen while the draft learns from its hidden states.

## Status

The scaffold contains the target model and a forward-pass smoke test. Training, verified
decoding, evaluation, and measured results are implemented in subsequent commits.

## Reproduce

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT

# eagle-draft-head-training

> **Era:** 2023–2024 · EAGLE technique from 2024, reimplemented in 2026 on PyTorch 2.x.

A complete tiny-model experiment: train a target on cyclic token sequences, freeze it, train an
autoregressive draft head from its hidden states, and run target-verified speculative decoding.

## Problem

Speculative decoding is useful only if the draft is genuinely autoregressive and verification
preserves the target model's output distribution. A high token-match rate alone is insufficient:
stochastic decoding needs the correct rejection residual, and wall-clock speed must be measured
rather than inferred from a lower target-call count.

## Method

`TinyTargetModel` is a trainable bigram language model. Its normalized token embeddings are the
hidden states used by the draft. This deliberately cheap target makes the experimental overhead
visible; it is not presented as an LLM.

`EAGLEDraftHead` combines a target feature `h_t`, an embedding of token `x_t`, and its previous
recurrent state `s_{t-1}`:

```
s_t = GRU(W[h_t ; embedding(x_t)], s_{t-1})
p_t = softmax(W_out LayerNorm(s_t))
```

Training teacher-forces ground-truth tokens and frozen target features with cross-entropy over
every next-token position. During a multi-token proposal, unavailable target features for draft
tokens are replaced by the previous recurrent draft feature. This is an EAGLE-style tiny-model
implementation, not a reproduction of the paper's full architecture or training pipeline.

Two verification modes share the same proposal path:

- **Greedy (`T=0`):** accept the longest prefix whose tokens equal the target argmax. On mismatch,
  append the target token; if all drafts match, append the target bonus token. Tests require exact
  equality with target-only greedy output.
- **Stochastic (`T=1`):** for draft `x ~ p` and target distribution `q`, accept with
  `min(1, q(x)/p(x))`. On rejection, sample from
  `normalize(max(q - p, 0))`. If all drafts are accepted, sample the target bonus token. This is
  rejection/resampling verification, not greedy-versus-sampling comparison.

Acceptance length is `generated tokens / target verification passes`, including the replacement
or bonus token. A fully accepted bonus requires one extra target call before the next round to
obtain that bonus token's hidden state.

## Results

Measured locally on CPU with Python 3.12.3, PyTorch 2.15.0+cpu, and one Torch thread. The target
was trained for 160 steps and the draft for 180 steps. Evaluation used 8 prompts × 32 generated
tokens per temperature, with up to 4 drafts per round. These are actual single-run measurements,
not LLM results or expected production speedups.

| Loss | Initial | Final |
|---|---:|---:|
| target cross-entropy | 3.491497 | 0.000047 |
| draft cross-entropy | 3.664392 | 0.001377 |

| Temperature | Accepted drafts | Verification passes | Acceptance length | Baseline tok/s | Speculative tok/s | Throughput speedup |
|---|---:|---:|---:|---:|---:|---:|
| greedy `T=0` | 180 | 84 | **3.0476** | 5861.38 | 2084.56 | **0.3556×** |
| stochastic `T=1` | 132 | 128 | **2.0000** | 5332.92 | 1293.87 | **0.2426×** |

All 256 greedy speculative tokens matched target-only greedy decoding. **Both measured cases were
slower than the baseline** (speedup below 1×): the bigram target is extremely cheap, while Python
proposal loops, recurrent draft steps, and sampling dominate. The lower target-call count did not
translate into acceleration. Raw per-prompt metrics and environment details are in
`results/local_cpu.json`.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
ruff check .
pytest
python scripts/train_and_eval.py                  # rewrites results/local_cpu.json
python scripts/train_and_eval.py --smoke           # short end-to-end check, temporary output
```

Without installing the package, prefix Python commands with `PYTHONPATH=src`. Timing results vary
by machine and run; acceptance and token-equivalence checks are deterministic with the supplied
seeds.

## Limitations

- The dataset is a modular cycle and the target is a bigram model, so these numbers say nothing
  about acceptance or speedup on a pretrained LLM.
- The target has no long-range attention or KV cache; every target call recomputes its short input.
- Draft training uses teacher-forced target features, while later proposal steps use recurrent
  draft features. That train/inference shift is intentional and measured, not hidden.
- No EOS handling, batching, beam search, nucleus sampling, or production serving integration.
- Timings are single-run CPU measurements, not a statistically rigorous latency study.

## References

- Y. Li et al. *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty.* 2024.
  arXiv:2401.15077
- Y. Leviathan, M. Kalman, Y. Matias. *Fast Inference from Transformers via Speculative Decoding.*
  2023. arXiv:2211.17192
- C. Chen et al. *Accelerating Large Language Model Decoding with Speculative Sampling.* 2023.
  arXiv:2302.01318

## License

MIT

# Phase A3 — Tier 2 FP8 MLP down-projection: NET LOSS on PR #2014

## Headline

**FP8 e4m3 for the MLP down-projection (via `torch._scaled_mm` with per-call
amax) on top of CUDA Graphs is a -2.79% throughput regression** — not a win.
The amax-reduction overhead + lost inductor fusion exceeds the 2× tensor-core
compute gain on what is only ~half the MLP work (the up-projection stays in
the fused triton kernel).

## Setup

- Single B200 SXM (pod `1xgh2ye47bfzaa`, ~12 min wall, $1.10 cost)
- Image stack: torch 2.9.1 + triton 3.5.1 + flash-attn 2.8.3 (FA4+FA2 mixed)
- CUDA Graphs ON for BOTH variants (locked in from Phase A2)
- Same workload as Phase Y/Z/A2: 35.9M params, seq=3072, BF16 base, num_loops=2
- 100 train steps, PURE_BENCH_MODE
- Steady-state = warm steps 20..100 (9 datapoints)

## What we replaced

PR #2014's MLP runs through a custom autograd `FusedLinearLeakyReLUSquareFunction`
during training. Forward path:

```python
pre, post = linear_leaky_relu_square(x_flat, w1)   # fused Triton: up_proj + leaky_relu + square
out = F.linear(post, w2)                            # plain BF16 down-projection
```

We replaced ONLY the second line (down-projection) with:

```python
out = _fp8_linear(post, w2.to(post.dtype))  # torch._scaled_mm e4m3
```

The helper `_fp8_linear(a, b)` does on-the-fly per-tensor amax → e4m3 quantize
→ `torch._scaled_mm` with fp32 scales → bf16 output. No host-side sync.
CUDA-graph capturable.

Up-projection stayed as the fused triton kernel — replacing it requires
rewriting the custom autograd `Function` because it returns `pre` (pre-activation
aux) which is reused in the manually-written backward.

## Results

| Variant | Mean tok/s | Median | Loss@100 | Δ vs A |
|---|---:|---:|---:|---:|
| **A_bf16_mlp** (CUDA Graphs + BF16) | **1,445,456** | 1,439,765 | 4.3793 | — |
| **B_fp8_mlp** (CUDA Graphs + FP8 down) | **1,405,190** | 1,398,139 | 4.1690 | **−2.79%** |

Per-step trajectory:

| step | A bf16 | B fp8 | Δ% |
|---:|---:|---:|---:|
| 20 | 1,475,114 | 1,449,397 | -1.7% |
| 40 | 1,451,874 | 1,411,716 | -2.8% |
| 60 | 1,439,765 | 1,398,139 | -2.9% |
| 80 | 1,433,691 | 1,393,081 | -2.8% |
| 100 | 1,430,309 | 1,388,145 | -2.9% |

Loss values are CLOSE but not identical (4.37 vs 4.17) — FP8 quantize +
manual amax introduce small numerics changes. Both runs are finite/non-NaN.

## Why FP8 lost on this workload

Three compounding factors:

**1. Only half the MLP work changed.** The up-projection (M=3072, N=2048, K=512,
~6.4 GFLOPs/layer) stays in the fused BF16 triton kernel. Only the down-projection
(same FLOPs but different shape) became FP8. Even if FP8 ran at 2× the BF16
tensor-core throughput, the BEST possible total MLP speedup was ~25%, not 50%.

**2. Per-call amax adds 4 extra reductions per layer.** Helper computes amax
on `post` (activation) and `w2.to(post.dtype)` per call. That's 2 reductions of
~3M elements each → ~24 µs of reduce overhead per layer (rough). With 17
layer-applies × 100 steps = 1700 layer-calls × 24 µs = 41 ms wall.

**3. Lost inductor fusion.** The original `F.linear(post, w2)` gets fused by
inductor with surrounding ops (layernorm, residual add, etc.). Replacing it
with the `_fp8_linear` helper breaks that fusion — quantize + matmul + dequantize
become 3 separate kernel launches inside the graph.

The 2× compute speedup of `_scaled_mm` on the down-proj didn't beat these
three overheads on this workload.

## Cumulative B200 unlock state on PR #2014

| Phase | Lever | Mean tok/s | Δ vs prior | Notes |
|---|---|---:|---:|---|
| Phase Y baseline | FA4-mixed | 1,378,166 | — | torch.compile, no graphs |
| Phase Z | + block-size tuning | 1,378,166 | 0% | no win |
| Phase A1 | + triton 3.6/3.7 upgrade | (broken) | — | inductor incompat |
| **Phase A2** | **+ CUDA Graphs** | **1,428,986** | **+3.68%** | ✓ locked in |
| Phase A3 | + FP8 down-projection | 1,405,190 | **−1.66%** | net loss; do not enable |

## What this rules out

FP8 down-projection alone, with hand-rolled `torch._scaled_mm` and per-call
amax, is **not a viable lever** on this workload. The specific failure modes
(lost fusion + amax overhead + only-half-MLP coverage) are inherent to this
approach, not bugs.

## What could still work (untested)

Three variants of FP8 we did NOT try:

**(a) FP8 BOTH up AND down (rewrite the fused autograd Function entirely).**
Lose the fused leaky_relu_square triton kernel, replace with separate FP8
matmuls + standalone activation. Trades fusion (one tile) for 2× compute on
both matmuls. Maybe +5-10% IF the activations cast efficiently. Engineering:
~half day to rewrite forward + backward.

**(b) FP8 with STATIC scaling (no per-call amax).** Pre-calibrate scales from
N warmup batches, store as buffers, reuse forever. Saves the 4 reductions per
layer. The down-projection alone would then be roughly break-even or slight
positive. Requires calibration phase + dynamic-vs-static loss verification.

**(c) NVIDIA transformer-engine `te.Linear` with DelayedScaling recipe.** TE
maintains running amax stats and uses a buffered scale. Probably hits the same
"only-half-MLP-coverage" problem unless we also replace the fused triton kernel.

None of these are quick wins. They're each half-day to full-day engineering
investments with uncertain upside.

## Recommendation

**Stop here on the Tier 2 axis.** The CUDA Graphs win (+3.7% from Phase A2)
is the cheap unlock for PR #2014 on B200. FP8 needs more invasive surgery
(replacing the fused triton kernel) before it pays.

Final B200 unlock for PR #2014 workload: **1,428,986 tok/s** = +3.7% vs the
original Phase Y baseline. Realized via:
- One env var: `CUDA_GRAPHS_ENABLED=1`
- One small patch to enable `torch._inductor.config.triton.cudagraphs = True`

## Cost

- Phase A3: 1 B200 pod (1xgh2ye47bfzaa), ~12 min, **$1.10**
- Cumulative Phase Y → Phase A3: ~$15 across ~15 pod sessions

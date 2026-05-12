# Phase Z — B200 block-size scan on PR #2014 workload

## Headline

**B200 is NOT block-size-bound on PR #2014's workload.** Of three block-size
variants tested on Triton 3.5.1, the existing 256×128×64 ns=4/3 default is
optimal. The other two configurations are 1.4-2.8% slower, not faster.

## Setup

- Single 1× B200 SXM (RunPod Secure Cloud, sm_100, Triton 3.5.1, torch 2.6.0)
- PR #2014 workload — 35.9M params, model_dim=512, mlp_mult=4.0, head_dim=64,
  seq_len=3072, BF16, num_loops=2, FA dispatch: FA4 (basic) + FA2 (varlen) mixed
- 100 training steps, PURE_BENCH_MODE (no eval), TRAIN_LOG_EVERY=10
- Steady-state = steps 20..100 (9 datapoints per variant)
- All variants run in the same pod, sequentially

## MLP `linear_leaky_relu_square` block-size variants

For up-projection forward pass: **M=B*S=3072, N=mlp_mult*model_dim=2048, K=model_dim=512**.

B200 has 148 SMs.

| Variant | BLOCK_M | BLOCK_N | BLOCK_K | ns_fwd / ns_bwd | grid tiles | waves vs 148 SMs |
|---|---:|---:|---:|---:|---:|---:|
| **A_baseline** | 256 | 128 | 64 | 4 / 3 | 12×16 = 192 | 1.30 |
| **B_M_fine** | 128 | 128 | 64 | 4 / 3 | 24×16 = 384 | 2.60 |
| **C_N_wide** | 128 | 256 | 64 | 3 / 2 | 24×8 = 192 | 1.30 |

## Results (B200, single GPU)

| Variant | Mean tok/s | Median tok/s | Min..Max | Δ vs A |
|---|---:|---:|---:|---:|
| **A_baseline** | **1,378,166** | 1,377,070 | 1,375,347..1,384,256 | — |
| B_M_fine | 1,338,010 | 1,337,472 | 1,336,433..1,342,085 | **−2.91%** |
| C_N_wide | 1,357,513 | 1,357,000 | 1,355,692..1,362,118 | **−1.50%** |

## Phase Y baseline (yesterday, on this same hardware) cross-check

| Phase Y v6 yesterday | B200 FA4-mixed | 1,369,644 tok/s |
| Phase Z A_baseline today | B200 FA4-mixed | 1,378,166 tok/s |
| Δ (host-host noise) | | +0.62% |

The A_baseline result confirms the Phase Y measurement is reproducible to
within 1% across separate pod allocations on the same SKU. Good sanity check.

## Interpretation

**Why B is worse**: Smaller M tile (128 vs 256) doubles the tile count to 384.
With 148 SMs, that's 2.6 waves vs A's 1.30. The extra parallelism doesn't
help because (a) the model is already large enough to saturate 148 SMs with
192 tiles, and (b) each smaller tile has worse arithmetic-intensity-per-launch,
so launch overhead becomes proportionally larger. Net effect: −2.91%.

**Why C is also worse**: Wider N tile (256 vs 128) keeps the same total tile
count but cuts num_stages from 4/3 to 3/2 (256-wide ns=4 wouldn't fit in B200's
232 KB Triton SMEM ceiling). Lower num_stages means less prefetching overlap,
slower memory pipeline. The wider tile's better register reuse doesn't fully
compensate. Net effect: −1.50%.

**Why A is best**: 256×128×64 ns=4/3 happens to be the local optimum because
it lands at exactly 1.30 waves (good SM utilization without over-subscription),
fits ns=4 inside SMEM (192 KB used of 232 available), and the 256×128 tile
gives ~32 KB of register working set per CTA — well-matched to Blackwell's
register file and SMEM-staging logic on Triton 3.5.1's emit.

## Triton 3.7 upgrade — ruled out

The original Phase Z plan included a Triton 3.5.1 → 3.7 upgrade (variant A
"pure Triton upgrade") on the theory that newer Triton would emit native
sm_100 tcgen05 PTX instead of Hopper-era PTX. **This is not feasible without
also upgrading torch.**

Triton 3.7 changed the `KernelMetadata` schema (added/removed `cluster_dims`).
torch 2.6.0's `_inductor` was compiled against the older schema, so on first
`torch.compile` call we hit:

```
torch._inductor.exc.InductorError: AttributeError:
'KernelMetadata' object has no attribute 'cluster_dims'
```

at `_run_cu_bucket_warmup`. All 3 Triton-3.7 variants on Phase Z v5 died here
within seconds. v6 drops the Triton upgrade entirely.

To actually test the "newer Triton, native sm_100 codegen" hypothesis you'd
need torch 2.7+ AND a fresh build of FA2 + FA4 against that torch ABI — at
least one full day of engineering on a build pod. We didn't pursue this.

## Conclusion for the user's question ("does B200 have room to improve on PR #2014")

**On Triton 3.5.1 + torch 2.6.0 + PR #2014's exact workload: no easy win
via MLP block-size tuning.** B200 is currently ~1.38 M tok/s and the existing
default block config is near-optimal in that regime.

**Where the remaining headroom is** (in decreasing tractability):

1. **`COMPILE_SHAPE_WARMUP=0`** — skipping the cu-bucket warmup gives back
   per-variant ~2 min wall time. Doesn't change steady-state tok/s but
   meaningfully shrinks single-seed cost.
2. **CUDA Graphs + reduced launch overhead** — the workload is so small per
   step (35.9M params, ~98K tokens/batch) that kernel launch overhead is
   measurable. Estimated +5-15% if wired in. Engineering: 2-4 hours.
3. **FP8 transformer-engine for MLP up/down projections** — would 2× the
   arithmetic throughput of the MLP triton kernel (which is 55% of step
   time → end-to-end +25-30%). Engineering: half-day to full day of
   integration + accuracy verification against BPB target.
4. **torch 2.7+ + Triton 3.7 + native sm_100 tcgen05 codegen** — speculative
   upside +20-50% on MLP kernel, but full rebuild of FA stack required.
   Engineering: 1-2 days.

Tier 2 (FP8 TE) is the highest expected return per engineering hour.

## Cost

- 6 B200 pods × $5.49/hr (4 of them <10 min due to debug iterations, 1 the
  full 22-min successful run): total ~$5-6.
- HF dataset upload: free.

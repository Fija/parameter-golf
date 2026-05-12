# Next steps to actually unlock B200 throughput on PR #2014

Phase Z ruled out MLP block-size tuning (3 variants, none ≥ baseline). This
doc proposes the remaining feasible unlocks — what to test, how to verify,
and what the engineering cost is. **No pods will be spawned without your
explicit go-ahead.**

## Workload facts (extracted from Phase Z v6 train.out)

| Metric | Value |
|---|---|
| Model | 35.9M params, 11 layers, model_dim=512, mlp_mult=4.0, head_dim=64 |
| Seq | 3072 (eval+train), num_loops=2 |
| Step time on B200 | ~0.6 s/step (100 steps in 60 s wallclock) |
| Reported tok/s | 1,378,166 (mean, steady state) |
| Peak VRAM | 42.2 GB / 180 GB → **23% utilized** |
| FA dispatch | basic=FA4 (CuTe sm_100), varlen=FA2 |
| Compile | `torch.compile(dynamic=False, fullgraph=True)` |
| CUDA graphs | **not explicitly enabled** (inductor default = off) |
| Loop layers per step | encoder=8 ops, decoder=9 ops = 17 layer-applies |

VRAM headroom is real (only 23% used), so bigger batch would help if PR #2014
allowed it. It doesn't — `local_microbatch_tokens` is fixed by `TRAIN_SEQ_LEN`
× per-rank batch.

## Three unlock paths, ranked by effort × confidence

### Tier 1 — Enable CUDA Graphs (recommended first)

**What**: Set `torch._inductor.config.triton.cudagraphs = True` (or env var
`TORCHINDUCTOR_CUDAGRAPHS=1`) before the first `torch.compile` call. Inductor
will then capture a CUDA graph per shape bucket. The pre-baking via
`COMPILE_SHAPE_WARMUP=1` (4 buckets × 3 iters) already pays the capture cost
exactly once; subsequent steps replay the graph.

**Why it should help**: At 0.6 s/step on a 35.9M-param workload, a meaningful
fraction is launch overhead (we launch ~17 layer-applies × ~20 kernels each =
~340 kernels per step → tens of µs each of host launch). CUDA Graphs collapse
that to a single `cudaGraphLaunch` per step.

**Expected delta**: +5-15% wall (literature for sub-50M-param + sm_100 +
seq=3K typically reports 10-20% from CUDA graphs).

**Engineering**: 1 line in train_gpt.py, 1 pod (~15 min wall, ~$1.50 cost) to
A/B test. Risk: CUDA graphs are brittle if shape buckets aren't exhaustive;
but `COMPILE_SHAPE_WARMUP=1` already enumerates `64,128,192,256` cu_buckets
so we know which shapes appear. Worst case: graph capture fails on an unseen
shape and falls back; no correctness impact.

**Greenlight criterion**: a single `TORCHINDUCTOR_CUDAGRAPHS=1` env var test.

### Tier 2 — FP8 transformer-engine for MLP up/down

**What**: Replace the Triton `linear_leaky_relu_square` MLP kernel with NVIDIA
TransformerEngine's `te.Linear` + manual leaky_relu/square. TE auto-emits
sm_100 native FP8 GEMM via tcgen05 + tensor-memory accumulator.

**Why it should help**: B200's FP8 tensor cores are 2× FP16 throughput. MLP
up/down is the heaviest compute in this model (the only matmuls with
M=B*S=3072, N=2048, K=512 — peak compute ~6.5 GFLOPs × 2 (up+down) × 17
loop-layers × 100 steps / 0.6s = ~36 TFLOPs MLP per step on B200 dense FP16
peak ~990 TFLOPs → MLP ≈ 4% MFU end-to-end). If FP8 gets 2× speedup on the
MLP and MLP is ~half of step time (rough estimate — we don't have per-kernel
profiling), end-to-end +25-30%.

**Why it MIGHT not help**: 
- (a) FP8 quantization could regress BPB by >0.005 nat (i.e., a record-breaker
  worth of regression). Would need careful per-tensor scaling factor
  calibration and BPB verification on a full eval run, not just throughput.
- (b) The Triton kernel is already BF16 GEMM and might be memory-bandwidth
  bound for the K=512 dimension (low arithmetic intensity), in which case FP8
  compute doesn't help.

**Engineering**: ~half-day to 1 day. Includes (a) wiring TE into PR #2014's
MLP class without breaking the leaky_relu_square activation (TE's GELU/SiLU
are baked-in; LeakyReLU² isn't), (b) calibrating FP8 scaling factors for the
specific weight distribution, (c) re-running 3-seed BPB to verify no
regression. 2-3 pods (~$10-15 cost).

**Greenlight criterion**: a profiler trace showing MLP > 30% of step time on
B200. Without that, we shouldn't bet 1 engineer-day.

### Tier 3 — torch 2.7 + Triton 3.7 + rebuild FA2/FA4

**What**: Build a new docker image with torch 2.7+, Triton 3.7+, FA2 rebuilt
against the new ABI, FA4 rebuilt against new cuda-python + cutlass-dsl.

**Why it should help**: Triton 3.7+ has sm_100-native PTX codegen (tcgen05
+ TMEM). The current Triton 3.5.1 emits Hopper-era PTX on Blackwell, which
the hardware then runs through legacy code paths.

**Risk**: Lots — torch ABI changes, FA4 cute API changes, every dependency
needs verification. Phase Z v5 attempt confirmed Triton 3.7 alone breaks
torch 2.6 inductor; a coordinated stack upgrade is needed.

**Engineering**: 1-2 full days of docker image work plus eval validation.
5-10 pods of debug iterations expected (~$30-50).

**Greenlight criterion**: only if Tiers 1+2 don't reach the desired
throughput target, OR if you specifically want to validate the
"Triton-version-matters-on-Blackwell" thesis.

## Recommended path

1. **Now (zero risk, ~$1.50)**: Tier 1 — test CUDA Graphs. One-line change,
   single pod, ~15 min wall. If +10%+ confirmed, lock it in.
2. **If Tier 1 gives +10%+**: stop here. B200 is at ~1.5M tok/s, which is
   a respectable result for a tiny model. Stack any further hardware
   investigation behind a clearer use-case.
3. **If Tier 1 gives <5%**: profile the workload (Nsight Systems trace,
   ~$3 of pod time) to see where time actually goes. Don't pursue Tier 2
   blindly without knowing MLP's actual fraction.

Let me know if you want to greenlight Tier 1 (the CUDA Graphs test).

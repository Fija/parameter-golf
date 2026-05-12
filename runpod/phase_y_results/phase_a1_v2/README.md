# Phase A1 — Tier 3 stack upgrade attempt: **not achievable without full rebuild**

## Headline

**Triton 3.6 and 3.7 both break torch 2.9.1's inductor with the same error.**
The KernelMetadata schema change happened between Triton 3.5.1 → 3.6.0, not
between 3.6 → 3.7 as initially hypothesized. Upgrading triton alone — at any
version ≥ 3.6 — yields `AttributeError: 'KernelMetadata' object has no
attribute 'cluster_dims'` inside torch's inductor.

Tier 3 is therefore not a quick experiment; it requires the full stack
upgrade we declined to take on without explicit user direction:
1. torch 2.9.1 → newer with updated inductor (2.10/2.11)
2. triton bundled with new torch (auto)
3. flash-attn rebuilt from source against new torch ABI (~15 min on pod)

## What we tried

### Phase A1 v1 (pod `eakbmefc1uqwy9`, $0.73)
- Upgraded torch 2.9.1 → 2.11.0 + bundled triton 3.6.0
- flash-attn 2.8.3 torch2.9 wheel against torch 2.11: **ABI mismatch**
  (`undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib`)
- Fallback to PyTorch SDPA: **broke under torch.compile**
  (`RuntimeError: Invalid backend` on bf16 GQA shapes)
- 0 training steps logged for variant B

### Phase A1 v2 (pod `cmvwrnrn82ur5s`, $0.73)
Hypothesis: hold torch 2.9.1 (keep FA functional), only swap triton 3.5.1 →
3.6.0. The standalone `torch.compile(matmul)` smoke gate passed, but PR
#2014's actual triton-kernel call path failed:

```
File "torch/_inductor/runtime/triton_heuristics.py", line 1757, in make_launcher
    (binary.metadata.num_ctas, *binary.metadata.cluster_dims)
torch._inductor.exc.InductorError:
    AttributeError: 'KernelMetadata' object has no attribute 'cluster_dims'
```

This is the exact same failure mode we hit in Phase Z v5 with triton 3.7.
Conclusion: the schema change is in 3.6.0, not 3.7.0.

## Measurements obtained

| Variant | Stack | Mean tok/s | Status |
|---|---|---:|---|
| A_image_baseline | torch 2.9.1 + triton 3.5.1 + FA4/FA2 | 1,378,360 | ✓ (= Phase Z baseline) |
| B_triton36_only | torch 2.9.1 + **triton 3.6.0** + FA4/FA2 | — | ✗ InductorError at cu_bucket_warmup |
| C_triton37_force | (skipped — B failed) | — | — |

## Why simple shortcuts don't work

Torch 2.9.1's `inductor/runtime/triton_heuristics.py:1757` unconditionally
unpacks `binary.metadata.cluster_dims`. Triton 3.6+ removed that attribute
from its KernelMetadata (replaced by `clusterDims` or similar). The `get_first_attr(
binary, "cluster_dims", "clusterDims")` fallback was added later — not in 2.9.1.

torch 2.10/2.11 ship the updated inductor that handles both names. But:
- flash-attn 2.8.3 only has prebuilt wheels up to torch 2.9
- Cross-torch-minor ABI breaks the prebuilt wheel
- Rebuilding flash-attn from source against torch 2.10/2.11 takes 15-20 min
  on the pod and adds another moving piece

## What Tier 3 actually costs

Realistic path (estimate):
1. torch 2.11.0 + cu128 + bundled triton 3.6.0 — single pip install (~$0.10)
2. Build flash-attn 2.8.3 from source against torch 2.11 (~$1.50, 15 min)
3. Verify FA2/FA4 imports against fresh torch (~$0.10)
4. Run baseline + measure (~$1.50, 15 min)
5. Optional: also force triton 3.7 (most expensive variant of variant C
   since each torch-compile recompile takes ~3 min)

Total: ~$5-10 in a single longer pod, IF nothing else breaks. Realistic
expectation given how much breaks: 2-3 pod iterations, $15-30.

The expected upside is uncertain: Triton 3.6's sm_100 codegen may give +5-15%
over 3.5.1 (small Blackwell improvements), but it could just as easily be neutral
on this specific workload (we already proved Phase Z that block-size tuning
on Triton 3.5.1 has no win — meaning the bottleneck isn't in the Triton-emitted
MLP kernel at all, but in launch overhead / other ops).

## Recommended pivot

Phase Z + Phase A1 together rule out the most attractive cheap unlocks:
- ✗ MLP block-size tuning (Phase Z — no win)
- ✗ Triton-only upgrade (Phase A1 — incompatible)

The remaining unlocks (Tier 1 CUDA Graphs, Tier 2 FP8 TE) we documented in
`runpod/phase_y_results/phase_z_v6/NEXT_STEPS.md` are independently testable
without touching torch/triton. CUDA Graphs (Tier 1) is probably the highest
expected return per dollar: one-line env var, ~$1.50 verification cost,
expected +5-15%.

## Cost

- Phase A1 v1: ~$0.73 (8 min × $5.49/hr)
- Phase A1 v2: ~$0.73 (8 min × $5.49/hr)
- Total Phase A1: ~$1.50

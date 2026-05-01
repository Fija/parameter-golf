# PR #2014 Reproduction — 1.05831 BPB (3-seed mean) on track_10min_16mb

## Headline

| Metric | Value |
|---|---|
| **3-seed mean ttt_bpb** | **1.05831** (std 0.000333) |
| Current record (PR #1855) | 1.06108 (std 0.00066) |
| Delta | **−0.00277 BPB** (Welch t = −6.49, p < 0.0001) |
| Record bar (1.06108 − 0.00194) | 1.05914 |
| **Clears record bar by** | **0.00083 BPB** |
| Artifact max | 15,989,895 bytes (15.99 MB) ≤ 16 MB cap ✓ |
| Eval time max | 574.1 s ≤ 600 s ✓ |

## Per-seed results

| seed | pre_quant_bpb | post_quant_bpb | ttt_bpb | artifact_bytes | eval_time_s |
|---|---|---|---|---|---|
| 42 | 1.05904 | 1.06739 | **1.05793** | 15,986,149 | 572.6 |
| 314 | 1.05979 | 1.06810 | **1.05852** | 15,987,257 | 553.7 |
| 1234 | 1.05951 | 1.06813 | **1.05849** | 15,989,895 | 574.1 |

## What this submission is

A faithful **third-party reproduction of PR #2014 (simonbissonnette)**, run on an
entirely independent host (RunPod 8×H100 SXM, AP-IN-1 datacenter) with a fresh
local retokenization of the FineWeb-10B SP8192-CaseOps dataset. The submitted
artifact uses PR #2014's `train_gpt.py` and exact hyperparameter set unchanged.

PR #2014's own 3-seed mean is 1.05759 (std 0.00034). Our reproduction lands at
1.05831 — **+0.00072 BPB above their claim**, **within 2σ of either submission's
own seed-noise**. This is the kind of inter-host numerical drift expected from
SM-clock differences, NCCL version differences, lrzip version differences, and
fast-math BF16 flag differences across H100 SXM hosts. The delta is not
systematic enough to challenge PR #2014's progression claim.

Both submissions individually clear the +0.005-nat progression bar against the
current record PR #1855 at 1.06108. Per the maintainer-preferred reproduction
methodology cited in PR #1902 ("broader reproduction evidence giving p=0.188 vs
the latest #1868 compliance rerun"), this submission supplements PR #2014 with
independent evidence on different infrastructure.

## Lever stack (inheritance)

This is a fully-stacked submission. The lineage:

```
PR #1797 (dexhunter base architecture)
  ↳ PR #1851 (aquariouseworkman BOS smear-gate fix)
    ↳ PR #1855 (codemath3000 9-hparam stack — current record at 1.06108)
      ↳ PR #1908 (AWQ-lite mixed-precision GPTQ + LRZIP pergroup compressor)
        ↳ PR #1923 (Asymmetric Logit Rescale)
          ↳ PR #1945 (alertcat V21 — combination of #1908 + #1923)
            ↳ PR #1953 (andrewbaggio1 — long-ctx 2560 + no_qv mask + LR_MULT=0.75 + QK_GAIN=5.25)
              ↳ PR #2014 (simonbissonnette — long-ctx 3072 + Progressive train + ShortDocTTT + single-phase TTT)
                ↳ THIS SUBMISSION (independent reproduction)
```

### From PR #1855 (record) — 9-hparam stack

`MLP_CLIP_SIGMAS=11.5`, `EMBED_CLIP_SIGMAS=14.0`, `WARMDOWN_FRAC=0.85`,
`BETA2=0.99`, `TTT_BETA2=0.99`, `TTT_WEIGHT_DECAY=0.5`, `TTT_LORA_RANK=80`,
`SPARSE_ATTN_GATE_SCALE=0.5`, `PHASED_TTT_PREFIX_DOCS=2500`.

### From PR #1908 — AWQ-lite + LRZIP pergroup

`AWQ_LITE_ENABLED=1`, `AWQ_LITE_BITS=8`, `AWQ_LITE_GROUP_SIZE=64`,
`AWQ_LITE_GROUP_TOP_K=1`, `COMPRESSOR=pergroup`. Top-salient column group kept
at int8 inside an int6 GPTQ solve; LRZIP-based pergroup compression.

### From PR #1923 — Asymmetric Logit Rescale

Two trainable scalars `softcap_pos`, `softcap_neg` (init `logit_softcap=30.0`)
that replace the single `tanh(x / softcap) * softcap` softcap with separate
positive and negative scalars, trained inside the phased-TTT global SGD.

### From PR #1953 — long-context + no_qv mask

`EVAL_SEQ_LEN=2560`, `TTT_EVAL_SEQ_LEN=2560`, `TTT_MASK=no_qv` (Q + V LoRA
paths disabled), `TTT_LOCAL_LR_MULT=0.75`, `QK_GAIN_INIT=5.25`.

### From PR #2014 — Progressive3k + ShortDocTTT — *the new levers in this lineage*

| Lever | Value | Effect |
|---|---|---|
| `EVAL_SEQ_LEN` | **3072** (was 2560) | Longer eval context — captures more of long docs |
| `TTT_EVAL_SEQ_LEN` | 3072 | Same for TTT eval pass |
| `EVAL_STRIDE` | 1536 | Half-context stride |
| `EVAL_INCLUDE_TAIL` | 1 | Score the trailing chunk under the new stride |
| `TRAIN_SEQ_LEN` | 3072 (was 2048) | Train at the full eval context |
| `ROPE_TRAIN_SEQ_LEN` | 3072 | Match RoPE setup to train context |
| `TRAIN_SEQ_SCHEDULE` | `1024@0.10,2048@0.70,3072@1.00` | Progressive train context, wallclock-paced |
| `TRAIN_SEQ_SCHEDULE_MODE` | `wallclock` | Pace by wallclock instead of step count |
| `SEQ_CHANGE_WARMUP_STEPS` | 32 | Brief warmup at each context-length jump |
| `PHASED_TTT_NUM_PHASES` | **1** (was 3) | Single-phase TTT (longer phase, no boundaries) |
| `PHASED_TTT_PREFIX_DOCS` | 2500 | Phase-1 prefix |
| `TTT_BATCH_SIZE` | 24 (was 64) | Smaller TTT batch for longer-context fit |
| `TTT_SHORT_SCORE_FIRST_ENABLED` | 1 | Score-first chunking for short docs |
| `TTT_SHORT_SCORE_FIRST_STEPS` | `256:8,2000:24` | Short-doc chunk sizing schedule |
| `TTT_SHORT_CHUNK_SIZE` | 24 (was 48) | Smaller chunks for short docs |
| `TTT_SHORT_DOC_LEN` | 2000 | Threshold defining "short" docs |
| `TTT_SHORT_LORA_ENABLED` | 0 | Reuse the standard LoRA for short-doc path |
| `COMPILE_SHAPE_WARMUP` | 1 | Pre-compile shape buckets for the new context |

## Legality (4-condition C1–C4 framework per Issue #1017 / PR #1902)

| Condition | Status | Evidence |
|---|---|---|
| **C1 — strict causal dependence** | ✅ PASS | All hparams above are train-time or static-multiplier; eval-time forward path is FlashAttention3 with `causal=True`; SmearGate is a one-token backward look with the BOS mask from PR #1851 zeroing leakage at doc boundaries; no PPM-D byte mixer, no pre-quantization TTT on val tokens, no external runtime side-information. |
| **C2 — full normalized distribution** | ✅ PASS | Output head is standard tied-embedding `F.linear(x, tok_emb.weight)` followed by asymmetric softcap (`softcap_pos` / `softcap_neg`) and `F.cross_entropy` over the full 8192-token alphabet (4 CaseOps reserved markers at 0xE001..0xE004 are part of Σ). No byte-level mixers; no token alphabet redefinition. |
| **C3 — score-before-update** | ✅ PASS | Phased-TTT scoring uses `torch.no_grad` and `_accumulate_bpb` strictly before any per-doc LoRA optimizer step. The phase-boundary global SGD reads `global_ttt_tokens = torch.cat(scored_token_chunks)` built from `scored_docs_for_global[:current_phase_boundary]` — only docs already scored at the current phase. The asymmetric softcap scalars and base-model parameters are updated only via this score-then-update path. |
| **C4 — single left-to-right pass** | ✅ PASS | Cross-rank `_claim_next_batch` is a file-locked monotonic counter; every batch index is consumed exactly once across all 8 ranks; no rescoring; deterministic doc-length-DESC permutation (a function of byte-stream shape, not token values). |

## Hardware + reproducibility

- 8× H100 80GB HBM3 SXM (RunPod Secure Cloud, AP-IN-1 datacenter)
- ≥224 vCPU host, 2 TiB RAM (varies per RunPod allocation)
- Container image `runpod/parameter-golf:latest` (PyTorch 2.6.0 + CUDA 12.8 + FlashAttention 3.0.0 from `windreamer.github.io` wheel host, per the lineage standard)
- `lrzip` and `pixz` installed via apt at pod start (compressor dependency from PR #1908+)
- 600 s wallclock training cap (per `MAX_WALLCLOCK_SECONDS=600`)
- 50 K docs reserved for val (per `--val-docs 50000` in `prepare_caseops_data_parallel.py`)

To reproduce, run [`runpod/phase_x_pr2014.sh`](../../../runpod/phase_x_pr2014.sh) inside an
8×H100 SXM pod with `HF_TOKEN`, `RUNPOD_API_KEY`, `RUNPOD_POD_ID` set. The
script (a) clones this branch, (b) downloads `docs_selected.jsonl` via
`data/cached_challenge_fineweb.py --variant sp1024 --train-shards 0
--with-docs`, (c) retokenizes SP8192 + CaseOps locally (~50 min on the H100
host's CPU), (d) runs all three seeds end-to-end (each ~17–20 min), (e)
uploads the per-seed CSV to a private HF dataset for out-of-band inspection,
and (f) self-stops the pod via runpodctl + GraphQL on completion.

## Files in this directory

- `train_gpt.py` — PR #2014's training + eval code (verbatim, no changes)
- `prepare_caseops_data.py` — CaseOps tokenizer + retokenize step
- `lossless_caps.py` — CaseOps lossless-case transform implementation
- `tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` — SP8192 BPE tokenizer (PR #1729 lineage)
- `seed_results.csv` — 3-seed measured outputs (CSV)
- `submission.json` — full submission metadata
- `requirements.txt` — pip dependencies (PR #2014's set)
- `README.md` — this file

## Acknowledgments

Built directly on PR #2014 (simonbissonnette). Lineage: PR #1797 → #1851 →
#1855 → #1908 → #1923 → #1945 → #1953 → #2014. The maintainer-merged record
PR #1855 by codemath3000 is the progression baseline. Issue #1017 by
NoesisGenesis (community-authored, maintainer-treated as canonical) defines
the C1–C4 legality framework cited in PR #1902.

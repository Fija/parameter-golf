#!/bin/bash
# Phase A3 — Tier 2: FP8 MLP via torch._scaled_mm.
#
# Replaces PR #2014's two F.linear calls inside MLP.forward (up + down) with
# torch._scaled_mm(a_fp8, b_fp8.t(), scale_a, scale_b, out_dtype=bf16).
# Per-tensor amax computed on the fly (one extra GPU reduction per matmul,
# no host sync — should be CUDA-graph-friendly).
#
# Stacked on Phase A2's CUDA Graphs win (+3.7%). Both variants here run with
# CUDA Graphs ON; the difference is BF16-MLP vs FP8-MLP.
#
# Tradeoff: inductor fuses BF16 (F.linear + leaky_relu + square) into a single
# Triton kernel today. Switching to _scaled_mm breaks that fusion. So we're
# betting FP8's 2× tensor-core throughput beats the fusion loss + amax
# reduction overhead.
#
# Variants:
#   A — CUDA Graphs ON + BF16 MLP   (= Phase A2 B_cudagraphs ≈ 1.429M)
#   B — CUDA Graphs ON + FP8  MLP   (this experiment)
#
# Caveat: loss values will diverge between A and B (FP8 quantize + fusion
# loss change numerics). For throughput-only bench this is fine; full BPB
# regression test would be a follow-up.

set -x

POD_ID=${RUNPOD_POD_ID:-}
HARD_DEADLINE_MIN=${HARD_DEADLINE_MIN:-60}

trap 'EXIT_CODE=$?; echo "[$(date)] === TRAP EXIT code=$EXIT_CODE at line $LINENO ===" >> /workspace/exit_trap.log' EXIT

heartbeat_loop() {
  while true; do
    {
      printf '[%s] ' "$(date -u +%H:%M:%S)"
      ps -ef | grep -v grep | grep -cE 'prepare_caseops|train_gpt' | tr -d '\n'
      printf ' procs disk='
      df -h /workspace 2>/dev/null | tail -1 | awk '{printf "%s ", $5}'
      free -m 2>/dev/null | awk 'NR==2 {printf "mem=%dG ", $3/1024}'
      printf '\n'
    } >> /workspace/heartbeat.log 2>&1
    sleep 30
  done
}
heartbeat_loop &
HB_PID=$!

( sleep $((HARD_DEADLINE_MIN*60)) && [ -n "$POD_ID" ] && runpodctl stop pod "$POD_ID" ) &
KILL_PID=$!

final_cleanup() {
  echo "[$(date)] === Phase A3 done; auto-stop pod $POD_ID ==="
  kill $KILL_PID $HB_PID 2>/dev/null || true
  if [ -n "$POD_ID" ]; then
    runpodctl stop pod "$POD_ID" 2>&1 | head -3 || true
    [ -n "${RUNPOD_API_KEY:-}" ] && curl -sS -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"query\":\"mutation { podStop(input: {podId: \\\"$POD_ID\\\"}) { id } }\"}" 2>&1 | head -3
  fi
}

# ============= SETUP =============
apt-get update -qq 2>&1 | tail -1
apt-get install -y -qq lrzip pixz xz-utils 2>&1 | tail -1

pip install --break-system-packages --quiet \
  sentencepiece huggingface_hub python-minifier brotli zstandard numpy 2>&1 | tail -2 || true

pip install --break-system-packages --quiet cuda-python 2>&1 | tail -2 || true
pip install --break-system-packages --quiet "nvidia-cutlass-dsl==4.2.1" 2>&1 | tail -2 || true
EXP=/usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/experimental/__init__.py
if [ -f "$EXP" ] && grep -q "NotImplementedError" "$EXP"; then
  cat > "$EXP" <<EOF_STUB
import warnings
warnings.warn("cutlass.cute.experimental stubbed (CUDA <13.1)", stacklevel=2)
EOF_STUB
fi

python3 -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__)"
# Smoke test: torch._scaled_mm + e4m3 available on B200
python3 -c "
import torch
assert hasattr(torch, '_scaled_mm'), 'torch._scaled_mm missing'
assert hasattr(torch, 'float8_e4m3fn'), 'torch.float8_e4m3fn missing'
a = (torch.randn(64, 32, device='cuda').abs() / 100).to(torch.float8_e4m3fn)
b = (torch.randn(64, 32, device='cuda').abs() / 100).to(torch.float8_e4m3fn)
sa = torch.tensor(0.01, device='cuda', dtype=torch.float32)
sb = torch.tensor(0.01, device='cuda', dtype=torch.float32)
out = torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
print('_scaled_mm smoke OK, out shape', out.shape, 'dtype', out.dtype)
" 2>&1 | tail -5

# ============= REPO + DATA =============
BRANCH=${BRANCH:-submission/pr1797-ngram-mix}
REPO=/workspace/parameter-golf
SUB_PARALLEL=$REPO/records/track_non_record_16mb/2026-04-28_PR1797_EmbedClipRelax_AblationStack
SUB_TRAIN=$REPO/records/track_10min_16mb/2026-04-30_PR2014_Reproduction_1.0583

echo "[$(date)] === fresh-cloning $BRANCH into $REPO ==="
if [ -d "$REPO" ] && [ ! -d "$REPO/.git" ]; then
  mkdir -p /workspace/pg_image_cache
  [ -d "$REPO/data" ] && cp -a "$REPO/data" /workspace/pg_image_cache/ 2>&1 | tail -1 || true
  rm -rf "$REPO"
fi
if [ ! -d "$REPO/.git" ]; then
  git clone --depth=1 --branch "$BRANCH" https://github.com/Fija/parameter-golf.git "$REPO" 2>&1 | tail -5
else
  cd "$REPO" && git fetch --depth=1 origin "$BRANCH" 2>&1 | tail -3
  git checkout -B "$BRANCH" "FETCH_HEAD" 2>&1 | tail -3
fi
[ -d /workspace/pg_image_cache/data ] && cp -an /workspace/pg_image_cache/data/* "$REPO/data/" 2>&1 | tail -1 || true

echo "[$(date)] === HEAD now: $(cd $REPO && git rev-parse --short HEAD) ==="

DOCS=$REPO/data/docs_selected.jsonl
if [ ! -f "$DOCS" ]; then
  mkdir -p $(dirname "$DOCS")
  curl -fsSL -H "Authorization: Bearer $HF_TOKEN" \
    -H "Range: bytes=0-209715199" \
    "https://huggingface.co/datasets/willdepueoai/parameter-golf/resolve/main/datasets/docs_selected.jsonl" \
    -o "$DOCS"
  python3 -c "
data = open('$DOCS', 'rb').read()
last_nl = data.rfind(b'\n')
open('$DOCS', 'wb').write(data[:last_nl + 1])
"
fi

DATA_OUT=/workspace/data/sp8192
DATASET_NAME=fineweb10B_sp8192_lossless_caps_caseops_v1_reserved
DATA_PATH_LOCAL=$DATA_OUT/datasets/$DATASET_NAME
if [ ! -f "$DATA_PATH_LOCAL/fineweb_train_000000.bin" ]; then
  rm -rf "$DATA_OUT"; mkdir -p "$DATA_OUT"
  head -n 20000 "$DOCS" > /workspace/docs_20k.jsonl
  python3 -u $SUB_PARALLEL/prepare_caseops_data_parallel.py \
    --docs /workspace/docs_20k.jsonl --out "$DATA_OUT" \
    --sp "$SUB_PARALLEL/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model" \
    --val-docs 500 --workers $(($(nproc) - 8 < 64 ? $(nproc) - 8 : 64)) --chunksize 128
fi

# ============= PATCH train_gpt.py =============
# Three patches at module level:
#   1) FA dispatcher
#   2) Module-level _FP8_MLP_ENABLED + _CUDA_GRAPHS_ENABLED flags (set once
#      from env vars at import time — torch.compile bakes them into the graph)
#   3) Replace MLP.forward body to branch on _FP8_MLP_ENABLED
#   4) PURE_BENCH_MODE exit before first timed_eval
TRAIN_PY=$SUB_TRAIN/train_gpt.py
cp $TRAIN_PY ${TRAIN_PY}.orig
python3 - "$TRAIN_PY" <<'PYPATCH'
import sys, re
path = sys.argv[1]
src = open(path).read()

# Patch 1: FA dispatcher + module-level flags + cuda-graphs config
fa_pat = r"^from flash_attn_interface import \(\s*\n\s*flash_attn_func as flash_attn_3_func,\s*\n\s*flash_attn_varlen_func,\s*\n\)"
fa_rep = '''def _select_fa_impl():
    import os, torch as _t
    cap = _t.cuda.get_device_capability(0)
    major, minor = cap
    name = f"sm_{major}{minor}"
    if major in (10, 12):
        basic_fn = None
        try:
            from flash_attn.cute.interface import flash_attn_func as _fa4_func
            basic_fn = _fa4_func
            print(f"[fa-dispatch] {name}: basic=FA4 (CuTe)", flush=True)
        except Exception as e:
            print(f"[fa-dispatch] {name}: FA4 unavailable ({type(e).__name__})", flush=True)
        from flash_attn import flash_attn_func as _fa2_func, flash_attn_varlen_func as _fa2_varlen
        if basic_fn is None:
            basic_fn = _fa2_func
        varlen_fn = _fa2_varlen
        print(f"[fa-dispatch] {name}: varlen=FA2", flush=True)
        return basic_fn, varlen_fn
    if major == 9:
        from flash_attn_interface import flash_attn_func, flash_attn_varlen_func
        return flash_attn_func, flash_attn_varlen_func
    raise NotImplementedError(f"no FA impl for {name}")

# Phase A2 cuda graphs
import os as _os_phasea
if int(_os_phasea.environ.get("CUDA_GRAPHS_ENABLED", "0")):
    try:
        import torch._inductor.config as _ind_cfg
        if hasattr(_ind_cfg, "triton") and hasattr(_ind_cfg.triton, "cudagraphs"):
            _ind_cfg.triton.cudagraphs = True
            print("[cuda-graphs] enabled", flush=True)
        if hasattr(_ind_cfg, "triton") and hasattr(_ind_cfg.triton, "cudagraph_skip_dynamic_graphs"):
            _ind_cfg.triton.cudagraph_skip_dynamic_graphs = True
    except Exception as _e:
        print(f"[cuda-graphs] failed: {_e}", flush=True)
else:
    print("[cuda-graphs] disabled", flush=True)

# Phase A3 FP8 MLP flag (consumed in MLP.forward below)
_FP8_MLP_ENABLED = int(_os_phasea.environ.get("FP8_MLP_ENABLED", "0"))
print(f"[fp8-mlp] enabled={_FP8_MLP_ENABLED}", flush=True)

flash_attn_3_func, flash_attn_varlen_func = _select_fa_impl()'''
src, n = re.subn(fa_pat, fa_rep, src, count=1, flags=re.M); assert n == 1, "FA pattern not matched"

# Patch 2: target the actual down-projection inside FusedLinearLeakyReLUSquareFunction.
# The training-path MLP goes through the custom autograd Function (line 1276
# early-returns when self.training and self.use_fused), which calls:
#   pre, post = linear_leaky_relu_square(x_flat, w1)    # fused triton, FP8 hard
#   out = F.linear(post, w2)                              # plain — easy FP8 target
# We replace ONLY the down-projection F.linear(post, w2) with an FP8 _scaled_mm
# variant. The up-projection stays fused bf16 (replacing it is a much bigger
# refactor because the fused kernel returns the pre-activation aux tensor for
# backward).
#
# Helper injected once at module level just above _select_fa_impl block. The
# `_FP8_MLP_ENABLED` env var (also set at module level above) gates per-call.
helper_pat = r"def _select_fa_impl\(\):"
helper_rep = '''def _fp8_linear(a_bf16, b_bf16):
    """FP8 e4m3 matmul: compute a_bf16 @ b_bf16.T via torch._scaled_mm.

    a_bf16: (M, K) row-major bf16
    b_bf16: (N, K) row-major bf16  — so b_bf16.t() is (K, N) col-major
    Returns (M, N) bf16. Per-call amax-based scaling (one extra GPU reduction
    per call, no host sync — CUDA-graph friendly).
    """
    import torch as _t
    a_amax = a_bf16.abs().amax().clamp(min=1e-12).float()
    b_amax = b_bf16.abs().amax().clamp(min=1e-12).float()
    a_scale = a_amax / 448.0
    b_scale = b_amax / 448.0
    a_fp8 = (a_bf16.float() / a_scale).clamp(-448.0, 448.0).to(_t.float8_e4m3fn)
    b_fp8 = (b_bf16.float() / b_scale).clamp(-448.0, 448.0).to(_t.float8_e4m3fn)
    return _t._scaled_mm(a_fp8, b_fp8.t(),
                          scale_a=a_scale, scale_b=b_scale,
                          out_dtype=a_bf16.dtype)


def _select_fa_impl():'''
src, n2 = re.subn(helper_pat, helper_rep, src, count=1); assert n2 == 1, "helper inject failed"

# Replace `out = F.linear(post, w2)` inside the autograd Function fwd
down_pat = r"        out = F\.linear\(post, w2\)\n"
down_rep = "        out = _fp8_linear(post, w2.to(post.dtype)) if _FP8_MLP_ENABLED else F.linear(post, w2)\n"
src, n3 = re.subn(down_pat, down_rep, src, count=1)
assert n3 == 1, f"down-projection F.linear(post, w2) not matched (n={n3})"

# Patch 3: PURE_BENCH_MODE exit
m = re.search(r"(\n)(\s*)timed_eval\(\s*\n\s*\"diagnostic pre-quantization", src)
assert m, "PURE_BENCH_MODE inject site not found"
indent = m.group(2)
inject = (m.group(1)
          + indent + 'if int(__import__("os").environ.get("PURE_BENCH_MODE", "0")):\n'
          + indent + '    print("[PURE_BENCH_MODE] training complete, skipping eval", flush=True)\n'
          + indent + '    import sys as _sys; _sys.exit(0)\n')
src = src[:m.start()] + inject + src[m.start()+1:]

open(path, "w").write(src)
import py_compile
py_compile.compile(path, doraise=True)
print("patched OK, syntax valid")
PYPATCH

# ============= COMMON TRAIN ENV =============
export DATA_PATH=$DATA_PATH_LOCAL TOKENIZER_PATH=$SUB_PARALLEL/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
export VOCAB_SIZE=8192 CASEOPS_ENABLED=1
export ITERATIONS=100 MAX_WALLCLOCK_SECONDS=300 TRAIN_LOG_EVERY=10
export TRAIN_SEQ_LEN=3072 EVAL_SEQ_LEN=3072 TTT_EVAL_SEQ_LEN=3072
export TRAIN_SEQ_SCHEDULE="" PURE_BENCH_MODE=1
export FUSED_CE_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1 SMEAR_GATE_ENABLED=1 GATE_WINDOW=12
export LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4 LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64
export TTT_WARM_START_A=1 EMBED_BITS=7 MIN_LR=0.1 MATRIX_LR=0.026
export MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 MLP_CLIP_SIGMAS=11.5 EMBED_CLIP_SIGMAS=14.0
export GPTQ_RESERVE_SECONDS=4.0 GPTQ_CALIBRATION_BATCHES=16
export WARMDOWN_FRAC=0.85 BETA2=0.99 TTT_BETA2=0.99 TTT_WEIGHT_DECAY=0.5
export TTT_LORA_RANK=80 SPARSE_ATTN_GATE_SCALE=0.5 PHASED_TTT_NUM_PHASES=1 PHASED_TTT_PREFIX_DOCS=2500
export AWQ_LITE_ENABLED=1 AWQ_LITE_BITS=8 AWQ_LITE_GROUP_SIZE=64 AWQ_LITE_GROUP_TOP_K=1
export COMPRESSOR=pergroup
export TTT_MASK=no_qv TTT_Q_LORA=0 TTT_V_LORA=0 TTT_LOCAL_LR_MULT=0.75 QK_GAIN_INIT=5.25
export NGRAM_MIX_ENABLED=0 TEMP_SCALE_ENABLED=0 PPM_MIX_ENABLED=0 TTT_SHORT_SCORE_FIRST_ENABLED=0 TTT_WARM_START_MEAN_ENABLED=0

# Both variants run with cuda graphs ON (locked in from Phase A2)
export CUDA_GRAPHS_ENABLED=1

mkdir -p /workspace/runs

run_variant() {
  local tag=$1; shift
  local fp8=$1; shift
  local desc="$@"
  echo "[$(date)] ============================================================"
  echo "[$(date)] === VARIANT: $tag — $desc"
  echo "[$(date)] ============================================================"
  RID=a3_${tag}
  RDIR=/workspace/runs/$RID
  rm -rf "$RDIR"; mkdir -p "$RDIR"
  export RUN_ID=$RID SEED=42 QUANTIZED_MODEL_PATH=$RDIR/model.bin
  export FP8_MLP_ENABLED=$fp8
  ( cd "$SUB_TRAIN" && \
    torchrun --standalone --nproc_per_node=1 "$SUB_TRAIN/train_gpt.py" \
  ) 2>&1 | tee "$RDIR/train.out" || echo "[WARN] $tag returned non-zero"
}

# A: bf16 MLP baseline with cuda graphs ON (= Phase A2 B)
run_variant A_bf16_mlp 0 "bf16 MLP + cuda graphs (= Phase A2 B_cudagraphs)"

# B: fp8 MLP with cuda graphs ON
run_variant B_fp8_mlp 1 "FP8 e4m3 MLP (torch._scaled_mm) + cuda graphs"

# ============= PARSE + UPLOAD =============
python3 - <<'PYPARSE'
import re, statistics, json, os
results = {}
for tag in ("A_bf16_mlp", "B_fp8_mlp"):
    path = f"/workspace/runs/a3_{tag}/train.out"
    if not os.path.exists(path):
        print(f"{tag}: skipped (no log)"); continue
    log = open(path).read()
    rows = []
    for ln in log.split("\n"):
        m = re.search(r"^(\d+)/\d+\s+train_loss:\s+([\d.]+|nan|NaN|inf)\s+train_time:\s+([\d.]+)m\s+tok/s:\s+(\d+)", ln)
        if m: rows.append({"step": int(m.group(1)), "loss": m.group(2), "throughput": float(m.group(4))})
    warm_rows = [r for r in rows if r.get("step", 0) > 10]
    throughputs = [r["throughput"] for r in warm_rows]
    summary = {
        "tag": tag, "n_logged_steps": len(rows), "n_steady_state": len(throughputs),
        "throughput_mean": statistics.mean(throughputs) if throughputs else None,
        "throughput_median": statistics.median(throughputs) if throughputs else None,
        "min_throughput": min(throughputs) if throughputs else None,
        "max_throughput": max(throughputs) if throughputs else None,
        "loss_at_100": rows[-1]["loss"] if rows else None,
    }
    results[tag] = summary
    print(f"\n{tag}:")
    for k, v in summary.items(): print(f"  {k}: {v}")

with open("/workspace/runs/phase_a3_summary.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n\n=== COMPARISON ===")
print(f"{'Variant':<22} {'Mean tok/s':>14} {'Median':>14} {'Loss@100':>10}")
for tag, s in results.items():
    print(f"{tag:<22} {s['throughput_mean'] or 0:>14,.0f} {s['throughput_median'] or 0:>14,.0f} {str(s.get('loss_at_100', '?')):>10}")
a = results.get("A_bf16_mlp", {}).get("throughput_mean") or 0
b = results.get("B_fp8_mlp", {}).get("throughput_mean") or 0
if a and b:
    pct = (b - a) / a * 100
    print(f"\nDelta (B vs A): {pct:+.2f}%  ({b - a:+,.0f} tok/s)")
print("\nReference (Phase A2 B_cudagraphs):       1,428,986 tok/s")
print("Reference (Phase Z baseline, no graphs): 1,378,166 tok/s")
PYPARSE

# Upload to HF
python3 - <<'PYUP' || echo "[WARN] HF upload"
import os, glob
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo_id="FijaEE/parameter-golf-fa4-bench", repo_type="dataset", private=True, exist_ok=True)
for tag in ("A_bf16_mlp", "B_fp8_mlp"):
    for f in glob.glob(f"/workspace/runs/a3_{tag}/*"):
        if os.path.isfile(f):
            try:
                api.upload_file(
                    path_or_fileobj=open(f, "rb").read(),
                    path_in_repo=f"phase_a3/{tag}/{os.path.basename(f)}",
                    repo_id="FijaEE/parameter-golf-fa4-bench",
                    repo_type="dataset", commit_message="phase A3 variant " + tag,
                )
                print("uploaded", f)
            except Exception as e: print(f, e)
for f in ["/workspace/runs/phase_a3_summary.json", "/workspace/heartbeat.log", "/workspace/exit_trap.log"]:
    if os.path.exists(f):
        try:
            api.upload_file(path_or_fileobj=open(f, "rb").read(),
                           path_in_repo=f"phase_a3/{os.path.basename(f)}",
                           repo_id="FijaEE/parameter-golf-fa4-bench",
                           repo_type="dataset", commit_message="phase A3 final")
            print("uploaded", f)
        except Exception as e: print(f, e)
PYUP

final_cleanup

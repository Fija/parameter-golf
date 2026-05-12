#!/bin/bash
# Phase Z — unlock B200's true throughput on PR #2014 workload
#
# Yesterday's Phase Y baseline: B200 FA4-mixed = 1.40 M tok/s @ 0.563s/step
# Workload analysis:
#   - MLP `linear_leaky_relu_square` triton kernel = ~55% of step time
#   - Attention = ~10%
#   - Other (optim, EMA, scalar ops, launch overhead) = ~35%
#   - MFU on B200 = 0.02% — kernel emit was Hopper-era PTX, not tcgen05
#
# Hypothesis: upgrading Triton 3.5.1 → 3.7+ lets the MLP kernel emit native
# sm_100 PTX (tcgen05.mma + TMEM accumulator). Block size tuning also helps.
#
# Phase Z runs 3 variants on SAME pod for clean comparison:
#   A. Triton 3.7 baseline (same block config as Phase Y v6)
#   B. A + K-fat tile (128×256×128 ns=2)  — more arithmetic intensity per tile
#   C. A + MN-fat tile (256×256×32 ns=4) — bigger output tile, smaller K
#
# Plus baseline reference (Triton 3.5.1, original image) by skipping upgrade
# in variant 0.

set -u
set -x

POD_ID=${RUNPOD_POD_ID:-}
HARD_DEADLINE_MIN=${HARD_DEADLINE_MIN:-60}

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
  echo "[$(date)] === Phase Z done; auto-stop pod $POD_ID ==="
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

python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); cap=torch.cuda.get_device_capability(0); print('capability', cap)"

# Standard deps
pip install --break-system-packages --quiet \
  sentencepiece huggingface_hub python-minifier brotli zstandard numpy 2>&1 | tail -2 || true

# FA4 deps (sm_100)
pip install --break-system-packages --quiet cuda-python 2>&1 | tail -2 || true
pip install --break-system-packages --quiet "nvidia-cutlass-dsl==4.2.1" 2>&1 | tail -2 || true
# Stub experimental (CUDA 13.1 check)
EXP=/usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/experimental/__init__.py
if [ -f "$EXP" ] && grep -q "NotImplementedError" "$EXP"; then
  cp "$EXP" "$EXP.orig"
  cat > "$EXP" <<EOF_STUB
import warnings
warnings.warn("cutlass.cute.experimental stubbed (CUDA <13.1)", stacklevel=2)
EOF_STUB
fi

# Phase Z key change: upgrade Triton to 3.7+
echo "[$(date)] === BEFORE: triton version ==="
python3 -c "import triton; print('triton', triton.__version__)"
pip install --break-system-packages --upgrade "triton>=3.7.0,<4.0" 2>&1 | tail -3 || true
echo "[$(date)] === AFTER: triton version ==="
python3 -c "import triton; print('triton', triton.__version__)"

# Verify FA4 still imports after Triton upgrade
python3 -c "from flash_attn.cute.interface import flash_attn_func; print('FA4 import OK')" 2>&1 | tail -3 || echo "[WARN] FA4 import broken"

# ============= REPO + DATA =============
BRANCH=${BRANCH:-submission/pr1797-ngram-mix}
REPO=/workspace/parameter-golf
SUB_PARALLEL=$REPO/records/track_non_record_16mb/2026-04-28_PR1797_EmbedClipRelax_AblationStack
SUB_TRAIN=$REPO/records/track_10min_16mb/2026-04-30_PR2014_Reproduction_1.0583
if [ ! -d "$REPO/.git" ]; then
  git clone --depth=1 --branch "$BRANCH" https://github.com/Fija/parameter-golf.git "$REPO" 2>&1 | tail -3
fi

# Range download docs
cd $REPO
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

# Mini-retokenize SP8192
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
TRAIN_PY=$SUB_TRAIN/train_gpt.py
cp $TRAIN_PY ${TRAIN_PY}.orig
python3 - "$TRAIN_PY" <<'PYPATCH'
import sys, re
path = sys.argv[1]
src = open(path).read()

# 1) FA dispatcher (FA4 basic + FA2 varlen for now, FA4_FULL gated by env)
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
            print(f"[fa-dispatch] {name}: FA4 unavailable ({type(e).__name__}); FA2 basic", flush=True)
        from flash_attn import flash_attn_func as _fa2_func, flash_attn_varlen_func as _fa2_varlen
        if basic_fn is None:
            basic_fn = _fa2_func
        varlen_fn = _fa2_varlen
        print(f"[fa-dispatch] {name}: varlen=FA2", flush=True)
        return basic_fn, varlen_fn
    if major == 9:
        from flash_attn_interface import flash_attn_func, flash_attn_varlen_func
        print(f"[fa-dispatch] {name}: FA3", flush=True)
        return flash_attn_func, flash_attn_varlen_func
    raise NotImplementedError(f"no FA impl for {name}")

flash_attn_3_func, flash_attn_varlen_func = _select_fa_impl()'''
src, n = re.subn(fa_pat, fa_rep, src, count=1, flags=re.M); assert n == 1

# 2) MLP block size — read from env vars (Phase Z variants override default)
bp = r"    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 256, 128, 64\n    num_stages = 4 if forward else 3"
br = '''    import os as _os
    _ovr_m = int(_os.environ.get("LRS_BLOCK_M", "0"))
    if _ovr_m > 0:
        BLOCK_SIZE_M = _ovr_m
        BLOCK_SIZE_N = int(_os.environ.get("LRS_BLOCK_N", "128"))
        BLOCK_SIZE_K = int(_os.environ.get("LRS_BLOCK_K", "64"))
        _ovr_s = int(_os.environ.get("LRS_NUM_STAGES_FWD", "0"))
        _ovr_sb = int(_os.environ.get("LRS_NUM_STAGES_BWD", "0"))
        num_stages = _ovr_s if forward else (_ovr_sb if _ovr_sb else max(1, _ovr_s - 1))
    else:
        _cap = torch.cuda.get_device_capability(a.device)
        if _cap == (9, 0):
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 256, 128, 64
            num_stages = 4 if forward else 3
        elif _cap == (10, 0):
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 256, 128, 64
            num_stages = 4 if forward else 3
        elif _cap == (12, 0):
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 128, 64
            num_stages = 3 if forward else 2
        else:
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 128, 32
            num_stages = 3 if forward else 2'''
src, n = re.subn(bp, br, src, count=1); assert n == 1

# 3) PURE_BENCH_MODE exit before first timed_eval
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
cd "$SUB_TRAIN"
export DATA_PATH=$DATA_PATH_LOCAL TOKENIZER_PATH=$SUB_PARALLEL/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
export VOCAB_SIZE=8192 CASEOPS_ENABLED=1
export ITERATIONS=100 MAX_WALLCLOCK_SECONDS=300 TRAIN_LOG_EVERY=10
export TRAIN_SEQ_LEN=3072 EVAL_SEQ_LEN=3072 TTT_EVAL_SEQ_LEN=3072
export TRAIN_SEQ_SCHEDULE="" PURE_BENCH_MODE=1
# PR #2014 stack
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

mkdir -p /workspace/runs

# ============= RUN VARIANTS =============
run_variant() {
  local tag=$1; shift
  local desc="$@"
  echo "[$(date)] ============================================================"
  echo "[$(date)] === VARIANT: $tag — $desc"
  echo "[$(date)] ============================================================"
  RID=z_${tag}
  RDIR=/workspace/runs/$RID
  rm -rf "$RDIR"; mkdir -p "$RDIR"
  export RUN_ID=$RID SEED=42 QUANTIZED_MODEL_PATH=$RDIR/model.bin
  torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "$RDIR/train.out" || echo "[WARN] $tag returned non-zero"
}

# A: Triton 3.7 baseline (default block 256×128×64 ns=4) — measures pure Triton upgrade effect
unset LRS_BLOCK_M LRS_BLOCK_N LRS_BLOCK_K LRS_NUM_STAGES_FWD LRS_NUM_STAGES_BWD
run_variant A_triton37_baseline "Triton 3.7 + default 256x128x64 ns=4 (pure Triton upgrade)"

# B: K-fat tile (more arithmetic intensity per tile)
export LRS_BLOCK_M=128 LRS_BLOCK_N=256 LRS_BLOCK_K=128
export LRS_NUM_STAGES_FWD=2 LRS_NUM_STAGES_BWD=2
run_variant B_block_Kfat "Triton 3.7 + 128x256x128 ns=2 (K-fat tile, 192 KB SMEM)"

# C: MN-fat tile (bigger output tile, smaller K)
export LRS_BLOCK_M=256 LRS_BLOCK_N=256 LRS_BLOCK_K=32
export LRS_NUM_STAGES_FWD=4 LRS_NUM_STAGES_BWD=3
run_variant C_block_MNfat "Triton 3.7 + 256x256x32 ns=4 (MN-fat tile, 128 KB SMEM)"

# ============= PARSE + UPLOAD =============
python3 - <<'PYPARSE'
import re, csv, statistics, json, os
results = {}
for tag in ("A_triton37_baseline", "B_block_Kfat", "C_block_MNfat"):
    path = f"/workspace/runs/z_{tag}/train.out"
    if not os.path.exists(path): continue
    log = open(path).read()
    rows = []
    for ln in log.split("\n"):
        m = re.search(r"^(\d+)/\d+\s+train_loss:\s+([\d.]+)\s+train_time:\s+([\d.]+)m\s+tok/s:\s+(\d+)", ln)
        if m: rows.append({"step": int(m.group(1)), "throughput": float(m.group(4))})
    warm_rows = [r for r in rows if r.get("step", 0) > 10]
    throughputs = [r["throughput"] for r in warm_rows]
    summary = {
        "tag": tag,
        "n_logged_steps": len(rows),
        "n_steady_state": len(throughputs),
        "throughput_mean": statistics.mean(throughputs) if throughputs else None,
        "throughput_median": statistics.median(throughputs) if throughputs else None,
        "min_throughput": min(throughputs) if throughputs else None,
        "max_throughput": max(throughputs) if throughputs else None,
    }
    results[tag] = summary
    print(f"\n{tag}:")
    for k, v in summary.items(): print(f"  {k}: {v}")

with open("/workspace/runs/phase_z_summary.json", "w") as f:
    json.dump(results, f, indent=2)

# Comparison table
print("\n\n=== COMPARISON ===")
print(f"{'Variant':<28} {'Mean tok/s':>14} {'Median':>14}")
for tag, s in results.items():
    print(f"{tag:<28} {s['throughput_mean'] or 0:>14,.0f} {s['throughput_median'] or 0:>14,.0f}")

print("\nReference (Phase Y v6 yesterday):")
print(f"{'b200_v6_yesterday_baseline':<28} {1369644:>14,} {1369644:>14,}")
PYPARSE

# Upload all variants
python3 - <<'PYUP' || echo "[WARN] HF upload"
import os, glob
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo_id="FijaEE/parameter-golf-fa4-bench", repo_type="dataset", private=True, exist_ok=True)
for tag in ("A_triton37_baseline", "B_block_Kfat", "C_block_MNfat"):
    for f in glob.glob(f"/workspace/runs/z_{tag}/*"):
        if os.path.isfile(f):
            try:
                api.upload_file(
                    path_or_fileobj=open(f, "rb").read(),
                    path_in_repo=f"phase_z/{tag}/{os.path.basename(f)}",
                    repo_id="FijaEE/parameter-golf-fa4-bench",
                    repo_type="dataset", commit_message="phase Z variant " + tag,
                )
                print("uploaded", f)
            except Exception as e: print(f, e)
for f in ["/workspace/runs/phase_z_summary.json", "/workspace/heartbeat.log"]:
    if os.path.exists(f):
        api.upload_file(path_or_fileobj=open(f, "rb").read(),
                       path_in_repo=f"phase_z/{os.path.basename(f)}",
                       repo_id="FijaEE/parameter-golf-fa4-bench",
                       repo_type="dataset", commit_message="phase Z final")
        print("uploaded", f)
PYUP

final_cleanup

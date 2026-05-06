#!/bin/bash
# Phase Y — B200 vs 4x Pro 6000 Max-Q step-time benchmark on PR #2014 stack
#
# Both pods run this script. Auto-detects GPU capability and patches:
#   - FA dispatch: FA4 on Blackwell, FA3 on Hopper, FA2 fallback for sm_120
#   - linear_leaky_relu_square block sizes: tuned per SMEM size
#     - sm_90 (H100):  256x128x64 ns=4/3 (192 KB, 84% of 228 KB)
#     - sm_100 (B200): 256x128x64 ns=5/4 (240 KB, 94% of 256 KB)
#     - sm_120 (Pro 6000 Max-Q): 128x128x64 ns=3/2 (96 KB, 95% of 101 KB)
#
# Test: 100 steps at fixed seq_len=3072 (no progressive schedule), no eval.
# Produces step-time CSV uploaded to HF + auto-stops pod.
set -u
set -x

POD_ID=${RUNPOD_POD_ID:-}
HARD_DEADLINE_MIN=${HARD_DEADLINE_MIN:-60}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}     # 1 for B200, 4 for Pro 6000 Max-Q
PROBE_TAG=${PROBE_TAG:-unknown}          # "b200" or "pro6000_maxq_4x"

heartbeat_loop() {
  while true; do
    {
      printf '[%s] ' "$(date -u +%H:%M:%S)"
      ps -ef | grep -v grep | grep -cE 'prepare_caseops|train_gpt' | tr -d '\n'
      printf ' procs disk='
      df -h /workspace 2>/dev/null | tail -1 | awk '{printf "%s ", $5}'
      shards=$(ls /workspace/data/sp8192/datasets/*/fineweb_*.bin 2>/dev/null | wc -l)
      printf 'sp8192=%d\n' "$shards"
    } >> /workspace/heartbeat.log 2>&1
    sleep 30
  done
}
heartbeat_loop &
HB_PID=$!

( sleep $((HARD_DEADLINE_MIN*60)) && [ -n "$POD_ID" ] && runpodctl stop pod "$POD_ID" ) &
KILL_PID=$!

final_cleanup() {
  echo "[$(date)] === Phase Y ($PROBE_TAG) done; auto-stop pod $POD_ID ==="
  kill $KILL_PID $HB_PID 2>/dev/null || true
  if [ -n "$POD_ID" ]; then
    runpodctl stop pod "$POD_ID" 2>&1 | head -3 || true
    if [ -n "${RUNPOD_API_KEY:-}" ]; then
      curl -sS -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"query\":\"mutation { podStop(input: {podId: \\\"$POD_ID\\\"}) { id } }\"}" 2>&1 | head -3
    fi
  fi
}

# --- 1. Apt deps ---
apt-get update -qq 2>&1 | tail -1
apt-get install -y -qq lrzip pixz xz-utils 2>&1 | tail -1
which lrzip || echo "[WARN] lrzip missing"

# --- 2. Detect GPU capability ---
nvidia-smi -L | head -8
GPU_CAP=$(python3 -c "
import torch
cap = torch.cuda.get_device_capability(0)
print(f'{cap[0]}.{cap[1]}')")
GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
echo "[$(date)] GPU detected: $GPU_NAME (sm_${GPU_CAP/./})"

# --- 3. Pip deps + FA install (cuda-python is the key fix for FA4) ---
pip install --break-system-packages --quiet \
  sentencepiece huggingface_hub python-minifier brotli zstandard numpy cuda-python 2>&1 | tail -3 || true

# Install appropriate flash-attn variant per capability
case "$GPU_CAP" in
  9.0)
    echo "[$(date)] Hopper: using preinstalled FA3"
    python3 -c "from flash_attn_interface import flash_attn_func; print('FA3 OK')" 2>&1 || \
      pip install --break-system-packages --quiet flash-attn-3 || true
    ;;
  10.0)
    echo "[$(date)] Blackwell DC: install FA4 (best-effort) + FA2 fallback"
    pip install --break-system-packages --quiet cuda-python nvidia-cutlass-dsl 2>&1 | tail -3 || true
    pip install --break-system-packages --quiet flash-attn-4 2>&1 | tail -3 || true
    python3 -c "from flash_attn.cute.interface import flash_attn_func; print('FA4 OK')" 2>&1 | head -5 || \
      echo "[INFO] FA4 unavailable on sm_100 — will use FA2"
    # Always install FA2 as guaranteed fallback on Blackwell DC
    pip install --break-system-packages --quiet flash-attn 2>&1 | tail -3 || true
    python3 -c "from flash_attn import flash_attn_func; print('FA2 OK')" 2>&1 | head -3 || \
      echo "[WARN] FA2 also missing"
    ;;
  12.0)
    echo "[$(date)] Blackwell Workstation: trying FA4 + FA2 fallback"
    pip install --break-system-packages --quiet cuda-python nvidia-cutlass-dsl 2>&1 | tail -3 || true
    pip install --break-system-packages --quiet flash-attn-4 2>&1 | tail -3 || true
    python3 -c "from flash_attn.cute.interface import flash_attn_func; print('FA4-cute OK')" 2>&1 | head -3 || \
      echo "[INFO] FA4 not available on sm_120, will use FA2"
    pip install --break-system-packages --quiet flash-attn 2>&1 | tail -3 || true
    python3 -c "from flash_attn import flash_attn_func; print('FA2 OK')" 2>&1 | head -3 || \
      echo "[WARN] FA2 also not available — may fall back to SDPA"
    ;;
  *)
    echo "[WARN] unknown GPU capability $GPU_CAP — will try FA2"
    pip install --break-system-packages --quiet flash-attn 2>&1 | tail -3 || true
    ;;
esac

# --- 4. Repo clone ---
BRANCH=${BRANCH:-submission/pr1797-ngram-mix}
REPO=/workspace/parameter-golf
SUB_PARALLEL=$REPO/records/track_non_record_16mb/2026-04-28_PR1797_EmbedClipRelax_AblationStack
SUB_TRAIN=$REPO/records/track_10min_16mb/2026-04-30_PR2014_Reproduction_1.0583
if [ ! -d "$REPO/.git" ]; then
  rm -rf "$REPO"
  git clone --depth=1 --branch "$BRANCH" https://github.com/Fija/parameter-golf.git "$REPO" 2>&1 | tail -3 || { echo "[FATAL] clone"; exit 1; }
fi

# --- 5. Patch train_gpt.py for FA dispatch + block sizes ---
TRAIN_PY=$SUB_TRAIN/train_gpt.py
cp $TRAIN_PY ${TRAIN_PY}.orig
python3 - "$TRAIN_PY" <<'PYPATCH'
import sys, re
path = sys.argv[1]
src = open(path).read()

# Patch 1: replace the FA3 hard import with capability-dispatched selector
fa_import_pattern = r"^from flash_attn_interface import \(\s*\n\s*flash_attn_func as flash_attn_3_func,\s*\n\s*flash_attn_varlen_func,\s*\n\)"
fa_replacement = '''def _select_fa_impl():
    import torch as _t
    cap = _t.cuda.get_device_capability(0)
    major, minor = cap
    name = f"sm_{major}{minor}"
    # Both Blackwell variants: try FA4 first, fall back to FA2
    if major in (10, 12):
        try:
            from flash_attn.cute.interface import flash_attn_func, flash_attn_varlen_func
            print(f"[fa-dispatch] {name}: FA4 (CuTe)", flush=True)
            return flash_attn_func, flash_attn_varlen_func
        except (ImportError, RuntimeError, ModuleNotFoundError) as e:
            print(f"[fa-dispatch] {name}: FA4 unavailable ({type(e).__name__}: {e}); falling back to FA2", flush=True)
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        print(f"[fa-dispatch] {name}: FA2", flush=True)
        return flash_attn_func, flash_attn_varlen_func
    if major == 9:
        from flash_attn_interface import flash_attn_func, flash_attn_varlen_func
        print(f"[fa-dispatch] {name}: FA3", flush=True)
        return flash_attn_func, flash_attn_varlen_func
    raise NotImplementedError(f"no FA impl for {name}")

flash_attn_3_func, flash_attn_varlen_func = _select_fa_impl()'''

new_src, n = re.subn(fa_import_pattern, fa_replacement, src, count=1, flags=re.M)
assert n == 1, f"FA import patch failed (matched {n} times)"

# Patch 2: replace block-size constants in linear_leaky_relu_square
block_pattern = r"    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 256, 128, 64\n    num_stages = 4 if forward else 3"
block_replacement = '''    _cap = torch.cuda.get_device_capability(a.device)
    if _cap == (9, 0):                                    # Hopper
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 256, 128, 64
        num_stages = 4 if forward else 3
    elif _cap == (10, 0):                                 # Blackwell DC (B200/B300)
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 256, 128, 64
        num_stages = 5 if forward else 4                  # +1 for HBM3e
    elif _cap == (12, 0):                                 # Blackwell Workstation (Pro 6000)
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 128, 64
        num_stages = 3 if forward else 2                  # 96 KB SMEM
    else:                                                 # Ampere/Ada fallback
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K =  64, 128, 32
        num_stages = 3 if forward else 2'''

new_src2, n = re.subn(block_pattern, block_replacement, new_src, count=1)
assert n == 1, f"block patch failed (matched {n} times)"

# Patch 3: PURE_BENCH_MODE — find the timed_eval(...) call whose first arg is the diagnostic
# pre-quantization string, and inject our exit BEFORE the timed_eval line (not inside its args).
m = re.search(r"(\n)(\s*)timed_eval\(\s*\n\s*\"diagnostic pre-quantization", new_src2)
assert m, "PURE_BENCH_MODE inject site not found (expected timed_eval(\\n...\"diagnostic pre-quantization)"
indent = m.group(2)
inject = (m.group(1)
          + indent + 'if int(__import__("os").environ.get("PURE_BENCH_MODE", "0")):\n'
          + indent + '    print("[PURE_BENCH_MODE] training complete, skipping eval and exiting", flush=True)\n'
          + indent + '    import sys as _sys; _sys.exit(0)\n')
new_src2 = new_src2[:m.start()] + inject + new_src2[m.start()+1:]
print(f"injected PURE_BENCH_MODE exit before timed_eval('diagnostic pre-quantization' ...)")

open(path, "w").write(new_src2)

# Verify the patched file actually parses as valid Python
import py_compile
try:
    py_compile.compile(path, doraise=True)
    print(f"patched + syntax valid: {path}")
except py_compile.PyCompileError as e:
    print(f"[FATAL] patched train_gpt.py has syntax error: {e}")
    sys.exit(1)
PYPATCH

grep -A 5 "_select_fa_impl" $TRAIN_PY | head -25
echo "---"
grep -B 1 -A 12 "_cap = torch.cuda.get_device_capability(a.device)" $TRAIN_PY | head -30

# --- 6. Pull docs + mini-retokenize SP8192 (only first 50K docs) ---
cd $REPO
DOCS=$REPO/data/docs_selected.jsonl
[ ! -f "$DOCS" ] && { echo "[$(date)] === Downloading docs ==="; python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 0 --with-docs 2>&1 | tail -5 || { echo "[FATAL] docs"; exit 1; }; }

DATA_OUT=/workspace/data/sp8192
DATASET_NAME=fineweb10B_sp8192_lossless_caps_caseops_v1_reserved
DATA_PATH_LOCAL=$DATA_OUT/datasets/$DATASET_NAME

if [ ! -f "$DATA_PATH_LOCAL/fineweb_train_000000.bin" ]; then
  rm -rf "$DATA_OUT"; mkdir -p "$DATA_OUT"
  echo "[$(date)] === Mini-retokenize SP8192 (first 20K docs, ~80M tokens, enough for 100 steps) ==="
  head -n 20000 "$DOCS" > /workspace/docs_20k.jsonl
  WORKERS=$(python3 -c "print(min($(nproc) - 8, 64))")
  python3 -u $SUB_PARALLEL/prepare_caseops_data_parallel.py \
    --docs /workspace/docs_20k.jsonl --out "$DATA_OUT" \
    --sp "$SUB_PARALLEL/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model" \
    --val-docs 500 --workers $WORKERS --chunksize 128 || { echo "[FATAL] retokenize"; exit 1; }
fi
ls "$DATA_PATH_LOCAL" | head -3
du -sh "$DATA_PATH_LOCAL"
echo "[$(date)] === retokenize done ==="

# --- 7. Run 100-step bench ---
cd "$SUB_TRAIN"
export DATA_PATH=$DATA_PATH_LOCAL TOKENIZER_PATH=$SUB_PARALLEL/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
export VOCAB_SIZE=8192 CASEOPS_ENABLED=1
# Bench-specific overrides (no eval, fixed ctx, 100 steps)
export ITERATIONS=100
export MAX_WALLCLOCK_SECONDS=300
export TRAIN_SEQ_LEN=3072
export TRAIN_SEQ_SCHEDULE=""           # disable progressive schedule
export EVAL_SEQ_LEN=3072 TTT_EVAL_SEQ_LEN=3072
# PR #2014 stack (carry forward)
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
export NGRAM_MIX_ENABLED=0 TEMP_SCALE_ENABLED=0 PPM_MIX_ENABLED=0
# Skip TTT short-doc machinery for bench
export TTT_SHORT_SCORE_FIRST_ENABLED=0 TTT_WARM_START_MEAN_ENABLED=0
# PURE_BENCH_MODE: hard-exit after 100 train steps via injected sys.exit(0); skip ALL eval
export PURE_BENCH_MODE=1
export TRAIN_LOG_EVERY=10                              # log every 10 steps (10 datapoints over 100 steps)

mkdir -p /workspace/runs
RID=phy_${PROBE_TAG}
RDIR=/workspace/runs/$RID
mkdir -p "$RDIR"
export RUN_ID=$RID SEED=42 QUANTIZED_MODEL_PATH=$RDIR/model.bin

echo "[$(date)] === bench start ($PROBE_TAG, $NPROC_PER_NODE GPUs, ctx=3072, 100 steps) ==="
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE train_gpt.py 2>&1 | tee "$RDIR/train.out" || echo "[WARN] torchrun rc!=0"
echo "[$(date)] === bench done ==="

# --- 8. Parse step times ---
python3 - <<PYPARSE
import re, csv, statistics, json
log = open("$RDIR/train.out").read()
# Match lines like: "step    20/100 train_loss: 5.7251 train_time: 0.5m tok/s: 543290"
# or:               "step  20/100 | loss 5.72 | lr_mul 0.10 | mom 0.85 | 0.55 steps/s | 9s"
rows = []
for ln in log.split("\n"):
    # PR #2014 format: "N/100 train_loss: X train_time: Ym tok/s: Z"
    m = re.search(r"^(\d+)/\d+\s+train_loss:\s+([\d.]+)\s+train_time:\s+([\d.]+)m\s+tok/s:\s+(\d+)", ln)
    if m:
        rows.append({"step": int(m.group(1)), "loss": float(m.group(2)), "train_time_min": float(m.group(3)), "throughput": float(m.group(4))})

# Save raw step rows
with open("/workspace/runs/$RID/steps.csv", "w") as f:
    if rows:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow(r)

# Summary stats: skip first 10 steps (warmup), report median + p10/p90 of remaining
import json
warmup = 10
warm_rows = [r for r in rows if r.get("step", 0) > warmup]
def pick(key):
    vals = [r[key] for r in warm_rows if key in r and isinstance(r[key], (int, float))]
    return vals
import os
gpu_name = os.environ.get("GPU_NAME") or "unknown"
gpu_cap = "$GPU_CAP"
nproc = $NPROC_PER_NODE
tag = "$PROBE_TAG"

throughputs = [r["throughput"] for r in warm_rows if "throughput" in r] or [r["tok_per_s"] for r in warm_rows if "tok_per_s" in r]
summary = {
    "tag": tag,
    "gpu_name": gpu_name,
    "gpu_capability": gpu_cap,
    "nproc_per_node": nproc,
    "ctx": 3072,
    "warmup_steps_excluded": warmup,
    "steady_state_steps": len(warm_rows),
    "throughput_mean_tok_per_s": statistics.mean(throughputs) if throughputs else None,
    "throughput_median_tok_per_s": statistics.median(throughputs) if throughputs else None,
    "throughput_p10_tok_per_s": sorted(throughputs)[int(len(throughputs)*0.1)] if len(throughputs) >= 10 else None,
    "throughput_p90_tok_per_s": sorted(throughputs)[int(len(throughputs)*0.9)] if len(throughputs) >= 10 else None,
}
with open("/workspace/runs/$RID/summary.json", "w") as f: json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
PYPARSE

# --- 9. Upload all artifacts to HF ---
python3 - <<PYUP || echo "[WARN] HF upload"
import os, glob, base64
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo_id="FijaEE/parameter-golf-fa4-bench", repo_type="dataset", private=True, exist_ok=True)
tag = "$PROBE_TAG"
files = [
    f"/workspace/runs/phy_{tag}/train.out",
    f"/workspace/runs/phy_{tag}/steps.csv",
    f"/workspace/runs/phy_{tag}/summary.json",
    "/workspace/heartbeat.log",
]
for f in files:
    if os.path.exists(f):
        api.upload_file(
            path_or_fileobj=open(f, "rb").read(),
            path_in_repo=f"{tag}/{os.path.basename(f)}",
            repo_id="FijaEE/parameter-golf-fa4-bench",
            repo_type="dataset", commit_message=f"phase Y bench {tag}",
        )
        print("uploaded", f, flush=True)
PYUP

final_cleanup

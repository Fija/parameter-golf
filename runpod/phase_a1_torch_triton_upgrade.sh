#!/bin/bash
# Phase A1 — Tier 3 stack upgrade: torch 2.6.0 → 2.11.0, triton 3.5.1 → 3.6.0
# (Optional variant C: force triton 3.7 — likely breaks inductor, attempted only
#  after B succeeds.)
#
# Why these versions: pytorch.org/whl/cu128 ships torch 2.11.0 with bundled
# triton 3.6.0. Triton 3.7.0 was just released (2026-05-07) and is too new for
# any stable torch to bundle. Forcing triton 3.7 on top of torch 2.11 likely
# hits the same `KernelMetadata.cluster_dims` AttributeError that Phase Z v5
# hit on torch 2.6 (inductor still references cluster_dims unguarded in 2.11).
#
# Variants on same pod:
#   A. image baseline (torch 2.6.0 + triton 3.5.1) — sanity = Phase Z baseline
#   B. torch 2.11.0 + triton 3.6.0 (bundled) + flash-attn rebuilt from source
#   C. + force triton==3.7.0 on top — try if B succeeds
#
# Hard fail-fast smoke gates at every step so we don't burn $5.49/hr in
# silent installs.

set -x

POD_ID=${RUNPOD_POD_ID:-}
HARD_DEADLINE_MIN=${HARD_DEADLINE_MIN:-90}

trap 'EXIT_CODE=$?; echo "[$(date)] === TRAP EXIT code=$EXIT_CODE at line $LINENO ===" >> /workspace/exit_trap.log; ps -ef >> /workspace/exit_trap.log 2>&1' EXIT

heartbeat_loop() {
  while true; do
    {
      printf '[%s] ' "$(date -u +%H:%M:%S)"
      ps -ef | grep -v grep | grep -cE 'prepare_caseops|train_gpt|pip|build' | tr -d '\n'
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
  echo "[$(date)] === Phase A1 done; auto-stop pod $POD_ID ==="
  kill $KILL_PID $HB_PID 2>/dev/null || true
  if [ -n "$POD_ID" ]; then
    runpodctl stop pod "$POD_ID" 2>&1 | head -3 || true
    [ -n "${RUNPOD_API_KEY:-}" ] && curl -sS -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"query\":\"mutation { podStop(input: {podId: \\\"$POD_ID\\\"}) { id } }\"}" 2>&1 | head -3
  fi
}

# ============= COMMON SETUP =============
apt-get update -qq 2>&1 | tail -1
apt-get install -y -qq lrzip pixz xz-utils 2>&1 | tail -1

pip install --break-system-packages --quiet \
  sentencepiece huggingface_hub python-minifier brotli zstandard numpy 2>&1 | tail -2 || true

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
ls -la "$SUB_TRAIN/train_gpt.py" 2>&1 | head -1

# Range download docs
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

# Mini-retokenize
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

# ============= PATCH train_gpt.py (FA dispatch + MLP block override + PURE_BENCH) =============
TRAIN_PY=$SUB_TRAIN/train_gpt.py
cp $TRAIN_PY ${TRAIN_PY}.orig
python3 - "$TRAIN_PY" <<'PYPATCH'
import sys, re
path = sys.argv[1]
src = open(path).read()

# FA dispatcher: try FA4 CuTe, fall back to FA2 basic; varlen always FA2.
# If both FA2 and FA4 fail, fall back to torch SDPA scaled_dot_product_attention.
fa_pat = r"^from flash_attn_interface import \(\s*\n\s*flash_attn_func as flash_attn_3_func,\s*\n\s*flash_attn_varlen_func,\s*\n\)"
fa_rep = '''def _select_fa_impl():
    import os, torch as _t
    cap = _t.cuda.get_device_capability(0)
    major, minor = cap
    name = f"sm_{major}{minor}"
    basic_fn = None
    varlen_fn = None
    if major in (10, 12):
        try:
            from flash_attn.cute.interface import flash_attn_func as _fa4_func
            basic_fn = _fa4_func
            print(f"[fa-dispatch] {name}: basic=FA4 (CuTe)", flush=True)
        except Exception as e:
            print(f"[fa-dispatch] {name}: FA4 unavailable ({type(e).__name__}: {e})", flush=True)
        try:
            from flash_attn import flash_attn_func as _fa2_func, flash_attn_varlen_func as _fa2_varlen
            if basic_fn is None:
                basic_fn = _fa2_func
                print(f"[fa-dispatch] {name}: basic=FA2 (fallback)", flush=True)
            varlen_fn = _fa2_varlen
            print(f"[fa-dispatch] {name}: varlen=FA2", flush=True)
        except Exception as e:
            print(f"[fa-dispatch] {name}: FA2 unavailable ({type(e).__name__}: {e})", flush=True)
    elif major == 9:
        try:
            from flash_attn_interface import flash_attn_func, flash_attn_varlen_func
            print(f"[fa-dispatch] {name}: FA3", flush=True)
            return flash_attn_func, flash_attn_varlen_func
        except Exception as e:
            print(f"[fa-dispatch] {name}: FA3 unavailable ({type(e).__name__})", flush=True)
    # Last resort: SDPA wrappers (matches FA signature)
    if basic_fn is None or varlen_fn is None:
        print(f"[fa-dispatch] {name}: falling back to torch SDPA wrappers", flush=True)
        import torch.nn.functional as _F
        def _sdpa_basic(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **kw):
            if softmax_scale is None:
                softmax_scale = q.size(-1) ** -0.5
            return _F.scaled_dot_product_attention(
                q.transpose(1,2).contiguous(), k.transpose(1,2).contiguous(), v.transpose(1,2).contiguous(),
                is_causal=causal, scale=softmax_scale, dropout_p=dropout_p
            ).transpose(1,2).contiguous()
        if basic_fn is None: basic_fn = _sdpa_basic
        if varlen_fn is None: varlen_fn = _sdpa_basic  # workload may not need varlen
    return basic_fn, varlen_fn

flash_attn_3_func, flash_attn_varlen_func = _select_fa_impl()'''
src, n = re.subn(fa_pat, fa_rep, src, count=1, flags=re.M); assert n == 1, "FA pattern not matched"

# PURE_BENCH_MODE exit before first timed_eval
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

# ============= COMMON TRAIN ENV (same as Phase Z) =============
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

mkdir -p /workspace/runs

# ============= VARIANT RUNNER =============
run_variant() {
  local tag=$1; shift
  local desc="$@"
  echo "[$(date)] ============================================================"
  echo "[$(date)] === VARIANT: $tag — $desc"
  echo "[$(date)] ============================================================"
  RID=a1_${tag}
  RDIR=/workspace/runs/$RID
  rm -rf "$RDIR"; mkdir -p "$RDIR"
  export RUN_ID=$RID SEED=42 QUANTIZED_MODEL_PATH=$RDIR/model.bin
  # Snapshot the python stack for this variant
  echo "[$(date)] === stack snapshot ($tag) ===" > "$RDIR/stack.txt"
  python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" 2>&1 | tee -a "$RDIR/stack.txt"
  python3 -c "import triton; print('triton', triton.__version__)" 2>&1 | tee -a "$RDIR/stack.txt"
  python3 -c "import flash_attn; print('flash_attn', flash_attn.__version__)" 2>&1 | tee -a "$RDIR/stack.txt"
  python3 -c "from flash_attn.cute.interface import flash_attn_func; print('FA4 cute OK')" 2>&1 | tee -a "$RDIR/stack.txt"
  ( cd "$SUB_TRAIN" && \
    torchrun --standalone --nproc_per_node=1 "$SUB_TRAIN/train_gpt.py" \
  ) 2>&1 | tee "$RDIR/train.out" || echo "[WARN] $tag returned non-zero"
}

# ============= VARIANT A — IMAGE BASELINE =============
# Capture torch 2.6.0 + triton 3.5.1 baseline (= Phase Z A_baseline = 1.378M)
# for clean apples-to-apples comparison on THIS pod.
echo "[$(date)] === pre-A: image stack ==="
python3 -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__)"
# FA4 cute deps for image baseline (same as Phase Z)
pip install --break-system-packages --quiet cuda-python 2>&1 | tail -2 || true
pip install --break-system-packages --quiet "nvidia-cutlass-dsl==4.2.1" 2>&1 | tail -2 || true
EXP=/usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/experimental/__init__.py
if [ -f "$EXP" ] && grep -q "NotImplementedError" "$EXP"; then
  cat > "$EXP" <<EOF_STUB
import warnings
warnings.warn("cutlass.cute.experimental stubbed (CUDA <13.1)", stacklevel=2)
EOF_STUB
fi
run_variant A_image_baseline "torch 2.6.0 + triton 3.5.1 (image default)"

# ============= UPGRADE TO TORCH 2.11.0 + TRITON 3.6.0 =============
echo "[$(date)] ============================================================"
echo "[$(date)] === UPGRADING: torch 2.6.0 → 2.11.0, triton 3.5.1 → 3.6.0"
echo "[$(date)] ============================================================"

# Uninstall in order to avoid conflicts
pip uninstall -y --break-system-packages flash-attn flash_attn 2>&1 | tail -2 || true
pip uninstall -y --break-system-packages torch torchaudio torchvision triton pytorch-triton 2>&1 | tail -2 || true
# Clear pip cache (saves disk + forces clean install)
pip cache purge 2>&1 | tail -1 || true

# Install torch 2.11 from PyTorch's cu128 stable index — this also brings the bundled triton 3.6.0
pip install --break-system-packages --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0 2>&1 | tail -3

# Smoke gate B.1: torch + triton import
echo "[$(date)] === smoke B.1: torch+triton after upgrade ==="
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('compiled with triton:', torch.backends.cuda.is_built())"
python3 -c "import triton; print('triton', triton.__version__)" || { echo "[FATAL] triton broken after torch upgrade"; final_cleanup; exit 1; }

# Re-install FA stack against new torch
# Try prebuilt wheel for torch 2.9 (latest available) — if ABI compatible, save the build time
FA_WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
echo "[$(date)] === installing flash-attn 2.8.3 prebuilt (torch 2.9 wheel — best effort ABI) ==="
pip install --break-system-packages "$FA_WHEEL_URL" 2>&1 | tail -3 || {
  echo "[WARN] prebuilt 2.9 wheel failed; trying build from source"
  pip install --break-system-packages flash-attn==2.8.3 --no-build-isolation 2>&1 | tail -5
}

# Re-install cute deps (may need fresher version against new torch)
pip install --break-system-packages --upgrade cuda-python 2>&1 | tail -2 || true
# Try newer cutlass-dsl first, fall back to 4.2.1 if newer breaks
pip install --break-system-packages "nvidia-cutlass-dsl==4.2.1" --force-reinstall 2>&1 | tail -2 || true
# Re-stub experimental (path changes per torch version maybe)
for EXP in /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/experimental/__init__.py \
           /usr/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/experimental/__init__.py; do
  if [ -f "$EXP" ] && grep -q "NotImplementedError" "$EXP"; then
    cat > "$EXP" <<EOF_STUB
import warnings
warnings.warn("cutlass.cute.experimental stubbed (CUDA <13.1)", stacklevel=2)
EOF_STUB
    echo "[$(date)] stubbed $EXP"
  fi
done

# Smoke gate B.2: FA import after rebuild
echo "[$(date)] === smoke B.2: FA imports ==="
python3 -c "import flash_attn; print('flash_attn', flash_attn.__version__)" 2>&1 | tail -3 || echo "[WARN] flash_attn import broken"
python3 -c "from flash_attn import flash_attn_func, flash_attn_varlen_func; print('FA2 OK')" 2>&1 | tail -3 || echo "[WARN] FA2 import broken"
python3 -c "from flash_attn.cute.interface import flash_attn_func; print('FA4 OK')" 2>&1 | tail -3 || echo "[WARN] FA4 cute import broken"

# Smoke gate B.3: PURE_BENCH_MODE script still patches correctly with new torch
echo "[$(date)] === smoke B.3: re-verify train_gpt.py syntax ==="
python3 -m py_compile "$TRAIN_PY" && echo "syntax OK" || { echo "[FATAL] train_gpt.py syntax broke"; final_cleanup; exit 1; }

# ============= VARIANT B — torch 2.11 + triton 3.6 =============
run_variant B_torch211_triton36 "torch 2.11.0 + triton 3.6.0 (full upgrade)"

# ============= VARIANT C — force triton 3.7 (likely breaks; attempt if B succeeded) =============
B_STEPS=$(grep -c "/100 train_loss:" /workspace/runs/a1_B_torch211_triton36/train.out 2>/dev/null || echo 0)
if [ "$B_STEPS" -ge 5 ]; then
  echo "[$(date)] === VARIANT B succeeded ($B_STEPS log lines); attempting C ==="
  pip install --break-system-packages --upgrade "triton>=3.7.0,<4.0" 2>&1 | tail -3 || true
  python3 -c "import triton; print('triton', triton.__version__)"
  run_variant C_torch211_triton37 "torch 2.11.0 + force triton 3.7.0 (may break inductor)"
else
  echo "[$(date)] === VARIANT B failed (only $B_STEPS log lines); skipping C ==="
fi

# ============= PARSE + UPLOAD =============
python3 - <<'PYPARSE'
import re, statistics, json, os
results = {}
for tag in ("A_image_baseline", "B_torch211_triton36", "C_torch211_triton37"):
    path = f"/workspace/runs/a1_{tag}/train.out"
    if not os.path.exists(path):
        print(f"{tag}: skipped (no log)"); continue
    log = open(path).read()
    rows = []
    for ln in log.split("\n"):
        m = re.search(r"^(\d+)/\d+\s+train_loss:\s+([\d.]+)\s+train_time:\s+([\d.]+)m\s+tok/s:\s+(\d+)", ln)
        if m: rows.append({"step": int(m.group(1)), "throughput": float(m.group(4))})
    warm_rows = [r for r in rows if r.get("step", 0) > 10]
    throughputs = [r["throughput"] for r in warm_rows]
    summary = {
        "tag": tag, "n_logged_steps": len(rows), "n_steady_state": len(throughputs),
        "throughput_mean": statistics.mean(throughputs) if throughputs else None,
        "throughput_median": statistics.median(throughputs) if throughputs else None,
        "min_throughput": min(throughputs) if throughputs else None,
        "max_throughput": max(throughputs) if throughputs else None,
    }
    results[tag] = summary
    print(f"\n{tag}:")
    for k, v in summary.items(): print(f"  {k}: {v}")

with open("/workspace/runs/phase_a1_summary.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n\n=== COMPARISON ===")
print(f"{'Variant':<30} {'Mean tok/s':>14} {'Median':>14}")
for tag, s in results.items():
    print(f"{tag:<30} {s['throughput_mean'] or 0:>14,.0f} {s['throughput_median'] or 0:>14,.0f}")
print("\nReference (Phase Z A_baseline yesterday):")
print(f"{'phase_z_baseline':<30} {1378166:>14,} {1377070:>14,}")
PYPARSE

# Upload to HF
python3 - <<'PYUP' || echo "[WARN] HF upload"
import os, glob
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo_id="FijaEE/parameter-golf-fa4-bench", repo_type="dataset", private=True, exist_ok=True)
for tag in ("A_image_baseline", "B_torch211_triton36", "C_torch211_triton37"):
    for f in glob.glob(f"/workspace/runs/a1_{tag}/*"):
        if os.path.isfile(f):
            try:
                api.upload_file(
                    path_or_fileobj=open(f, "rb").read(),
                    path_in_repo=f"phase_a1/{tag}/{os.path.basename(f)}",
                    repo_id="FijaEE/parameter-golf-fa4-bench",
                    repo_type="dataset", commit_message="phase A1 variant " + tag,
                )
                print("uploaded", f)
            except Exception as e: print(f, e)
for f in ["/workspace/runs/phase_a1_summary.json", "/workspace/heartbeat.log", "/workspace/exit_trap.log"]:
    if os.path.exists(f):
        try:
            api.upload_file(path_or_fileobj=open(f, "rb").read(),
                           path_in_repo=f"phase_a1/{os.path.basename(f)}",
                           repo_id="FijaEE/parameter-golf-fa4-bench",
                           repo_type="dataset", commit_message="phase A1 final")
            print("uploaded", f)
        except Exception as e: print(f, e)
PYUP

final_cleanup

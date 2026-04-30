#!/bin/bash
# Phase U — Stack on top of PR #1953 (1.05855 BPB), pushing record bar.
#
# PR #1953 = PR #1945 (= #1855 + AWQ-lite + Asymmetric Logit Rescale) + 4 levers:
#   - EVAL_SEQ_LEN=2560, TTT_EVAL_SEQ_LEN=2560 (long-ctx)
#   - TTT_MASK=no_qv (TTT_Q_LORA=0, TTT_V_LORA=0)
#   - TTT_LOCAL_LR_MULT=0.75
#   - QK_GAIN_INIT=5.25
#
# Phase U adds ONE NEW lever:
#   - PHASED_TTT_PREFIX_DOCS = 2500 → 2800 (more global SGD warmup before later phases)
#
# Hypothesis: more prefix docs in phase 1 = more global SGD updates = smoother
# base_model adaptation. Untested at rank=80; PR #1965 tested 3000 with rank=56
# (regressed) but the rank confound makes that uninformative.
#
# Goals:
#   - mean ttt_bpb < 1.05914 (record bar vs current PR #1855 = 1.06108)
#   - artifact <= 16 MB
# Aborts on seed-1 if BPB > 1.060 (clear regression vs #1953's 1.05855) OR
# artifact > 16 MB.
#
# Robustness (4 prior pod failures taught us):
#   - No 'set -e'
#   - Heartbeat log every 30s
#   - GPU keepalive (defeat any GPU-idle watchdog)
#   - Pod stop ONLY on natural completion (no trap-on-EXIT)
#   - Per-seed CSV uploaded to HF immediately

set -u
set -x

POD_ID=${RUNPOD_POD_ID:-}
HARD_DEADLINE_MIN=${HARD_DEADLINE_MIN:-360}
HB_LOG=/workspace/heartbeat.log

heartbeat_loop() {
  while true; do
    {
      printf '[%s] ' "$(date -u +%H:%M:%S)"
      ps -ef | grep -v grep | grep -cE 'prepare_caseops_data_parallel|train_gpt' | tr -d '\n'
      printf ' procs '
      df -h /workspace 2>/dev/null | tail -1 | awk '{printf "disk=%s ", $5}'
      free -m 2>/dev/null | awk 'NR==2 {printf "mem_used=%dG ", $3/1024}'
      shards=$(ls /workspace/data/sp8192/datasets/*/fineweb_*.bin 2>/dev/null | wc -l)
      printf 'sp8192=%d\n' "$shards"
    } >> $HB_LOG 2>&1
    sleep 30
  done
}
heartbeat_loop &
HB_PID=$!

gpu_keepalive() {
  python3 -u -c "
import time, torch
print('[gpu_keepalive] starting on', torch.cuda.device_count(), 'GPUs', flush=True)
xs = [torch.randn(64, device=f'cuda:{i}') for i in range(torch.cuda.device_count())]
for it in range(99999):
    for x in xs:
        x = (x * 1.001).clamp(-2, 2)
    torch.cuda.synchronize()
    if it % 60 == 0:
        print(f'[gpu_keepalive] tick {it}', flush=True)
    time.sleep(30)
" >> /workspace/keepalive.log 2>&1 &
}
gpu_keepalive
KEEP_PID=$!

( sleep $((HARD_DEADLINE_MIN*60)) && echo "[$(date)] hard deadline" >> $HB_LOG && [ -n "$POD_ID" ] && runpodctl stop pod "$POD_ID" ) &
KILL_PID=$!

final_cleanup() {
  echo "[$(date)] === Phase U done; auto-stop pod $POD_ID ==="
  kill $KILL_PID $HB_PID $KEEP_PID 2>/dev/null || true
  if [ -n "$POD_ID" ]; then
    runpodctl stop pod "$POD_ID" 2>&1 | head -3 || true
    if [ -n "${RUNPOD_API_KEY:-}" ]; then
      curl -sS -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"query\":\"mutation { podStop(input: {podId: \\\"$POD_ID\\\"}) { id } }\"}" 2>&1 | head -3
    fi
  fi
}

# Apt deps for PR #1953 stack
apt-get update -qq 2>&1 | tail -2
apt-get install -y -qq lrzip pixz xz-utils 2>&1 | tail -2 || echo "[WARN] lrzip apt failed"
which lrzip || echo "[WARN] lrzip missing"

export TORCHINDUCTOR_CACHE_DIR=/workspace/torch_inductor
export TRITON_CACHE_DIR=/workspace/triton
export HF_HOME=/workspace/hf
export TOKENIZERS_PARALLELISM=false NCCL_NET=Socket
mkdir -p /workspace/runs $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR $HF_HOME

pip install --break-system-packages --quiet sentencepiece huggingface_hub python-minifier brotli zstandard numpy 2>&1 | tail -3 || \
  echo "[WARN] pip install errored; continuing"

BRANCH=${BRANCH:-submission/pr1797-ngram-mix}
REPO=/workspace/parameter-golf
SUB_PARALLEL=$REPO/records/track_non_record_16mb/2026-04-28_PR1797_EmbedClipRelax_AblationStack
SUB_TRAIN=$REPO/records/track_10min_16mb/2026-04-30_LongCtx_NoQV_QK525_on_1945_1.0586
if [ ! -d "$REPO/.git" ]; then
  rm -rf "$REPO"
  git clone --depth=1 --branch "$BRANCH" https://github.com/Fija/parameter-golf.git "$REPO" 2>&1 | tail -5 || \
    { echo "[FATAL] git clone failed"; exit 1; }
fi

# === Step 1: Download docs_selected.jsonl ===
cd $REPO
DOCS=$REPO/data/docs_selected.jsonl
if [ ! -f "$DOCS" ]; then
  echo "[$(date)] === Downloading docs_selected.jsonl ==="
  python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 0 --with-docs 2>&1 | tail -10 || \
    { echo "[FATAL] docs download failed"; exit 1; }
fi
ls -la "$DOCS" || { echo "[FATAL] docs not found"; exit 1; }

# === Step 2: Retokenize SP8192 ===
DATA_OUT=/workspace/data/sp8192
DATASET_NAME=fineweb10B_sp8192_lossless_caps_caseops_v1_reserved
DATA_PATH_LOCAL=$DATA_OUT/datasets/$DATASET_NAME
if [ ! -f "$DATA_PATH_LOCAL/fineweb_train_000000.bin" ]; then
  rm -rf "$DATA_OUT"
  mkdir -p "$DATA_OUT"
  echo "[$(date)] === Retokenizing SP8192 ==="
  NCPU=$(nproc)
  WORKERS=$(python3 -c "print(min($NCPU - 8, 96))")
  echo "NCPU=$NCPU WORKERS=$WORKERS"
  python3 -u $SUB_PARALLEL/prepare_caseops_data_parallel.py \
    --docs "$DOCS" \
    --out  "$DATA_OUT" \
    --sp   "$SUB_PARALLEL/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model" \
    --val-docs 50000 \
    --workers $WORKERS \
    --chunksize 128 || \
    { echo "[FATAL] retokenize crashed"; ls -la "$DATA_OUT/datasets/" 2>&1; exit 1; }
  # Output dir is hardcoded "fineweb10B_sp8192_..." — already matches
fi
ls "$DATA_PATH_LOCAL" 2>&1 | head -3
du -sh "$DATA_PATH_LOCAL" 2>&1
echo "[$(date)] === retokenize SP8192 DONE ==="

# Stop GPU keepalive — training needs full GPU
kill $KEEP_PID 2>/dev/null || true

# === Step 3: Phase U 3-seed train using PR #1953's train_gpt.py ===
cd "$SUB_TRAIN"

# PR #1953's exact constants (from train_seed42.log hyperparameter dump)
export DATA_PATH=$DATA_PATH_LOCAL
export TOKENIZER_PATH=$SUB_PARALLEL/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
export VOCAB_SIZE=8192 CASEOPS_ENABLED=1
export MAX_WALLCLOCK_SECONDS=600

# PR #1855 9-hparam stack (carried from #1855 → #1908 → #1923 → #1945 → #1953)
export FUSED_CE_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1
export SMEAR_GATE_ENABLED=1 GATE_WINDOW=12
export LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4
export LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64
export TTT_WARM_START_A=1 EMBED_BITS=7 MIN_LR=0.1 MATRIX_LR=0.026
export MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 MLP_CLIP_SIGMAS=11.5
export EMBED_CLIP_SIGMAS=14.0
export GPTQ_RESERVE_SECONDS=4.0 GPTQ_CALIBRATION_BATCHES=16
export PHASED_TTT_NUM_PHASES=3
export WARMDOWN_FRAC=0.85
export BETA2=0.99
export TTT_BETA2=0.99
export TTT_WEIGHT_DECAY=0.5
export TTT_LORA_RANK=80
export SPARSE_ATTN_GATE_SCALE=0.5

# AWQ-lite (#1908) + lrzip pergroup compressor
export AWQ_LITE_ENABLED=1 AWQ_LITE_BITS=8 AWQ_LITE_GROUP_SIZE=64 AWQ_LITE_GROUP_TOP_K=1
export COMPRESSOR=pergroup

# PR #1953's 4 levers
export EVAL_SEQ_LEN=2560
export TTT_EVAL_SEQ_LEN=2560
export TTT_MASK=no_qv
export TTT_Q_LORA=0
export TTT_V_LORA=0
export TTT_LOCAL_LR_MULT=0.75
export QK_GAIN_INIT=5.25

# *** PHASE U NEW LEVER: PREFIX_DOCS 2500 → 2800 ***
export PHASED_TTT_PREFIX_DOCS=2800

# Disable mixers we never use
export NGRAM_MIX_ENABLED=0 TEMP_SCALE_ENABLED=0 PPM_MIX_ENABLED=0

SEEDS=${SEEDS:-"42 314 1234"}
RUN_ID_PREFIX=${RUN_ID_PREFIX:-phu}
CSV=/workspace/runs/${RUN_ID_PREFIX}_summary.csv
echo "seed,pre_quant_bpb,post_quant_bpb,ttt_bpb,artifact_bytes,eval_time_s" > $CSV

ABORT_REASON=""
SEED_NUM=0
for SEED in $SEEDS; do
  SEED_NUM=$((SEED_NUM+1))
  RID=${RUN_ID_PREFIX}_s${SEED}
  RDIR=/workspace/runs/$RID
  rm -rf "$RDIR"; mkdir -p "$RDIR"
  export RUN_ID=$RID SEED=$SEED QUANTIZED_MODEL_PATH=$RDIR/model.bin
  echo "[$(date)] === SEED $SEED ($SEED_NUM/3) ==="
  torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "$RDIR/train_gpt.out" || echo "[WARN] torchrun returned non-zero"
  PRE=$(grep -oE "diagnostic pre-quantization post-ema[^|]*val_bpb:[0-9.]+" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+$' || echo "")
  POST=$(grep -oE "diagnostic quantized[^|]*val_bpb:[0-9.]+" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+$' || echo "")
  TTT=$(grep -oE "quantized_ttt_phased[^|]*val_bpb:[0-9.]+" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+$' || echo "")
  ART=$(grep -oE "Total submission size [^:]+: *[0-9]+ bytes" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9]+' | tail -1 || echo "")
  EV=$(grep -oE "total_eval_time:[0-9.]+s" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+' || echo "")
  echo "$SEED,$PRE,$POST,$TTT,$ART,$EV" >> $CSV
  echo "[$(date)] === SEED $SEED DONE: ttt=$TTT artifact=$ART ==="

  python3 - "$CSV" "$RUN_ID_PREFIX" <<'PY' || echo "[WARN] HF upload failed for this seed"
import sys, os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo_id="FijaEE/parameter-golf-phase-u-results", repo_type="dataset", private=True, exist_ok=True)
api.upload_file(
    path_or_fileobj=open(sys.argv[1], 'rb').read(),
    path_in_repo=f"results/{sys.argv[2]}_summary.csv",
    repo_id="FijaEE/parameter-golf-phase-u-results",
    repo_type="dataset", commit_message=f"{sys.argv[2]} progress",
)
print("uploaded CSV after seed", os.environ.get("SEED"), flush=True)
PY

  if [ $SEED_NUM -eq 1 ] && [ -n "$ART" ]; then
    if [ "$ART" -gt 16000000 ]; then
      ABORT_REASON="seed-1 art $ART > 16M (cap bust)"; echo "[$(date)] === ABORT: $ABORT_REASON ==="; break
    fi
    if [ -n "$TTT" ]; then
      cmp=$(python3 -c "print(int(float('$TTT') > 1.060))")
      [ "$cmp" = "1" ] && ABORT_REASON="seed-1 BPB $TTT > 1.060 (regression)" && echo "[$(date)] === ABORT: $ABORT_REASON ===" && break
    fi
  fi
done

echo
echo "=== PHASE U FINAL (#1953 + prefix=2800) ==="
cat $CSV
[ -n "$ABORT_REASON" ] && echo "ABORTED: $ABORT_REASON"

python3 - <<PY
import csv, math, statistics
rows = list(csv.DictReader(open("$CSV")))
ttt = [float(r["ttt_bpb"]) for r in rows if r.get("ttt_bpb")]
art = [int(r["artifact_bytes"]) for r in rows if r.get("artifact_bytes")]
if len(ttt) >= 2:
    m = statistics.mean(ttt); s = statistics.stdev(ttt)
    print(f"\nmean ttt_bpb = {m:.6f}   std = {s:.6f}   n={len(ttt)}")
    base_m, base_s = 1.06108, 0.00066
    diff = m - base_m
    se = math.sqrt(s**2/len(ttt) + base_s**2/3)
    t = diff / se if se > 0 else 0
    print(f"vs PR #1855 (1.06108+/-0.00066): delta={diff:+.6f}  Welch t={t:.2f}")
    rec_bar = 0.005 / math.log(2) / 3.7266
    print(f"record bar (0.005 nats ~= {rec_bar:.5f} BPB):  {'CLEAR' if -diff > rec_bar else 'MISS'}")
    pr1953_m = 1.05855
    print(f"vs PR #1953 (1.05855): delta={m - pr1953_m:+.6f}")
print()
if art:
    print(f"All artifacts under 16 MB cap: {all(a <= 16_000_000 for a in art)}  (max={max(art):,})")
PY

# Natural success path
final_cleanup

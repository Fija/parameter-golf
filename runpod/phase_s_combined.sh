#!/bin/bash
# Phase S COMBINED — retokenize + train all on the 8×H100 SXM pod.
#
# After three failed attempts to do retokenize on community-cloud CPU pods (one
# hit deadline mid-retokenize, two lost partial work to mysterious pod stops),
# this script does everything on one secure-cloud H100 pod where the disk
# persists and the secure tier is more reliable. Costs more per hour but
# avoids all inter-pod hopping.
#
# Pipeline:
#   1. Download docs_selected.jsonl (~3 min)
#   2. Retokenize SP10240 with caseops_v1_reserved (~30-90 min on H100 pod's CPU)
#   3. Skip HF upload (data stays local on this pod)
#   4. Run Phase S 3-seed (~75 min)
#   5. Auto-stop pod
#
# Cost projection: ~$24/hr × 3h ≈ $72.
set -euo pipefail
set -x

POD_ID=${RUNPOD_POD_ID:-}
HARD_DEADLINE_MIN=${HARD_DEADLINE_MIN:-360}

( sleep $((HARD_DEADLINE_MIN*60)) && [ -n "$POD_ID" ] && runpodctl stop pod "$POD_ID" ) &
KILL_PID=$!

cleanup() {
  kill $KILL_PID 2>/dev/null || true
  echo "[$(date)] === Phase S COMBINED done; auto-stop pod $POD_ID ==="
  if [ -n "$POD_ID" ]; then
    runpodctl stop pod "$POD_ID" 2>&1 | head -3 || true
    if [ -n "${RUNPOD_API_KEY:-}" ]; then
      curl -sS -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"query\":\"mutation { podStop(input: {podId: \\\"$POD_ID\\\"}) { id } }\"}" 2>&1 | head -3
    fi
  fi
}
trap cleanup EXIT

export TORCHINDUCTOR_CACHE_DIR=/workspace/torch_inductor
export TRITON_CACHE_DIR=/workspace/triton
export HF_HOME=/workspace/hf
export TOKENIZERS_PARALLELISM=false NCCL_NET=Socket
mkdir -p /workspace/runs $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR $HF_HOME

pip install --break-system-packages --quiet sentencepiece huggingface_hub python-minifier brotli zstandard numpy 2>&1 | tail -3
which pyminify >/dev/null 2>&1 || true

BRANCH=${BRANCH:-submission/pr1797-ngram-mix}
REPO=/workspace/parameter-golf
SUB=$REPO/records/track_non_record_16mb/2026-04-28_PR1797_EmbedClipRelax_AblationStack
if [ ! -d "$REPO/.git" ]; then
  rm -rf "$REPO"
  git clone --depth=1 --branch "$BRANCH" https://github.com/Fija/parameter-golf.git "$REPO"
else
  (cd "$REPO" && git fetch --depth=1 origin "$BRANCH" && git reset --hard "origin/$BRANCH")
fi

# === Step 1: Download docs_selected.jsonl ===
cd $REPO
DOCS=$REPO/data/docs_selected.jsonl
if [ ! -f "$DOCS" ]; then
  echo "[$(date)] === Downloading docs_selected.jsonl ==="
  python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 0 --with-docs 2>&1 | tail -5
fi
ls -la "$DOCS"

# === Step 2: Retokenize SP10240 ===
DATA_OUT=/workspace/data/sp10240
DATASET_NAME=fineweb10B_sp10240_lossless_caps_caseops_v1_reserved
DATA_PATH_LOCAL=$DATA_OUT/datasets/$DATASET_NAME
if [ ! -f "$DATA_PATH_LOCAL/fineweb_train_000000.bin" ]; then
  rm -rf "$DATA_OUT"
  mkdir -p "$DATA_OUT"
  echo "[$(date)] === Retokenizing SP10240 ==="
  NCPU=$(nproc)
  WORKERS=$(python3 -c "print(min($NCPU - 4, 64))")
  echo "NCPU=$NCPU WORKERS=$WORKERS"
  python3 -u $SUB/prepare_caseops_data_parallel.py \
    --docs "$DOCS" \
    --out  "$DATA_OUT" \
    --sp   "$SUB/tokenizers/fineweb_10240_bpe_lossless_caps_caseops_v1_reserved.model" \
    --val-docs 50000 \
    --workers $WORKERS \
    --chunksize 128
  # Rename hardcoded sp8192 → sp10240
  SRC_DIR="$DATA_OUT/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved"
  if [ -d "$SRC_DIR" ] && [ "$SRC_DIR" != "$DATA_PATH_LOCAL" ]; then
    mv "$SRC_DIR" "$DATA_PATH_LOCAL"
  fi
fi
ls "$DATA_PATH_LOCAL" | head -3
du -sh "$DATA_PATH_LOCAL"

# === Step 3: Run legality test ===
cd "$SUB"
python3 test_ngram_legality.py

# === Step 4: Phase S 3-seed train ===
export DATA_PATH=$DATA_PATH_LOCAL/
export TOKENIZER_PATH=$SUB/tokenizers/fineweb_10240_bpe_lossless_caps_caseops_v1_reserved.model
export VOCAB_SIZE=10240 CASEOPS_ENABLED=1
export MAX_WALLCLOCK_SECONDS=620
export FUSED_CE_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1
export SMEAR_GATE_ENABLED=1 GATE_WINDOW=12
export LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4
export LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64
export TTT_WARM_START_A=1 EMBED_BITS=7 MIN_LR=0.1 MATRIX_LR=0.026
export MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0
export GPTQ_RESERVE_SECONDS=0.5 GPTQ_CALIBRATION_BATCHES=16
export PHASED_TTT_PREFIX_DOCS=2500 PHASED_TTT_NUM_PHASES=3
# Full PR #1855 9-hparam stack + EMBED_CLIP=20 (Phase Q's cap-saver) for SP10240
export MLP_CLIP_SIGMAS=11.5
export EMBED_CLIP_SIGMAS=20.0
export WARMDOWN_FRAC=0.85
export BETA2=0.99
export TTT_BETA2=0.99
export TTT_WEIGHT_DECAY=0.5
export TTT_LORA_RANK=80
export SPARSE_ATTN_GATE_SCALE=0.5
export COMPRESSOR=brotli
export NGRAM_MIX_ENABLED=0 TEMP_SCALE_ENABLED=0 PPM_MIX_ENABLED=0

SEEDS=${SEEDS:-"42 314 1234"}
RUN_ID_PREFIX=${RUN_ID_PREFIX:-phs}
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
  torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "$RDIR/train_gpt.out"
  PRE=$(grep -oE "diagnostic pre-quantization post-ema[^|]*val_bpb:[0-9.]+" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+$' || echo "")
  POST=$(grep -oE "diagnostic quantized[^|]*val_bpb:[0-9.]+" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+$' || echo "")
  TTT=$(grep -oE "quantized_ttt_phased[^|]*val_bpb:[0-9.]+" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+$' || echo "")
  ART=$(grep -oE "Total submission size [^:]+: *[0-9]+ bytes" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9]+' | tail -1 || echo "")
  EV=$(grep -oE "total_eval_time:[0-9.]+s" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+' || echo "")
  echo "$SEED,$PRE,$POST,$TTT,$ART,$EV" >> $CSV
  echo "[$(date)] === SEED $SEED DONE: ttt=$TTT artifact=$ART ==="

  # Upload progress CSV to a fresh tiny HF dataset so we can fetch results even if pod dies.
  python3 - "$CSV" "$RUN_ID_PREFIX" <<'PY' || true
import sys, os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo_id="FijaEE/parameter-golf-phase-s-results", repo_type="dataset", private=True, exist_ok=True)
api.upload_file(
    path_or_fileobj=open(sys.argv[1], 'rb').read(),
    path_in_repo=f"results/{sys.argv[2]}_summary.csv",
    repo_id="FijaEE/parameter-golf-phase-s-results",
    repo_type="dataset", commit_message=f"{sys.argv[2]} progress",
)
PY

  if [ $SEED_NUM -eq 1 ] && [ -n "$ART" ]; then
    if [ "$ART" -gt 16000000 ]; then
      ABORT_REASON="seed-1 art $ART > 16M (cap bust)"; echo "[$(date)] === ABORT: $ABORT_REASON ==="; break
    fi
    if [ -n "$TTT" ]; then
      cmp=$(python3 -c "print(int(float('$TTT') > 1.062))")
      [ "$cmp" = "1" ] && ABORT_REASON="seed-1 BPB $TTT > 1.062" && echo "[$(date)] === ABORT: $ABORT_REASON ===" && break
    fi
  fi
done

echo
echo "=== PHASE S FINAL (SP10240) ==="
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
print()
if art:
    print(f"All artifacts under 16 MB cap: {all(a <= 16_000_000 for a in art)}  (max={max(art):,})")
PY

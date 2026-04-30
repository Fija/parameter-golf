#!/bin/bash
# Phase V (Pod B) — Muon optimizer for TTT LoRA on PR #1953 stack.
# +1 lever vs PR #1953: TTT_OPTIMIZER=muon (instead of adamw), with TTT_MUON_LR_MULT=8x default.
# Hypothesis: Muon's orthogonalized momentum SGD better suits low-rank LoRA than Adam, 1-2 milli-BPB win.
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
      printf ' procs disk='
      df -h /workspace 2>/dev/null | tail -1 | awk '{printf "%s ", $5}'
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
xs = [torch.randn(64, device=f'cuda:{i}') for i in range(torch.cuda.device_count())]
for it in range(99999):
    for x in xs: x = (x * 1.001).clamp(-2, 2)
    torch.cuda.synchronize()
    time.sleep(30)
" >> /workspace/keepalive.log 2>&1 &
}
gpu_keepalive
KEEP_PID=$!

( sleep $((HARD_DEADLINE_MIN*60)) && [ -n "$POD_ID" ] && runpodctl stop pod "$POD_ID" ) &
KILL_PID=$!

final_cleanup() {
  echo "[$(date)] === Phase V done; auto-stop pod $POD_ID ==="
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

apt-get update -qq 2>&1 | tail -1
apt-get install -y -qq lrzip pixz xz-utils 2>&1 | tail -1
which lrzip || echo "[WARN] lrzip missing"

export TORCHINDUCTOR_CACHE_DIR=/workspace/torch_inductor TRITON_CACHE_DIR=/workspace/triton HF_HOME=/workspace/hf
export TOKENIZERS_PARALLELISM=false NCCL_NET=Socket
mkdir -p /workspace/runs $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR $HF_HOME
pip install --break-system-packages --quiet sentencepiece huggingface_hub python-minifier brotli zstandard numpy 2>&1 | tail -2 || true

BRANCH=${BRANCH:-submission/pr1797-ngram-mix}
REPO=/workspace/parameter-golf
SUB_PARALLEL=$REPO/records/track_non_record_16mb/2026-04-28_PR1797_EmbedClipRelax_AblationStack
SUB_TRAIN=$REPO/records/track_10min_16mb/2026-04-30_LongCtx_NoQV_QK525_on_1945_1.0586
if [ ! -d "$REPO/.git" ]; then
  rm -rf "$REPO"
  git clone --depth=1 --branch "$BRANCH" https://github.com/Fija/parameter-golf.git "$REPO" 2>&1 | tail -3 || { echo "[FATAL] clone"; exit 1; }
fi

cd $REPO
DOCS=$REPO/data/docs_selected.jsonl
[ ! -f "$DOCS" ] && { echo "[$(date)] === Downloading docs ==="; python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 0 --with-docs 2>&1 | tail -5 || { echo "[FATAL] docs"; exit 1; }; }

DATA_OUT=/workspace/data/sp8192
DATASET_NAME=fineweb10B_sp8192_lossless_caps_caseops_v1_reserved
DATA_PATH_LOCAL=$DATA_OUT/datasets/$DATASET_NAME
if [ ! -f "$DATA_PATH_LOCAL/fineweb_train_000000.bin" ]; then
  rm -rf "$DATA_OUT"; mkdir -p "$DATA_OUT"
  echo "[$(date)] === Retokenizing SP8192 ==="
  WORKERS=$(python3 -c "print(min($(nproc) - 8, 96))")
  python3 -u $SUB_PARALLEL/prepare_caseops_data_parallel.py \
    --docs "$DOCS" --out "$DATA_OUT" \
    --sp "$SUB_PARALLEL/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model" \
    --val-docs 50000 --workers $WORKERS --chunksize 128 || { echo "[FATAL] retokenize"; exit 1; }
fi
echo "[$(date)] === retokenize SP8192 DONE ==="
kill $KEEP_PID 2>/dev/null || true

cd "$SUB_TRAIN"
export DATA_PATH=$DATA_PATH_LOCAL TOKENIZER_PATH=$SUB_PARALLEL/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
export VOCAB_SIZE=8192 CASEOPS_ENABLED=1 MAX_WALLCLOCK_SECONDS=600
export FUSED_CE_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1 SMEAR_GATE_ENABLED=1 GATE_WINDOW=12
export LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4 LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64
export TTT_WARM_START_A=1 EMBED_BITS=7 MIN_LR=0.1 MATRIX_LR=0.026
export MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 MLP_CLIP_SIGMAS=11.5 EMBED_CLIP_SIGMAS=14.0
export GPTQ_RESERVE_SECONDS=4.0 GPTQ_CALIBRATION_BATCHES=16
export PHASED_TTT_NUM_PHASES=3 PHASED_TTT_PREFIX_DOCS=2500
export WARMDOWN_FRAC=0.85 BETA2=0.99 TTT_BETA2=0.99 TTT_WEIGHT_DECAY=0.5
export TTT_LORA_RANK=80 SPARSE_ATTN_GATE_SCALE=0.5
export AWQ_LITE_ENABLED=1 AWQ_LITE_BITS=8 AWQ_LITE_GROUP_SIZE=64 AWQ_LITE_GROUP_TOP_K=1
export COMPRESSOR=pergroup
# PR #1953's 4 levers
export EVAL_SEQ_LEN=2560 TTT_EVAL_SEQ_LEN=2560
export TTT_MASK=no_qv TTT_Q_LORA=0 TTT_V_LORA=0
export TTT_LOCAL_LR_MULT=0.75 QK_GAIN_INIT=5.25
# *** PHASE V NEW LEVER: Muon for TTT LoRA ***
export TTT_OPTIMIZER=muon
export TTT_MUON_LR_MULT=${TTT_MUON_LR_MULT:-8.0}
export TTT_MUON_BACKEND_STEPS=${TTT_MUON_BACKEND_STEPS:-5}
export NGRAM_MIX_ENABLED=0 TEMP_SCALE_ENABLED=0 PPM_MIX_ENABLED=0

SEEDS=${SEEDS:-"42 314 1234"}
RUN_ID_PREFIX=${RUN_ID_PREFIX:-phv}
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
  torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "$RDIR/train_gpt.out" || echo "[WARN] torchrun rc!=0"
  PRE=$(grep -oE "diagnostic pre-quantization post-ema[^|]*val_bpb:[0-9.]+" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+$' || echo "")
  POST=$(grep -oE "diagnostic quantized[^|]*val_bpb:[0-9.]+" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+$' || echo "")
  TTT=$(grep -oE "quantized_ttt_phased[^|]*val_bpb:[0-9.]+" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+$' || echo "")
  ART=$(grep -oE "Total submission size [^:]+: *[0-9]+ bytes" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9]+' | tail -1 || echo "")
  EV=$(grep -oE "total_eval_time:[0-9.]+s" "$RDIR/train_gpt.out" | tail -1 | grep -oE '[0-9.]+' || echo "")
  echo "$SEED,$PRE,$POST,$TTT,$ART,$EV" >> $CSV
  echo "[$(date)] === SEED $SEED DONE: ttt=$TTT artifact=$ART ==="

  python3 - "$CSV" "$RUN_ID_PREFIX" <<'PY' || echo "[WARN] HF upload"
import sys, os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo_id="FijaEE/parameter-golf-phase-v-results", repo_type="dataset", private=True, exist_ok=True)
api.upload_file(
    path_or_fileobj=open(sys.argv[1], 'rb').read(),
    path_in_repo=f"results/{sys.argv[2]}_summary.csv",
    repo_id="FijaEE/parameter-golf-phase-v-results",
    repo_type="dataset", commit_message=f"{sys.argv[2]} progress",
)
PY

  if [ $SEED_NUM -eq 1 ] && [ -n "$ART" ]; then
    [ "$ART" -gt 16000000 ] && ABORT_REASON="seed-1 art $ART > 16M" && echo "[$(date)] === ABORT: $ABORT_REASON ===" && break
    if [ -n "$TTT" ]; then
      cmp=$(python3 -c "print(int(float('$TTT') > 1.062))")
      [ "$cmp" = "1" ] && ABORT_REASON="seed-1 BPB $TTT > 1.062" && echo "[$(date)] === ABORT: $ABORT_REASON ===" && break
    fi
  fi
done

echo
echo "=== PHASE V FINAL (Muon TTT LoRA) ==="
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
    base_m = 1.06108
    print(f"vs PR #1855 (1.06108): delta={m - base_m:+.6f}")
    print(f"vs record bar (1.05914): {'CLEAR' if m < 1.05914 else 'MISS'}")
    print(f"vs PR #1953 (1.05855): delta={m - 1.05855:+.6f}")
if art:
    print(f"All under 16 MB: {all(a <= 16_000_000 for a in art)} (max={max(art):,})")
PY

final_cleanup

#!/bin/bash
# Phase S retokenize driver — runs on a CPU pod.
#
# Goal: prepare TWO tokenized FineWeb-10B + CaseOps datasets:
#   1. SP10240  (primary — pre-TTT signal -1.5 milli-BPB on Phase L)
#   2. SP4096   (backup if SP10240 cap-busts)
#
# Sequence:
#   - clone repo
#   - download docs_selected.jsonl from HF willdepueoai/parameter-golf
#   - train SP4096 CaseOps tokenizer (~10 min on 48 cores)
#   - retokenize SP10240 (existing .model in repo) -> upload to HF
#   - retokenize SP4096 -> upload to HF
#   - auto-stop pod
#
# Time budget: ~2 hours total (download ~30 min, retokenize ~30 min each, upload ~15 min each).
set -euxo pipefail

POD_ID=${RUNPOD_POD_ID:-}
HARD_DEADLINE_MIN=${HARD_DEADLINE_MIN:-180}

# Hard deadline kill switch.
( sleep $((HARD_DEADLINE_MIN*60)) && [ -n "$POD_ID" ] && runpodctl stop pod "$POD_ID" ) &
KILL_PID=$!

cleanup() {
  kill $KILL_PID 2>/dev/null || true
  echo "[$(date)] === Phase S retokenize done; auto-stop pod $POD_ID ==="
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

WS=/workspace
cd $WS

# Install deps.
pip install --quiet sentencepiece huggingface_hub numpy 2>&1 | tail -3
which runpodctl >/dev/null 2>&1 || (curl -sL https://runpod.io/install-runpodctl | bash 2>&1 | tail -3)
runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null 2>&1 || true

# Clone repo (depth=1 of our branch).
BRANCH=${BRANCH:-submission/pr1797-ngram-mix}
REPO=$WS/parameter-golf
if [ ! -d "$REPO/.git" ]; then
  git clone --depth=1 --branch "$BRANCH" https://github.com/Fija/parameter-golf.git "$REPO"
fi
SUB=$REPO/records/track_non_record_16mb/2026-04-28_PR1797_EmbedClipRelax_AblationStack

# Download docs_selected.jsonl + manifest. cached_challenge_fineweb.py with
# --variant sp1024 --train-shards 0 --with-docs grabs ONLY docs_selected.jsonl
# + manifest + sp1024 val shards + tokenizer (val+tok ~5MB negligible).
# Path resolution in cached_challenge_fineweb places docs_selected.jsonl at
# data/docs_selected.jsonl (REMOTE_ROOT_PREFIX="datasets" is stripped).
cd $REPO
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 0 --with-docs 2>&1 | tail -10
DOCS=""
for cand in \
  "$REPO/data/docs_selected.jsonl" \
  "$REPO/data/datasets/docs_selected.jsonl" \
  "$REPO/docs_selected.jsonl"
do
  if [ -f "$cand" ]; then
    DOCS="$cand"
    break
  fi
done
if [ -z "$DOCS" ]; then
  echo "ERROR: docs_selected.jsonl not found in expected locations; searching..."
  find $REPO -name "docs_selected.jsonl" 2>/dev/null | head -5
  exit 1
fi
ls -la "$DOCS"
echo "DOCS=$DOCS"

# Patch train_sp10240_caseops.py: it imports lossless_caps from a path that doesn't exist on this branch.
PATCHED_TRAIN=$WS/train_caseops_tok.py
cp $REPO/runpod/train_sp10240_caseops.py $PATCHED_TRAIN
# Replace the import-path line — point sys.path at the SUB dir which has lossless_caps.py
python3 -c "
import re
src = open('$PATCHED_TRAIN').read()
src = re.sub(
    r'sys\\.path\\.insert\\(0, str\\(HERE\\.parent / .records. / .track_10min_16mb. / .2026-04-25_PR1797Base_NGramMix.\\)\\)',
    'sys.path.insert(0, \"$SUB\")',
    src,
)
open('$PATCHED_TRAIN', 'w').write(src)
print('patched train_caseops_tok.py')
"
grep -n "sys.path.insert" $PATCHED_TRAIN

# Train SP4096 CaseOps tokenizer (~10 min on 48 cores).
mkdir -p $SUB/tokenizers
SP4096_MODEL=$SUB/tokenizers/fineweb_4096_bpe_lossless_caps_caseops_v1_reserved
if [ ! -f "$SP4096_MODEL.model" ]; then
  echo "[$(date)] === Training SP4096 tokenizer ==="
  python3 $PATCHED_TRAIN \
    --docs "$DOCS" \
    --out_prefix "$SP4096_MODEL" \
    --vocab_size 4096 \
    --num_threads 48 2>&1 | tail -10
fi
ls -la $SP4096_MODEL.model

# Helper: retokenize for given vocab_size + .model file -> upload to HF.
retokenize_and_upload() {
  local VS=$1
  local MODEL=$2
  local REPO_ID=$3
  local OUT=$WS/data/sp${VS}
  rm -rf "$OUT"
  mkdir -p "$OUT"

  echo "[$(date)] === Retokenizing SP$VS ==="
  python3 $SUB/prepare_caseops_data_parallel.py \
    --docs "$DOCS" \
    --out  "$OUT" \
    --sp   "$MODEL" \
    --val-docs 50000 \
    --workers 32 \
    --chunksize 128 2>&1 | tail -20

  # Output dir is hardcoded "fineweb10B_sp8192_..." — rename to vocab-specific name.
  local SRC_DIR="$OUT/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved"
  local DST_DIR="$OUT/datasets/fineweb10B_sp${VS}_lossless_caps_caseops_v1_reserved"
  if [ -d "$SRC_DIR" ] && [ "$SRC_DIR" != "$DST_DIR" ]; then
    mv "$SRC_DIR" "$DST_DIR"
  fi
  ls -la "$DST_DIR" | head -5
  du -sh "$DST_DIR"

  echo "[$(date)] === Uploading SP$VS to HF $REPO_ID ==="
  python3 - <<PY
import os
from pathlib import Path
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo_id="$REPO_ID", repo_type="dataset", private=True, exist_ok=True)
api.upload_large_folder(
    folder_path="$OUT",
    repo_id="$REPO_ID",
    repo_type="dataset",
    num_workers=8,
)
print("upload done")
PY
}

# 1) SP10240 (primary)
retokenize_and_upload 10240 \
  "$SUB/tokenizers/fineweb_10240_bpe_lossless_caps_caseops_v1_reserved.model" \
  "FijaEE/parameter-golf-sp10240-caseops"

# 2) SP4096 (backup) — free up disk first
rm -rf $WS/data/sp10240
retokenize_and_upload 4096 \
  "${SP4096_MODEL}.model" \
  "FijaEE/parameter-golf-sp4096-caseops"

echo "[$(date)] === Phase S retokenize ALL DONE ==="
df -h /workspace 2>&1 | head -3

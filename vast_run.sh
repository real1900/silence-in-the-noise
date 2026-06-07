#!/bin/bash
# ---------------------------------------------------------------------------
# vast_run.sh - rent a GPU on vast.ai, run the 5-fold CV, pull results back,
#               destroy the instance.
#
# Requires:
#   - `vastai` CLI installed (pip install vastai)
#   - VAST_API_KEY exported, OR ~/.config/vastai/vast_api_key written
#
# Behavior:
#   1. Search for cheapest on-demand GPU with >= 16 GB VRAM
#   2. Create the instance with the pytorch:2.4.0-cuda12.1 image
#   3. Wait for it to come online (~2-3 min)
#   4. Upload the project, pip install deps, run the sweep
#   5. Rsync results back to ./paper_assets/
#   6. Destroy the instance
#
# Usage:
#   bash vast_run.sh                     # cheapest GPU, ~$0.20-0.40/hr
#   bash vast_run.sh --gpu 4090          # prefer 4090 (faster, $0.30-0.60/hr)
#   bash vast_run.sh --epochs 5          # tighter time budget
#   bash vast_run.sh --dry-run           # search only, don't rent
# ---------------------------------------------------------------------------
set -euo pipefail

EPOCHS=10
SWEEP="0,4"
TRAIN_BATCH=16            # real GPU has more memory than M1 Max
SEED=42
GPU_FILTER='gpu_name:RTX_3090|gpu_name:RTX_4090|gpu_name:A100'
GPU_PREF=""
DRY_RUN=0
MAX_PRICE=0.50            # USD/hr cap

while [[ $# -gt 0 ]]; do
  case "$1" in
    --epochs)      EPOCHS="$2"; shift 2;;
    --sweep)       SWEEP="$2"; shift 2;;
    --train-batch) TRAIN_BATCH="$2"; shift 2;;
    --seed)        SEED="$2"; shift 2;;
    --gpu)         GPU_PREF="$2"; shift 2;;
    --max-price)   MAX_PRICE="$2"; shift 2;;
    --dry-run)     DRY_RUN=1; shift;;
    -h|--help)     sed -n '2,30p' "$0"; exit 0;;
    *)             echo "Unknown arg: $1"; exit 1;;
  esac
done

# Resolve API key
if [[ -z "${VAST_API_KEY:-}" ]] && [[ -f "$HOME/.config/vastai/vast_api_key" ]]; then
  export VAST_API_KEY=$(cat "$HOME/.config/vastai/vast_api_key")
fi
if [[ -z "${VAST_API_KEY:-}" ]]; then
  echo "ERROR: set VAST_API_KEY env var or write it to ~/.config/vastai/vast_api_key"
  exit 1
fi
vastai set api-key "$VAST_API_KEY" 2>&1 | tail -1

# Choose GPU model preference
if [[ -n "$GPU_PREF" ]]; then
  GPU_FILTER="gpu_name:$GPU_PREF"
fi

cat <<EOF
================================================================
vast.ai launch plan
  api key set    : yes
  gpu filter     : $GPU_FILTER
  max price/hr   : \$$MAX_PRICE
  epochs         : $EPOCHS
  sweep (n_reg)  : $SWEEP
  train_batch    : $TRAIN_BATCH
================================================================
EOF

# 1. Search for the cheapest qualifying offer
echo ">>> searching offers ..."
SEARCH_QUERY="reliability>0.97 num_gpus=1 gpu_ram>=16 compute_cap>=800 compute_cap<=900 inet_down>100 inet_up>100 disk_space>=40 dph<$MAX_PRICE"
[[ -n "$GPU_PREF" ]] && SEARCH_QUERY+=" gpu_name=$GPU_PREF"

vastai search offers "$SEARCH_QUERY" -o dph 2>&1 | head -10

if [[ $DRY_RUN -eq 1 ]]; then
  echo "(dry-run) - exiting before rental"
  exit 0
fi

# Pick the cheapest offer ID
OFFER_ID=$(vastai search offers "$SEARCH_QUERY" -o dph --raw 2>&1 | python3 -c "
import sys, json
offers = json.load(sys.stdin)
if not offers: sys.exit('no offers match the filter')
o = offers[0]
print(o['id'])
print(f\"  picked: id={o['id']} gpu={o.get('gpu_name')} dph=\${o.get('dph_total',0):.3f} ram={o.get('gpu_ram',0)/1024:.0f}GB\", file=sys.stderr)
" )
echo "$OFFER_ID" | head -1
PICKED_ID=$(echo "$OFFER_ID" | head -1)

# 2. Create instance
echo ">>> renting offer $PICKED_ID ..."
INSTANCE_INFO=$(vastai create instance "$PICKED_ID" \
    --image pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel \
    --disk 40 --ssh --raw 2>&1)
echo "$INSTANCE_INFO" | tail -3
INSTANCE_ID=$(echo "$INSTANCE_INFO" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(data.get('new_contract', data.get('contract_id', '')))
")
if [[ -z "$INSTANCE_ID" ]]; then
  echo "ERROR: could not parse INSTANCE_ID from rental response"
  exit 1
fi
echo "INSTANCE_ID=$INSTANCE_ID"

trap "echo '>>> destroying instance $INSTANCE_ID'; yes | vastai destroy instance $INSTANCE_ID 2>&1 || true" EXIT

# 3. Wait for instance to be reachable
echo ">>> waiting for instance to come online (~2-3 min) ..."
for i in {1..40}; do
  INFO=$(vastai show instance "$INSTANCE_ID" --raw 2>&1)
  STATUS=$(echo "$INFO" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('actual_status','unknown'))")
  echo "  attempt $i status=$STATUS"
  if [[ "$STATUS" == "running" ]]; then
    SSH_HOST=$(echo "$INFO" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('ssh_host',''))")
    SSH_PORT=$(echo "$INFO" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('ssh_port',''))")
    echo "  ssh root@$SSH_HOST -p $SSH_PORT"
    break
  fi
  sleep 15
done

if [[ -z "${SSH_HOST:-}" ]]; then
  echo "ERROR: instance never came online"
  exit 1
fi

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10"

# 4a. Wait for SSH to actually accept connections
echo ">>> waiting for SSH to accept connections ..."
for i in {1..20}; do
  if ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" 'echo ssh-ready' 2>/dev/null | grep -q ssh-ready; then
    echo "  SSH ready on attempt $i"
    break
  fi
  if [[ $i -eq 20 ]]; then
    echo "ERROR: SSH never came up after 20 attempts"
    exit 1
  fi
  sleep 10
done

# 4b. Upload project + install deps + run training
HERE=$(cd "$(dirname "$0")"; pwd)
echo ">>> uploading code ..."
scp $SSH_OPTS -P "$SSH_PORT" \
    "$HERE/train_full_5fold.py" "$HERE/measure_real_results.py" \
    "root@$SSH_HOST":/root/

echo ">>> installing python deps + running training ..."
ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" "
  set -e
  cd /root
  # libsndfile is the only native dep we still need (for soundfile)
  DEBIAN_FRONTEND=noninteractive apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libsndfile1 > /dev/null
  # Use the image's default torch (2.4 cu121) - we don't need newer; we just need
  # transformers + datasets + soundfile, and our training script decodes audio
  # manually so torchcodec is not required.
  pip install --quiet --upgrade pip
  pip install --quiet 'transformers>=4.45,<6' 'datasets>=2.20,<5' soundfile
  python3 -c 'import torch; print(\"torch:\", torch.__version__, \"cuda:\", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'
  mkdir -p paper_assets
  set -o pipefail
  python3 train_full_5fold.py --epochs $EPOCHS --sweep $SWEEP --train-batch $TRAIN_BATCH --seed $SEED --folds all --num-workers 2 2>&1 | tee train.log
"

# 5. Pull results back (glob over real_5fold*.json/.npz so seed-suffixed names work)
echo ">>> rsyncing results back ..."
mkdir -p "$HERE/paper_assets"
ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" \
    'ls /root/paper_assets/real_5fold* 2>/dev/null || true'
scp $SSH_OPTS -P "$SSH_PORT" \
    "root@$SSH_HOST:/root/paper_assets/real_5fold*" \
    "root@$SSH_HOST":/root/train.log \
    "$HERE/paper_assets/" || true

# instance teardown happens via the trap
echo ">>> done. results in $HERE/paper_assets/real_5fold.json"

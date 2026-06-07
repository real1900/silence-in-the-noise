#!/bin/bash
# ---------------------------------------------------------------------------
# gcp_run.sh - provision an L4/T4/A100 GPU instance, run the 5-fold CV,
#              pull the results back, and tear the instance down.
#
# Run from your laptop. Requires:
#   - gcloud CLI installed and authenticated (run `gcloud auth login`)
#   - $GCP_PROJECT exported (or pass --project on the command line)
#   - $50 free-tier GCP credits is enough; total cost ~$2-8
#
# What it does:
#   1. creates a deep-learning VM with a GPU (default: 1x L4, 24 GB)
#   2. uploads the project
#   3. installs requirements
#   4. runs train_full_5fold.py
#   5. rsync's results back to ./paper_assets/
#   6. deletes the instance
#
# Usage:
#   bash gcp_run.sh                       # full run (1.5-2 hr, ~$2-4 spot)
#   bash gcp_run.sh --gpu a100            # use A100 (faster, more expensive)
#   bash gcp_run.sh --gpu t4 --no-spot    # T4 on-demand (cheapest, slowest)
#   bash gcp_run.sh --dry-run             # print plan, do nothing
# ---------------------------------------------------------------------------
set -euo pipefail

# ---- defaults ------------------------------------------------------------
INSTANCE="${INSTANCE:-ast-registers-$(date +%Y%m%d-%H%M%S)}"
ZONE="${ZONE:-us-central1-a}"
PROJECT="${GCP_PROJECT:-}"
GPU_TYPE="v100"        # v100 | l4 | t4 | a100 | p100 (only v100 + p100 are pre-quota'd in your project)
USE_SPOT="--provisioning-model=SPOT --instance-termination-action=DELETE"
DRY_RUN=0
EPOCHS=10
SWEEP="0,4"
TRAIN_BATCH=8                # V100 (16GB) tight at 16; bump to 16 on L4/A100

# ---- argument parsing ----------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project)     PROJECT="$2"; shift 2;;
    --zone)        ZONE="$2"; shift 2;;
    --instance)    INSTANCE="$2"; shift 2;;
    --gpu)         GPU_TYPE="$2"; shift 2;;
    --no-spot)     USE_SPOT=""; shift;;
    --epochs)      EPOCHS="$2"; shift 2;;
    --sweep)       SWEEP="$2"; shift 2;;
    --train-batch) TRAIN_BATCH="$2"; shift 2;;
    --dry-run)     DRY_RUN=1; shift;;
    -h|--help)     sed -n '2,30p' "$0"; exit 0;;
    *)             echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$PROJECT" ]]; then
  PROJECT=$(gcloud config get-value project 2>/dev/null || true)
  if [[ -z "$PROJECT" ]]; then
    echo "ERROR: set --project or 'gcloud config set project YOUR_PROJECT'"
    exit 1
  fi
fi

# ---- machine type and image based on GPU --------------------------------
case "$GPU_TYPE" in
  v100)
    MACHINE="n1-standard-8"          # 8 vCPU / 30 GB RAM / 1x V100 (16GB)
    ACCEL="type=nvidia-tesla-v100,count=1"
    ;;
  p100)
    MACHINE="n1-standard-8"          # 8 vCPU / 30 GB RAM / 1x P100 (16GB) - slower but works
    ACCEL="type=nvidia-tesla-p100,count=1"
    ;;
  l4)
    MACHINE="g2-standard-8"          # 8 vCPU / 32 GB RAM / 1x L4 (24GB) - REQUIRES QUOTA
    ACCEL="type=nvidia-l4,count=1"
    ;;
  t4)
    MACHINE="n1-standard-8"          # 8 vCPU / 30 GB RAM / 1x T4 (16GB) - REQUIRES QUOTA
    ACCEL="type=nvidia-tesla-t4,count=1"
    ;;
  a100)
    MACHINE="a2-highgpu-1g"          # 12 vCPU / 85 GB RAM / 1x A100 (40GB) - REQUIRES QUOTA
    ACCEL="type=nvidia-tesla-a100,count=1"
    ;;
  *)
    echo "Unknown --gpu '$GPU_TYPE'. Use v100 | p100 | l4 | t4 | a100."; exit 1;;
esac

IMAGE="--image-family=common-cu129-ubuntu-2204-nvidia-580 --image-project=deeplearning-platform-release"

cat <<EOF
================================================================
GCP launch plan
  project       : $PROJECT
  zone          : $ZONE
  instance      : $INSTANCE
  gpu           : $GPU_TYPE  ($ACCEL)
  machine       : $MACHINE
  spot          : $([ -n "$USE_SPOT" ] && echo yes || echo no)
  epochs        : $EPOCHS
  sweep (n_reg) : $SWEEP
  train_batch   : $TRAIN_BATCH
================================================================
EOF

if [[ $DRY_RUN -eq 1 ]]; then
  echo "(dry-run) skipping all gcloud calls"
  exit 0
fi

# ---- 1. create the instance ---------------------------------------------
echo ">>> creating instance $INSTANCE ..."
gcloud compute instances create "$INSTANCE" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type="$MACHINE" \
  --accelerator="$ACCEL" \
  $IMAGE \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True" \
  --boot-disk-size=80GB \
  --boot-disk-type=pd-balanced \
  $USE_SPOT \
  --scopes=cloud-platform

# automatically tear down on script exit (success or failure)
trap "echo '>>> tearing down $INSTANCE'; gcloud compute instances delete $INSTANCE --zone=$ZONE --quiet 2>/dev/null || true" EXIT

# wait for SSH to be ready
echo ">>> waiting for SSH ..."
for i in {1..30}; do
  if gcloud compute ssh "$INSTANCE" --zone="$ZONE" --command="echo ready" --quiet 2>/dev/null; then
    break
  fi
  sleep 10
done

# wait for nvidia-smi to be ready (driver finishes installing)
echo ">>> waiting for GPU driver ..."
for i in {1..30}; do
  if gcloud compute ssh "$INSTANCE" --zone="$ZONE" --command="nvidia-smi -L" --quiet 2>/dev/null; then
    gcloud compute ssh "$INSTANCE" --zone="$ZONE" --command="nvidia-smi -L"
    break
  fi
  sleep 15
done

# ---- 2. upload the project ----------------------------------------------
echo ">>> uploading project ..."
HERE=$(cd "$(dirname "$0")"; pwd)
gcloud compute scp --zone="$ZONE" --recurse \
  "$HERE/train_full_5fold.py" \
  "$HERE/measure_real_results.py" \
  "$INSTANCE":~/

# ---- 3. install dependencies --------------------------------------------
echo ">>> installing python deps ..."
gcloud compute ssh "$INSTANCE" --zone="$ZONE" --command='
  set -e
  python3 -m venv ~/venv
  source ~/venv/bin/activate
  pip install --upgrade pip
  # match the local environment that produced measure_real_results.py
  pip install --quiet \
      torch==2.10.0 torchaudio==2.10.0 \
      transformers==5.3.0 datasets==4.7.0 soundfile librosa \
      "numpy<3"
  python3 -c "import torch; print(\"torch:\", torch.__version__, \"cuda:\", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"\")"
'

# ---- 4. run training ----------------------------------------------------
echo ">>> running training (this is the long step) ..."
gcloud compute ssh "$INSTANCE" --zone="$ZONE" --command="
  source ~/venv/bin/activate
  mkdir -p paper_assets
  python3 train_full_5fold.py \
    --epochs $EPOCHS \
    --sweep $SWEEP \
    --train-batch $TRAIN_BATCH \
    --folds all 2>&1 | tee train.log
"

# ---- 5. pull results back -----------------------------------------------
echo ">>> rsyncing results back ..."
mkdir -p "$HERE/paper_assets"
gcloud compute scp --zone="$ZONE" --recurse \
  "$INSTANCE":~/paper_assets/real_5fold.json \
  "$INSTANCE":~/paper_assets/real_5fold_preds.npz \
  "$INSTANCE":~/train.log \
  "$HERE/paper_assets/" || true

# instance teardown happens via the trap above
echo ">>> done. results in $HERE/paper_assets/real_5fold.json"

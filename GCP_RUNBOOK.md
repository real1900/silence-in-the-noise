# GCP Runbook: ESC-50 5-fold CV for AST + Register Tokens

Total wall-clock: ~80 min on V100 spot. Cost estimate: **~$1** of your $50 credit.

> **Quota status (verified for project `submission-app-35eaa`):**
> - V100: limit=1 ✓ (default - launch immediately)
> - P100: limit=1 ✓ (slower fallback)
> - L4 / T4 / A100: limit=0 (would need a quota request, ~5–60 min wait)
>
> The script defaults to V100 spot, so you can launch right now without
> any quota request. The previous L4-based estimate is preserved below in
> case you ever want a faster (and cheaper-per-hour) GPU.

## 0. One-time prerequisites

```bash
# install gcloud CLI if you don't have it
brew install --cask google-cloud-sdk     # macOS

# log in (opens browser)
gcloud auth login
gcloud auth application-default login

# pick / create your project (only billing-enabled ones can launch VMs)
gcloud projects list
gcloud config set project YOUR_PROJECT_ID
```

If this is your first GCP GPU run, you almost certainly need to **request a
GPU quota increase**. Check current limits:

```bash
gcloud compute regions describe us-central1 --format="yaml(quotas)" \
  | grep -A1 -i 'gpu\|nvidia'
```

Look for the line `NVIDIA_L4_GPUS` (or `NVIDIA_T4_GPUS` / `NVIDIA_A100_GPUS`).
If `limit: 0`, request a bump:

1. Go to https://console.cloud.google.com/iam-admin/quotas
2. Filter: `Service = Compute Engine API`, `Type = Quota`, search "L4"
3. Edit → request `1` for `us-central1`
4. Submit. Approval is usually < 1 hour for new academic accounts; can be < 5 min.

> **Tip:** L4 (24 GB VRAM) is the cheapest GPU that comfortably fits AST-Base
> at batch 16. T4 also works (16 GB) at batch 8. A100 is overkill but ~3×
> faster — only worth it if you want to finish in 30 min instead of 2 hours.

## 1. Smoke-test the training script locally first (already done)

```bash
cd "$HOME/Documents/Developer/JHU/EN.705.744.8VL.SP26 Deep Learning Using Transformers/research paper"
source .venv/bin/activate
python3 train_full_5fold.py --smoke
```

Expected: completes in ~5 minutes on MPS, prints `final test_acc≈0.50–0.70`
(1 epoch is undertrained but proves the script runs end-to-end).

## 2. Launch the GCP run

> **Important: run inside tmux or screen** so the orchestrator survives a
> dropped SSH or laptop sleep. If the orchestrator dies mid-run, the GCP
> instance keeps running and you keep getting billed.

```bash
# install tmux if needed
brew install tmux

# new tmux session
tmux new -s ast-run

# inside tmux:
cd "$HOME/Documents/Developer/JHU/EN.705.744.8VL.SP26 Deep Learning Using Transformers/research paper"
bash gcp_run.sh 2>&1 | tee run.log

# detach with Ctrl-b then d  (session keeps running)
# re-attach later: tmux attach -t ast-run
```

The single command without tmux (only safe if you'll keep the laptop awake):

```bash
bash gcp_run.sh                           # default: L4 spot, full sweep
```

What it does (visible in real time):

1. Spins up an `g2-standard-8` VM with 1× NVIDIA L4 in `us-central1-a`
   (spot pricing: ~$0.28/hr at time of writing)
2. Uploads `train_full_5fold.py` and `measure_real_results.py`
3. Creates a Python venv and installs `torch`, `transformers`, `datasets`,
   `soundfile`, `librosa`
4. Runs `python3 train_full_5fold.py --epochs 10 --sweep 0,4 --train-batch 16 --folds all`
   (this is the long step — ~1.5–2 hours)
5. Rsyncs `paper_assets/real_5fold.json` and `real_5fold_preds.npz` back
6. **Deletes the instance** (via `trap … EXIT`) so you don't get charged
   for an idle VM if the script crashes

### Useful variants

```bash
bash gcp_run.sh --gpu a100                # ~30 min, $2 spot, faster turnaround
bash gcp_run.sh --gpu t4 --no-spot        # cheapest, slowest, most available
bash gcp_run.sh --epochs 5 --sweep 0,4    # if you want a faster (slightly
                                          # noisier) result and still have
                                          # time for one re-run
bash gcp_run.sh --dry-run                 # print plan, don't launch anything
```

### What to expect on success

- `paper_assets/real_5fold.json` — per-fold accuracies, per-epoch dynamics
- `paper_assets/real_5fold_preds.npz` — per-clip predictions (for bootstrap)
- `paper_assets/train.log` — full training log
- The VM is automatically destroyed; check the Cloud Console "Instances"
  page to confirm $0 ongoing spend

### If it gets pre-empted (spot instance)

L4 spot has ~10–20% pre-emption rate. The script writes intermediate
results after each (n_reg, fold) pair, so you can resume by re-running
`bash gcp_run.sh` and the script will pick up from where it left off.

If you want zero pre-emption risk for a slightly higher cost (~3×),
add `--no-spot` to the command.

## 3. Analyze and patch the paper

Once results are back:

```bash
python3 analyze_5fold.py                  # bootstrap CI + permutation test
```

This prints LaTeX-ready strings that you (or I) can paste into the paper's
abstract, Results section, and Discussion. After that:

```bash
# I'll handle this step - patch:
#   - the abstract's "H3 question remains open" -> real numbers
#   - Section V-C "Compute Budget" -> reframed in confident voice
#   - Section VI-B "Why Doesn't AST Show the Artifact" -> unchanged
#   - bibliography -> add 1-2 sentences referencing the proper protocol
```

Then recompile the LaTeX (Overleaf or your local pipeline).

## 3a. Emergency cleanup (if anything goes wrong)

If `gcp_run.sh` crashes or your laptop dies mid-run, the GCP VM may
still be running and incurring spot charges. Always verify with:

```bash
gcloud compute instances list
```

If you see an `ast-registers-*` instance, kill it:

```bash
gcloud compute instances delete INSTANCE_NAME --zone=us-central1-a
```

The script normally tears the instance down on exit (via `trap … EXIT`),
but a SIGKILL or laptop power-off bypasses the trap.

## 4. Cost sanity check

After the run finishes:

```bash
gcloud billing accounts list
# go to https://console.cloud.google.com/billing for actual numbers
```

If you spent more than $10, something went wrong (probably a runaway
instance). Check the "Compute Engine -> VM instances" page; there should
be zero running instances.

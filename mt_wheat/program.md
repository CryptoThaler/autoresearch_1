# mt_wheat — Autonomous Yield Prediction Research

Montana wheat yield prediction using the autoresearch protocol.
Adapted from Karpathy's autoresearch / autoresearch-mlx.

**Metric:** val_r2 (higher is better).  Target: 0.60+
**Time budget:** 300 seconds (5 minutes) per experiment.
**Hardware:** M4 Pro, 24 GB unified memory, MPS acceleration.

## Setup

Work with the user to:

1. **Agree on a run tag**: propose a tag (e.g. `wheat-mar15`).
   Create branch: `git checkout -b autoresearch/<tag>`
2. **Read the in-scope files**:
   - `config.py` — fixed constants, API endpoints, feature dimensions.
     Do not modify.
   - `prepare.py` — fixed data pipeline and evaluation functions.
     Do not modify.
   - `train.py` — the file you modify.  Model architecture, optimizer,
     training loop, hyperparameters.
3. **Verify data exists**: Check that `data_cache/` contains
   `temporal_train.pt`, `static_train.pt`, `targets_train.pt` and
   their val counterparts.  If not, tell the human to run
   `python prepare.py`.
4. **Initialize results.tsv** with header and baseline entry.
   Run `python train.py` once to establish YOUR baseline on this
   hardware.  Do NOT use numbers from other platforms.
5. **Confirm and go**: Once setup looks good and you get confirmation,
   kick off the experimentation loop.

## Experimentation

Each experiment trains on Apple Silicon via PyTorch MPS.  The training
script runs for a **fixed 5-minute budget** (wall clock training time,
excluding data loading and evaluation).

Launch: `python train.py`

**What you CAN do:**
- Modify `train.py` — everything is fair game: model architecture,
  optimizer, hyperparameters, training loop, batch size, model size,
  feature engineering within the loaded tensors, regularization,
  learning rate schedules, etc.

**What you CANNOT do:**
- Modify `prepare.py`.  It is read-only.  It contains the fixed
  evaluation, data loading, and constants.
- Modify `config.py`.  Fixed constants and API configuration.
- Install new packages beyond what is in pyproject.toml.
- Modify the evaluation harness.  The `evaluate_model()` function
  in `prepare.py` is ground truth.

**The goal: get the highest val_r2.**

Since the time budget is fixed, you do not need to worry about
training time — it is always 5 minutes.  Everything is fair game:
change the architecture, the optimizer, the hyperparameters, the
batch size, the model size.  The only constraint is that the code
runs without crashing and finishes within the time budget.

**Memory** is a soft constraint.  MPS uses unified memory shared with
the system.  Some increase is acceptable for meaningful val_r2 gains,
but it should not blow up.

**Simplicity criterion**: All else equal, simpler is better.  A small
improvement adding ugly complexity is not worth it.  Removing something
and getting equal or better results is a great outcome.  Weigh
complexity cost against improvement magnitude.

**The first run**: Always establish the baseline first by running
train.py as-is.

## Output format

The script prints a summary like:

```
---
val_r2: 0.423000
val_rmse: 5.234000
val_mae: 4.012000
training_seconds: 300.0
total_seconds: 312.4
peak_memory_mb: 1024.0
num_epochs: 847
num_params: 134100
```

Compare only val_r2 against your own baseline on the same hardware.

## Logging results

Log to `results.tsv` (tab-separated).  Header and 6 columns:

```
commitval_r2val_rmsememory_mbstatusdescription
```

1. git commit hash (short, 7 chars)
2. val_r2 achieved (e.g. 0.423000) — use 0.000000 for crashes
3. val_rmse achieved
4. peak memory in MB
5. status: `keep`, `discard`, or `crash`
6. short text description

Example:
```
commitval_r2val_rmsememory_mbstatusdescription
a1b2c3d-0.21744.10512keepbaseline
e4f5g6h0.15238.20520keepadd batch norm and reduce LR to 1e-3
i7j8k9l0.08940.50518discardtry transformer encoder (worse)
```

## The experiment loop

Runs on a dedicated branch (e.g. `autoresearch/wheat-mar15`).

LOOP FOREVER:

1. Look at git state: current branch/commit.
2. Tune `train.py` with an experimental idea.
3. `git add mt_wheat/train.py && git commit -m "experiment: <desc>"`
   (never `git add -A` — this is inside a larger repo)
4. Run: `python train.py > run.log 2>&1`
5. Read results: `grep "^val_r2:\|^val_rmse:\|^peak_memory_mb:" run.log`
6. If grep is empty, the run crashed.  `tail -n 50 run.log` for traceback.
7. Record in results.tsv.
8. If val_r2 improved (higher):
   `git add mt_wheat/results.tsv && git commit --amend --no-edit`
9. If val_r2 equal or worse:
   Record the discard hash, then `git reset --hard <prev kept commit>`

## Ideas to try (starting points)

Architecture:
- Add batch normalization after each conv/linear layer
- Try different kernel sizes (3, 5, 7) in temporal CNN
- Use bidirectional LSTM instead of CNN for temporal branch
- Add attention mechanism over temporal steps
- Try SE (squeeze-excitation) blocks
- Residual connections in the fusion head

Optimization:
- Lower learning rate (1e-4, 5e-4 instead of 1e-3)
- Cosine annealing schedule
- Warm restarts (CosineAnnealingWarmRestarts)
- AdamW with weight decay tuning
- Gradient clipping

Regularization:
- Dropout (0.1 to 0.5, different rates per branch)
- Weight decay (1e-4 to 1e-2)
- Early stopping within the 5-min budget
- Data augmentation (gaussian noise on inputs)

Feature engineering (within train.py):
- Normalize inputs to zero mean, unit variance
- Interaction features between temporal and static branches
- Polynomial features for key weather variables
- Running averages / differences of temporal features

## Timeout and crashes

Each experiment: ~6 min total (5 min train + eval overhead).
If a run exceeds 10 minutes, kill it — treat as failure.

Crashes: fix typos/imports quickly.  If the idea is fundamentally
broken, log "crash", revert, move on.

## NEVER STOP

Once the loop begins, do NOT pause to ask the human.  Do NOT ask
"should I keep going?" or "is this a good stopping point?".  The
human might be asleep.  You are autonomous.  If you run out of ideas,
think harder — re-read the files, try combining near-misses, try
more radical changes.  The loop runs until manually interrupted.

Overnight (8 hours) = ~80 experiments.
Weekend (48 hours) = ~480 experiments.

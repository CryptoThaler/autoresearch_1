# Quickstart: Montana Wheat Yield Prediction on M4 Pro

This is the single-file guide to get the autonomous research loop
running on your Apple Silicon Mac.  It covers setup, first run, and
the Claude Code integration.

---

## What this project does

An AI agent (Claude Code) autonomously improves a wheat yield
prediction model.  It edits `train.py`, runs a 5-minute training
experiment, measures val_r2, keeps improvements, reverts failures,
and loops — indefinitely — until you stop it.

The goal is to push val_r2 from the current baseline (~-0.2) into
the competitive range (0.6 – 0.8) through hundreds of automated
architecture and hyperparameter experiments.

---

## Two phases

### Phase 1 — Validate MLX (Option A)

This confirms your M4 Pro can run Apple Silicon ML workloads.
Clone the upstream autoresearch-mlx NLP experiment and run one
5-minute training cycle:

```bash
# Install uv (package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.zshrc

# Clone and run
cd ~/Projects
git clone https://github.com/trevin-creator/autoresearch-mlx.git
cd autoresearch-mlx
uv sync
uv run prepare.py --num-shards 10   # downloads ~2 GB of text data
uv run train.py                      # 5-min GPT training experiment
```

Success: you see `val_bpb: X.XX` printed at the end (expect ~1.8).
This proves MLX, uv, and Apple Silicon training all work.
Takes about 30 minutes total including downloads.

### Phase 2 — Launch wheat yield research (Option C)

This is the real project.  It uses PyTorch with MPS (not MLX)
because the yield prediction task needs different data pipelines
and model architectures than language modeling.

```bash
# Clone the repo
cd ~/Projects
git clone https://github.com/CryptoThaler/autoresearch_1.git
cd autoresearch_1/mt_wheat

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy requests pandas scipy

# Set API keys (add to ~/.zshrc for persistence)
export NASS_API_KEY=9F1B344A-CA91-36EE-997E-7F4C7F750432
export EARTHDATA_USER=<your_earthdata_username>
export EARTHDATA_PASS=<your_earthdata_password>

# Pull all data (one-time, ~20 minutes)
python prepare.py

# Establish your baseline
python train.py
```

The baseline run trains for 5 minutes and prints:
```
---
val_r2: <your_baseline>
val_rmse: <your_baseline>
...
```

Note these numbers.  They are your hardware-specific starting point.

---

## Launch the autonomous loop with Claude Code

Claude Code is a terminal-based AI agent that reads `program.md`
and runs the experiment loop on your machine.

```bash
# Create a fresh experiment branch
git checkout -b autoresearch/wheat-v1

# Open Claude Code and point it at the protocol
# (In Claude Code, tell it to read mt_wheat/program.md)
```

### What happens next

Claude Code will:
1. Read `program.md`, `train.py`, `config.py`, and `prepare.py`
2. Establish its baseline by running `python train.py`
3. Propose a change to `train.py` (architecture, hyperparameters, etc.)
4. Commit the change, run the experiment, read the results
5. If val_r2 improved: keep.  If worse: revert.
6. Log the result to `results.tsv`
7. Go to step 3.  Repeat forever.

### It does not stop on its own

The protocol explicitly instructs the agent to loop indefinitely.
It will not ask "should I keep going?" or pause for confirmation.
You can walk away, sleep, or leave for the weekend.

Each experiment takes ~6 minutes.  Overnight (8 hours) = ~80
experiments.  A full weekend = ~480 experiments.

### How to stop it

Close the Claude Code terminal, or press Ctrl+C.

---

## Monitoring while it runs

In a separate terminal:

```bash
# Watch results in real time
tail -f mt_wheat/results.tsv

# Count total experiments
wc -l mt_wheat/results.tsv

# See best val_r2 so far
sort -t$'\t' -k2 -rn mt_wheat/results.tsv | head -5

# Check git history
git log --oneline -20
```

---

## File roles

| File | Role | Editable? |
|------|------|-----------|
| `train.py` | Model, optimizer, training loop | AGENT EDITS THIS |
| `prepare.py` | Data pipeline + evaluation functions | READ-ONLY |
| `config.py` | Constants, API endpoints, FIPS codes | READ-ONLY |
| `program.md` | Autonomous agent protocol | READ-ONLY |
| `results.tsv` | Experiment log | AGENT APPENDS |
| `pyproject.toml` | Dependencies | READ-ONLY |

---

## What the agent can change in train.py

Everything is fair game:
- Model architecture (CNN depth, kernel sizes, LSTM, attention)
- Fusion strategy (concat, gating, cross-attention)
- Optimizer (Adam, AdamW, SGD, learning rate schedules)
- Regularization (dropout, weight decay, batch norm)
- Feature engineering (normalization, interactions, polynomials)
- Batch size, number of epochs within the 5-minute budget

---

## Expected progression

| Phase | val_r2 | What typically helps |
|-------|--------|---------------------|
| Baseline | -0.2 to 0.1 | Untuned model |
| Quick wins | 0.1 to 0.3 | LR tuning, normalization |
| Solid | 0.3 to 0.5 | Architecture improvements |
| Competitive | 0.5 to 0.7 | Feature interactions, ensembles |
| Research grade | 0.7+ | Novel architectures |

---

## Switching to another commodity

After reaching target R2 for Montana Wheat, edit `config.py`:

```python
TARGET_STATE = "20"        # Kansas FIPS
TARGET_CROP = "WHEAT"      # same crop, different geography
GROWING_SEASON_START = 10  # winter wheat: October
GROWING_SEASON_END = 6     # harvest: June
```

Re-run `python prepare.py` to pull new data, then restart the loop.

---

## Hardware requirements

- Apple Silicon Mac (M4 Pro with 24 GB confirmed)
- macOS 14+
- Python 3.10+
- ~10 GB disk (data cache + model checkpoints)
- Internet for initial API data pulls

The dataset is tiny (~1 MB tensors for 54 counties x 24 years).
The 24 GB unified memory is far more than needed.  MPS GPU
acceleration handles the training.  Each 5-minute experiment uses
a small fraction of available compute.

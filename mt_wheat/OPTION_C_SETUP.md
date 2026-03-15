# Option C: Autoresearch Loop for Montana Wheat Yield Prediction

This guide adapts the autoresearch-mlx autonomous experiment protocol
to the mt_wheat yield-prediction task.  The agent (Claude Code)
edits `train.py`, runs fixed-budget experiments, measures `val_r2`,
keeps improvements, reverts failures, and loops indefinitely.

## Architecture overview

```
autoresearch_1/
  mt_wheat/
    config.py       — constants, API endpoints, feature dims (READ-ONLY)
    prepare.py      — data pipeline: Census + POWER + MODIS + NASS +
                       NRCS -> tensors (READ-ONLY)
    train.py        — model + optimizer + loop (AGENT EDITS THIS)
    program.md      — autonomous experiment protocol (READ-ONLY)
    results.tsv     — experiment log (AGENT APPENDS)
    pyproject.toml  — dependencies
    OPTION_A_SETUP.md
    OPTION_C_SETUP.md
```

The pattern mirrors autoresearch-mlx exactly:
- prepare.py is FIXED (data + evaluation)
- train.py is MUTABLE (model + optimizer + hyperparameters)
- program.md is the agent instructions
- results.tsv is the experiment log

The key difference: **val_r2** replaces **val_bpb** as the metric.
Higher is better (target: 0.6 – 0.8).  The current baseline is
around -0.2, meaning there is enormous room for improvement.

## Prerequisites

- Option A completed (autoresearch-mlx runs on your M4 Pro)
- NASS API key: export NASS_API_KEY=9F1B344A-CA91-36EE-997E-7F4C7F750432
- NASA Earthdata credentials (registered)
- Python 3.10+, uv installed

## Step 1 — Set up the mt_wheat directory

```bash
cd ~/Projects
git clone https://github.com/CryptoThaler/autoresearch_1.git
cd autoresearch_1/mt_wheat

# Or if you already have the repo:
cd ~/Projects/autoresearch_1/mt_wheat
```

## Step 2 — Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy requests pandas scipy
```

## Step 3 — Set environment variables

```bash
export NASS_API_KEY=9F1B344A-CA91-36EE-997E-7F4C7F750432
export EARTHDATA_USER=<your_username>
export EARTHDATA_PASS=<your_password>
```

Add these to ~/.zshrc for persistence.

## Step 4 — Run data pipeline (one-time)

```bash
python prepare.py
```

This takes ~20 minutes on first run (API calls to 5 sources for
54 Montana counties x 24 years).  All data caches to `data_cache/`.
Subsequent runs are instant.

Expected output:
- Census: 54 county centroids
- NASA POWER: 54 x 24 = 1,296 weather series
- MODIS NDVI/EVI: satellite vegetation indices (or graceful fallback)
- NASS: wheat yields + irrigation percentages
- NRCS: soil properties (AWC, clay%, organic matter, pH, Ksat)
- Final tensors: temporal [N, 12, 10] + static [N, 15] + targets [N]

## Step 5 — Establish baseline

```bash
python train.py > run.log 2>&1
grep "^val_r2:\|^val_rmse:\|^peak_memory_mb:" run.log
```

Initialize results.tsv:
```
commitval_r2val_rmsememory_mbstatusdescription
<hash><r2><rmse><mem>keepbaseline
```

## Step 6 — Launch autonomous loop

Point Claude Code at `program.md`:

```bash
git checkout -b autoresearch/wheat-v1
# Open Claude Code, point it at mt_wheat/program.md
# Let it run overnight
```

The agent will:
1. Read train.py and the current val_r2 baseline
2. Propose a change (architecture, hyperparameters, features, etc.)
3. Edit train.py, commit, run 5-min experiment
4. If val_r2 improves -> keep; if worse -> revert
5. Loop indefinitely

## What the agent can change

Everything in train.py is fair game:
- Model architecture (CNN layers, kernel sizes, MLP widths)
- Fusion strategy (concatenate, attention, gating)
- Optimizer (Adam, AdamW, SGD with momentum, learning rate schedules)
- Regularization (dropout, weight decay, batch norm, layer norm)
- Feature engineering (interaction terms, polynomial features)
- Training loop (gradient accumulation, mixed precision, warm restarts)
- Batch size, learning rate, number of epochs within budget

## What the agent CANNOT change

- prepare.py (data pipeline and evaluation functions)
- config.py (constants and API configuration)
- The 300-second (5-minute) time budget
- The evaluation metric (val_r2, val_rmse)

## Expected progression

| Phase | val_r2 range | What typically helps |
|-------|-------------|---------------------|
| Baseline | -0.2 to 0.1 | Basic model, untuned |
| Quick wins | 0.1 to 0.3 | Learning rate tuning, proper normalization |
| Solid model | 0.3 to 0.5 | Architecture improvements, dropout tuning |
| Competitive | 0.5 to 0.7 | Feature interactions, ensemble-like tricks |
| Research grade | 0.7+ | Novel architectures, advanced regularization |

## M4 Pro specifics

- Use MPS backend: `torch.device("mps")` for GPU acceleration
- 24 GB unified memory — plenty for this dataset (~1 MB tensors)
- Each experiment: ~5 min train + ~30s eval = ~6 min total
- Overnight (8 hours): ~80 experiments
- Weekend run (48 hours): ~480 experiments

## Monitoring

While Claude Code runs autonomously, you can monitor:

```bash
# Watch live results
tail -f mt_wheat/results.tsv

# Count experiments
wc -l mt_wheat/results.tsv

# See best result so far
sort -t$'\t' -k2 -rn mt_wheat/results.tsv | head -5

# Check git log
git log --oneline -20
```

## Switching commodities

After reaching target R2 for Montana Wheat, switch to a new commodity
by editing config.py:

```python
TARGET_STATE = "20"        # Kansas FIPS
TARGET_CROP = "WHEAT"      # same crop, different state
GROWING_SEASON_START = 10  # winter wheat: Oct
GROWING_SEASON_END = 6     # harvest: June
```

Re-run `prepare.py` to pull new data, then restart the loop.

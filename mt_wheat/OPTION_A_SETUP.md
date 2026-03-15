# Option A: Validate autoresearch-mlx on M4 Pro

This guide gets the upstream autoresearch-mlx NLP experiment running
on your M4 Pro (24 GB) to confirm MLX, uv, and the autonomous loop
all work before we switch to the yield-prediction task.

## Prerequisites

- macOS 14+ on Apple Silicon (M4 Pro confirmed)
- Python 3.10 – 3.13
- ~10 GB free disk (data shards + cache)
- Internet for initial downloads

## Step 1 — Install uv (one-time)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.zshrc      # or restart terminal
uv --version         # should print 0.x.x
```

## Step 2 — Clone autoresearch-mlx

```bash
cd ~/Projects   # or wherever you keep repos
git clone https://github.com/trevin-creator/autoresearch-mlx.git
cd autoresearch-mlx
```

## Step 3 — Install dependencies

```bash
uv sync          # creates .venv, installs mlx, numpy, pyarrow, etc.
```

## Step 4 — Download data + train tokenizer

```bash
uv run prepare.py --num-shards 10   # ~2 GB, takes ~5 min
```

This downloads 10 text shards from HuggingFace + 1 pinned validation
shard, then trains an 8192-token BPE tokenizer.  Everything caches to
`~/.cache/autoresearch/`.

## Step 5 — Run baseline experiment

```bash
uv run train.py
```

This trains a 4-layer GPT for exactly 5 minutes (wall clock), then
evaluates val_bpb.  On M4 Pro expect:
- val_bpb around 1.8 – 2.0  (depends on exact batch/throughput)
- peak memory ~20 – 27 GB unified
- ~6–7 minutes total (5 min train + compile/eval overhead)

## Step 6 — Record your baseline

The script prints a summary block ending with `val_bpb: X.XXXXXX`.
Note this number — it is YOUR hardware-specific baseline.

## Step 7 — (Optional) Run one autonomous loop cycle

To verify the full Claude Code integration:

```bash
git checkout -b autoresearch/test1
# Point Claude Code at program.md
# Or manually: edit train.py, commit, run, evaluate, keep/revert
```

## What success looks like

- `uv sync` completes without errors
- `prepare.py` downloads shards and trains tokenizer
- `train.py` runs for ~5 min, prints val_bpb < 3.0
- No OOM errors (24 GB should be plenty for depth=4)
- MPS/MLX uses unified memory efficiently

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `uv: command not found` | Re-source shell: `source ~/.zshrc` |
| `mlx` import error | Ensure Apple Silicon, not Rosetta: `arch` should say `arm64` |
| OOM during eval | Reduce `FINAL_EVAL_BATCH_SIZE` in train.py from 256 to 128 |
| Download failures | Re-run `uv run prepare.py` — it skips already-downloaded shards |

## Next step

Once this runs clean, proceed to **Option C** — adapting the
autoresearch loop for Montana Wheat yield prediction.
See `OPTION_C_SETUP.md`.

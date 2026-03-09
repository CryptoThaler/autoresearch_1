"""
Autoresearch-EnvSci: crop yield prediction training script.
Single-GPU, single-file. The agent modifies this file.

Usage: uv run train.py
"""

import os
import gc
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    TIME_BUDGET, BIWEEKLY_STEPS, NUM_FEATURES, TARGET_CROP,
    load_splits, make_dataloader,
    evaluate_rmse, evaluate_r2, evaluate_mae,
)

# ---------------------------------------------------------------------------
# Crop Yield Prediction Model
# ---------------------------------------------------------------------------

class YieldPredictor(nn.Module):
      """Baseline MLP model for crop yield prediction.

          Takes growing-season features (T time steps x F features) and
              predicts a single scalar yield value (bu/acre).

                  The baseline averages over time steps then passes through an MLP.
                      This is intentionally simple — the agent should improve it.
                          """

    def __init__(self, n_steps=BIWEEKLY_STEPS, n_features=NUM_FEATURES,
                                  hidden_dim=256, n_layers=3, dropout=0.1):
                                            super().__init__()
                                            self.n_steps = n_steps
                                            self.n_features = n_features

        # Input: average over time steps -> (batch, n_features)
        input_dim = n_features

        layers = []
        in_dim = input_dim
        for i in range(n_layers):
                      layers.append(nn.Linear(in_dim, hidden_dim))
                      layers.append(nn.ReLU())
                      layers.append(nn.Dropout(dropout))
                      in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
              """
                      Args:
                                  x: tensor of shape (batch, T, F) — normalized features

                                          Returns:
                                                      predictions: tensor of shape (batch, 1) — yield in bu/acre
                                                              """
              # Baseline: simple mean pooling over time dimension
              x = x.mean(dim=1)  # (batch, F)
        return self.net(x)


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
HIDDEN_DIM = 256          # MLP hidden layer width
N_LAYERS = 3              # number of hidden layers
DROPOUT = 0.1             # dropout rate

# Optimization
LEARNING_RATE = 1e-3      # Adam learning rate
WEIGHT_DECAY = 1e-4       # L2 regularization
BATCH_SIZE = 64           # training batch size
WARMUP_STEPS = 50         # LR warmup steps
GRAD_CLIP = 1.0           # gradient clipping max norm

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load data
data = load_splits(device=device)
train_features = data["train_features"]
train_labels = data["train_labels"]
val_features = data["val_features"]
val_labels = data["val_labels"]

print(f"Train samples: {train_features.shape[0]}")
print(f"Val samples:   {val_features.shape[0]}")
print(f"Features: {train_features.shape[1]} steps x {train_features.shape[2]} features")
print(f"Crop: {TARGET_CROP}")
print(f"Time budget: {TIME_BUDGET}s")

# Build model
model = YieldPredictor(
      n_steps=BIWEEKLY_STEPS,
      n_features=NUM_FEATURES,
      hidden_dim=HIDDEN_DIM,
      n_layers=N_LAYERS,
      dropout=DROPOUT,
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")

# Optimizer
optimizer = torch.optim.AdamW(
      model.parameters(),
      lr=LEARNING_RATE,
      weight_decay=WEIGHT_DECAY,
)

# Loss function
criterion = nn.MSELoss()

# ---------------------------------------------------------------------------
# LR Schedule
# ---------------------------------------------------------------------------

def get_lr(step, total_steps):
      """Linear warmup then cosine decay."""
      if step < WARMUP_STEPS:
                return LEARNING_RATE * (step + 1) / WARMUP_STEPS
            progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    return LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * progress))

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
total_training_time = 0
step = 0
best_val_rmse = float("inf")
smooth_loss = 0

# Estimate total steps for LR schedule
steps_per_epoch = math.ceil(train_features.shape[0] / BATCH_SIZE)
estimated_total_steps = int(TIME_BUDGET / 0.01) * steps_per_epoch  # rough estimate

print(f"Steps per epoch: {steps_per_epoch}")
print()

# GC management
gc.collect()
gc.disable()

while True:
      model.train()

    for batch_x, batch_y in make_dataloader(train_features, train_labels,
                                                                                          BATCH_SIZE, shuffle=True):
                                                                                                    torch.cuda.synchronize() if device.type == "cuda" else None
                                                                                                    t0 = time.time()

        # Forward
        preds = model(batch_x).squeeze(-1)
        loss = criterion(preds, batch_y)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if GRAD_CLIP > 0:
                      torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        # LR schedule
        lr = get_lr(step, estimated_total_steps)
        for param_group in optimizer.param_groups:
                      param_group["lr"] = lr

        optimizer.step()

        torch.cuda.synchronize() if device.type == "cuda" else None
        t1 = time.time()
        dt = t1 - t0

        if step > 5:
                      total_training_time += dt

        # Logging
        loss_val = loss.item()
        ema_beta = 0.95
        smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_val
        debiased = smooth_loss / (1 - ema_beta ** (step + 1))

        if step % 100 == 0:
                      pct = 100 * min(total_training_time / TIME_BUDGET, 1.0)
                      remaining = max(0, TIME_BUDGET - total_training_time)
                      print(f"step {step:05d} ({pct:.1f}%) | loss: {debiased:.4f} | "
                            f"lr: {lr:.6f} | remaining: {remaining:.0f}s")

        step += 1

        # Time check
        if step > 5 and total_training_time >= TIME_BUDGET:
                      break

    if step > 5 and total_training_time >= TIME_BUDGET:
              break

print()
total_tokens = step * BATCH_SIZE

# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

model.eval()
val_rmse = evaluate_rmse(model, val_features, val_labels)
val_r2 = evaluate_r2(model, val_features, val_labels)
val_mae = evaluate_mae(model, val_features, val_labels)

t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0

# Final summary (matches program.md output format)
print("---")
print(f"val_rmse: {val_rmse:.6f}")
print(f"val_r2: {val_r2:.6f}")
print(f"val_mae: {val_mae:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds: {t_end - t_start:.1f}")
print(f"peak_vram_mb: {peak_vram_mb:.1f}")
print(f"num_steps: {step}")
print(f"num_params_K: {num_params / 1e3:.1f}")
print(f"crop: {TARGET_CROP.lower()}")
print(f"num_counties: {train_features.shape[0] + val_features.shape[0]}")
print(f"num_years: {len(set(c[1] for c in data['counties_train'] + data['counties_val']))}")

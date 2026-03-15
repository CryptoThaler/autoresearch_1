"""
Montana Wheat Yield Prediction - Training Script
Dual-branch model: temporal CNN + static MLP with fusion.
Optimized for M4 Pro with MPS acceleration.
Usage: python train.py
"""

import os
import gc
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    TIME_BUDGET, BIWEEKLY_STEPS, NUM_TEMPORAL_FEATURES,
    NUM_STATIC_FEATURES, TARGET_CROP, TARGET_STATE_NAME,
)
from prepare import (
    load_splits, make_dataloader,
    evaluate_rmse, evaluate_r2, evaluate_mae,
)


# ---------------------------------------------------------------------------
# Model: Dual-Branch Yield Predictor
# ---------------------------------------------------------------------------

class TemporalBranch(nn.Module):
    """1D CNN over biweekly time steps to capture seasonal patterns."""

    def __init__(self, in_channels, hidden=64, out_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.conv3 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden, out_dim)

    def forward(self, x):
        # x: (batch, time_steps, features) -> transpose for Conv1d
        x = x.transpose(1, 2)  # (batch, features, time_steps)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)  # (batch, hidden)
        return self.fc(x)


class StaticBranch(nn.Module):
    """MLP for soil + spatial + irrigation features."""

    def __init__(self, in_dim, hidden=64, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class WheatYieldPredictor(nn.Module):
    """Dual-branch model fusing temporal and static features."""

    def __init__(self, temporal_features=NUM_TEMPORAL_FEATURES,
                 static_features=NUM_STATIC_FEATURES,
                 temporal_hidden=64, static_hidden=64,
                 temporal_out=64, static_out=32,
                 fusion_hidden=128, dropout=0.15):
        super().__init__()
        self.temporal_branch = TemporalBranch(
            temporal_features, temporal_hidden, temporal_out
        )
        self.static_branch = StaticBranch(
            static_features, static_hidden, static_out
        )

        fusion_in = temporal_out + static_out
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden // 2, 1),
        )

    def forward(self, temporal, static):
        t_feat = self.temporal_branch(temporal)
        s_feat = self.static_branch(static)
        combined = torch.cat([t_feat, s_feat], dim=-1)
        return self.fusion(combined)


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# Model
TEMPORAL_HIDDEN = 64
STATIC_HIDDEN = 64
TEMPORAL_OUT = 64
STATIC_OUT = 32
FUSION_HIDDEN = 128
DROPOUT = 0.15

# Optimization
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
WARMUP_STEPS = 100
GRAD_CLIP = 1.0

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)

# Device selection: MPS (M4 Pro) > CUDA > CPU
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed(42)
else:
    device = torch.device("cpu")
print(f"Device: {device}")

# Load data
data = load_splits(device=device)
train_temporal = data["train_temporal"]
train_static = data["train_static"]
train_labels = data["train_labels"]
val_temporal = data["val_temporal"]
val_static = data["val_static"]
val_labels = data["val_labels"]

print(f"Train: {train_temporal.shape[0]} samples")
print(f"  temporal: {train_temporal.shape}")
print(f"  static:   {train_static.shape}")
print(f"Val: {val_temporal.shape[0]} samples")
print(f"Time budget: {TIME_BUDGET}s")

# Build model
model = WheatYieldPredictor(
    temporal_features=train_temporal.shape[-1],
    static_features=train_static.shape[-1],
    temporal_hidden=TEMPORAL_HIDDEN,
    static_hidden=STATIC_HIDDEN,
    temporal_out=TEMPORAL_OUT,
    static_out=STATIC_OUT,
    fusion_hidden=FUSION_HIDDEN,
    dropout=DROPOUT,
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
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
# Training Loop
# ---------------------------------------------------------------------------

t_train_start = time.time()
total_training_time = 0
step = 0
smooth_loss = 0

steps_per_epoch = math.ceil(train_temporal.shape[0] / BATCH_SIZE)
estimated_total_steps = int(TIME_BUDGET / 0.005) * steps_per_epoch

print(f"Steps per epoch: {steps_per_epoch}")
print()

gc.collect()
gc.disable()

while True:
    model.train()
    for bt, bs, by in make_dataloader(train_temporal, train_static, train_labels,
                                       BATCH_SIZE, shuffle=True):
        t0 = time.time()

        preds = model(bt, bs).squeeze(-1)
        loss = criterion(preds, by)

        optimizer.zero_grad()
        loss.backward()

        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        lr = get_lr(step, estimated_total_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.step()
        t1 = time.time()
        dt = t1 - t0

        if step > 5:
            total_training_time += dt

        loss_val = loss.item()
        ema_beta = 0.95
        smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_val
        debiased = smooth_loss / (1 - ema_beta ** (step + 1))

        if step % 200 == 0:
            pct = 100 * min(total_training_time / TIME_BUDGET, 1.0)
            rem = max(0, TIME_BUDGET - total_training_time)
            print(f"step {step:05d} ({pct:.1f}%) | loss: {debiased:.4f} | "
                  f"lr: {lr:.6f} | remaining: {rem:.0f}s")

        step += 1
        if step > 5 and total_training_time >= TIME_BUDGET:
            break
    if step > 5 and total_training_time >= TIME_BUDGET:
        break

print()

# ---------------------------------------------------------------------------
# Final Evaluation
# ---------------------------------------------------------------------------

model.eval()
val_rmse = evaluate_rmse(model, val_temporal, val_static, val_labels)
val_r2 = evaluate_r2(model, val_temporal, val_static, val_labels)
val_mae = evaluate_mae(model, val_temporal, val_static, val_labels)

t_end = time.time()
if device.type == "cuda":
    peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024
else:
    peak_vram = 0.0

print("---")
print(f"val_rmse: {val_rmse:.6f}")
print(f"val_r2: {val_r2:.6f}")
print(f"val_mae: {val_mae:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds: {t_end - t_start:.1f}")
print(f"peak_vram_mb: {peak_vram:.1f}")
print(f"num_steps: {step}")
print(f"num_params_K: {num_params / 1e3:.1f}")
print(f"crop: {TARGET_CROP.lower()}")
print(f"state: {TARGET_STATE_NAME.lower()}")
print(f"num_counties: {train_temporal.shape[0] + val_temporal.shape[0]}")
print(f"num_years: {len(set(c[1] for c in data['meta_train'] + data['meta_val']))}")

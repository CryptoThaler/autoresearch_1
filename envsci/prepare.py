"""
Autoresearch-EnvSci: data preparation and evaluation harness.

Downloads NASA POWER climate data and USDA NASS crop yield data,
builds county-level growing season feature tensors, and provides
the fixed evaluation functions.

Usage:
    uv run prepare.py                # full prep (download + process)
        uv run prepare.py --years 5      # only last 5 years (for testing)

        Data stored in ~/.cache/autoresearch-envsci/
        """

import os
import sys
import time
import json
import math
import argparse
from pathlib import Path

import requests
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300          # training time budget in seconds (5 minutes)
GROWING_SEASON_START = 4   # April (month number)
GROWING_SEASON_END = 10    # October (month number)
BIWEEKLY_STEPS = 14        # number of biweekly time steps Apr-Oct
NUM_FEATURES = 8           # features per time step from NASA POWER
TRAIN_YEAR_END = 2020      # train on years <= 2020
VAL_YEAR_START = 2021      # validate on years >= 2021
VAL_YEAR_END = 2023        # validate on years <= 2023
TARGET_CROP = "CORN"       # primary crop for yield prediction

# NASA POWER API parameters
POWER_PARAMS = [
      "T2M",              # temperature at 2m (C), daily mean
      "T2M_MAX",          # daily max temperature (C)
      "T2M_MIN",          # daily min temperature (C)
      "PRECTOTCORR",      # precipitation corrected (mm/day)
      "ALLSKY_SFC_SW_DWN", # solar radiation (MJ/m2/day)
      "RH2M",             # relative humidity at 2m (%)
      "WS2M",             # wind speed at 2m (m/s)
      "T2MDEW",           # dewpoint temperature (C)
]

# USDA NASS API
NASS_API_BASE = "https://quickstats.nass.usda.gov/api/api_GET/"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-envsci")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TENSOR_DIR = os.path.join(CACHE_DIR, "tensors")

# Top corn-producing US states (FIPS codes for state-level filtering)
# These cover ~85% of US corn production
CORN_BELT_STATES = [
      "17",  # Illinois
      "18",  # Indiana
      "19",  # Iowa
      "20",  # Kansas
      "26",  # Michigan
      "27",  # Minnesota
      "29",  # Missouri
      "31",  # Nebraska
      "38",  # North Dakota
      "39",  # Ohio
      "46",  # South Dakota
      "55",  # Wisconsin
]

# County centroids for NASA POWER queries (lat, lon)
# In practice these would be loaded from a shapefile or census data
# This is a simplified version using state-level representative points
STATE_CENTROIDS = {
      "17": (40.0, -89.0),   # Illinois
      "18": (39.8, -86.2),   # Indiana
      "19": (42.0, -93.5),   # Iowa
      "20": (38.5, -98.3),   # Kansas
      "26": (43.3, -84.5),   # Michigan
      "27": (45.0, -94.3),   # Minnesota
      "29": (38.5, -92.5),   # Missouri
      "31": (41.5, -99.8),   # Nebraska
      "38": (47.5, -100.5),  # North Dakota
      "39": (40.4, -82.8),   # Ohio
      "46": (44.5, -100.2),  # South Dakota
      "55": (44.5, -89.8),   # Wisconsin
}

# ---------------------------------------------------------------------------
# NASA POWER data download
# ---------------------------------------------------------------------------

def download_power_data(lat, lon, start_year, end_year):
      """Download daily climate data from NASA POWER API for a location."""
      url = "https://power.larc.nasa.gov/api/temporal/daily/point"
      params = {
          "parameters": ",".join(POWER_PARAMS),
          "community": "AG",
          "longitude": lon,
          "latitude": lat,
          "start": f"{start_year}0101",
          "end": f"{end_year}1231",
          "format": "JSON",
      }

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
              try:
                            resp = requests.get(url, params=params, timeout=60)
                            resp.raise_for_status()
                            data = resp.json()
                            return data.get("properties", {}).get("parameter", {})
except (requests.RequestException, ValueError) as e:
            print(f"  Attempt {attempt}/{max_attempts} failed for ({lat}, {lon}): {e}")
            if attempt < max_attempts:
                              time.sleep(2 ** attempt)
                  return None


def extract_growing_season(power_data, year):
      """Extract biweekly averaged features for the growing season of a given year."""
      if power_data is None:
                return None

      features = []
      for step in range(BIWEEKLY_STEPS):
                # Each step is ~14 days starting from April 1
                start_doy = (GROWING_SEASON_START - 1) * 30 + step * 14 + 1
                end_doy = start_doy + 13

          step_features = []
        for param in POWER_PARAMS:
                      param_data = power_data.get(param, {})
                      values = []
                      for doy in range(start_doy, end_doy + 1):
                                        # Convert DOY to date string YYYYMMDD
                                        from datetime import date, timedelta
                                        try:
                                                              d = date(year, 1, 1) + timedelta(days=doy - 1)
                                                              key = d.strftime("%Y%m%d")
                                                              val = param_data.get(key, -999)
                                                              if val != -999 and val is not None:
                                                                                        values.append(float(val))
                                        except (ValueError, OverflowError):
                                                              continue

                                    # Biweekly mean (or 0 if no valid data)
                                    step_features.append(np.mean(values) if values else 0.0)

        features.append(step_features)

    return np.array(features, dtype=np.float32)  # shape: (BIWEEKLY_STEPS, NUM_FEATURES)


# ---------------------------------------------------------------------------
# USDA NASS yield data download
# ---------------------------------------------------------------------------

def download_nass_yields(api_key, crop, start_year, end_year):
      """Download county-level crop yield data from USDA NASS Quick Stats API.

          Requires a free API key from https://quickstats.nass.usda.gov/api/
              Set NASS_API_KEY environment variable.
                  """
    all_data = []

    for state_fips in CORN_BELT_STATES:
              params = {
                            "key": api_key,
                            "commodity_desc": crop,
                            "statisticcat_desc": "YIELD",
                            "unit_desc": "BU / ACRE",
                            "freq_desc": "ANNUAL",
                            "agg_level_desc": "COUNTY",
                            "state_fips_code": state_fips,
                            "year__GE": str(start_year),
                            "year__LE": str(end_year),
                            "format": "JSON",
              }

        try:
                      resp = requests.get(NASS_API_BASE, params=params, timeout=30)
            resp.raise_for_status()
            result = resp.json()
            records = result.get("data", [])
            all_data.extend(records)
            print(f"  State {state_fips}: {len(records)} yield records")
except (requests.RequestException, ValueError) as e:
            print(f"  State {state_fips}: failed - {e}")

        time.sleep(0.5)  # rate limiting

    return all_data


def parse_yield_records(records):
      """Parse NASS yield records into a clean dictionary.

          Returns: dict mapping (county_fips, year) -> yield_bu_acre
              """
    yields = {}
    for rec in records:
              try:
                            county_code = rec.get("county_code", "")
                            state_code = rec.get("state_fips_code", "")
                            year = int(rec.get("year", 0))
                            value_str = rec.get("Value", "").replace(",", "")

            if not county_code or not state_code or not year:
                              continue
                          if value_str in ("", "(D)", "(Z)", "(NA)"):
                                            continue

            fips = f"{state_code}{county_code}"
            yield_val = float(value_str)
            yields[(fips, year)] = yield_val
except (ValueError, TypeError):
            continue

    return yields


# ---------------------------------------------------------------------------
# Feature tensor construction
# ---------------------------------------------------------------------------

def build_tensors(power_cache, yield_data, years):
      """Build aligned feature and label tensors.

          Args:
                  power_cache: dict mapping state_fips -> power_data dict
                          yield_data: dict mapping (county_fips, year) -> yield
                                  years: list of years to include

                                      Returns:
                                              features: tensor of shape (N, BIWEEKLY_STEPS, NUM_FEATURES)
                                                      labels: tensor of shape (N,) — yield in bu/acre
                                                              metadata: list of (county_fips, year) tuples
                                                                  """
    feature_list = []
    label_list = []
    meta_list = []

    for (fips, year), yield_val in sorted(yield_data.items()):
              if year not in years:
                            continue

        state_fips = fips[:2]
        if state_fips not in STATE_CENTROIDS:
                      continue

        # Get growing season features from power data
        power_data = power_cache.get(state_fips)
        gs_features = extract_growing_season(power_data, year)

        if gs_features is None:
                      continue

        feature_list.append(gs_features)
        label_list.append(yield_val)
        meta_list.append((fips, year))

    if not feature_list:
              return None, None, None

    features = torch.tensor(np.stack(feature_list), dtype=torch.float32)
    labels = torch.tensor(label_list, dtype=torch.float32)

    return features, labels, meta_list


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def compute_normalization(features):
      """Compute per-feature mean and std from training data.

          Args:
                  features: tensor of shape (N, T, F)

                      Returns:
                              mean: tensor of shape (1, 1, F)
                                      std: tensor of shape (1, 1, F)
                                          """
    # Flatten over N and T dimensions
    flat = features.reshape(-1, features.shape[-1])
    mean = flat.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, F)
    std = flat.std(dim=0, keepdim=True).unsqueeze(0)    # (1, 1, F)
    std = std.clamp(min=1e-6)
    return mean, std


# ---------------------------------------------------------------------------
# Data loading (imported by train.py)
# ---------------------------------------------------------------------------

def load_splits(device="cpu"):
      """Load pre-built train/val splits and normalization stats.

          Returns dict with keys:
                  train_features, train_labels, val_features, val_labels,
                          feature_mean, feature_std, feature_names, counties_train,
                                  counties_val
                                      """
    data = {}
    for key in ["train_features", "train_labels", "val_features", "val_labels",
                                 "feature_mean", "feature_std"]:
                                           path = os.path.join(TENSOR_DIR, f"{key}.pt")
                                           data[key] = torch.load(path, map_location=device, weights_only=True)

    with open(os.path.join(TENSOR_DIR, "feature_names.json")) as f:
              data["feature_names"] = json.load(f)

    with open(os.path.join(TENSOR_DIR, "counties_train.json")) as f:
              data["counties_train"] = json.load(f)

    with open(os.path.join(TENSOR_DIR, "counties_val.json")) as f:
              data["counties_val"] = json.load(f)

    return data


def make_dataloader(features, labels, batch_size, shuffle=True):
      """Simple dataloader yielding (X, y) batches.

          Args:
                  features: tensor (N, T, F)
                          labels: tensor (N,)
                                  batch_size: int
                                          shuffle: bool

                                              Yields:
                                                      (batch_features, batch_labels) tensors on same device as input
                                                          """
    N = features.shape[0]
    indices = torch.randperm(N) if shuffle else torch.arange(N)

    for start in range(0, N, batch_size):
              end = min(start + batch_size, N)
        idx = indices[start:end]
        yield features[idx], labels[idx]


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_rmse(model, features, labels, batch_size=256):
      """Root Mean Squared Error on yield predictions (bu/acre).

          This is the primary metric. Lower is better.
              """
    model.eval()
    all_preds = []
    N = features.shape[0]

    for start in range(0, N, batch_size):
              end = min(start + batch_size, N)
        batch_x = features[start:end]
        preds = model(batch_x).squeeze(-1)
        all_preds.append(preds)

    all_preds = torch.cat(all_preds)
    mse = ((all_preds - labels) ** 2).mean()
    return math.sqrt(mse.item())


@torch.no_grad()
def evaluate_r2(model, features, labels, batch_size=256):
      """R-squared (coefficient of determination).

          Higher is better. 1.0 = perfect, 0.0 = predicts mean.
              """
    model.eval()
    all_preds = []
    N = features.shape[0]

    for start in range(0, N, batch_size):
              end = min(start + batch_size, N)
        batch_x = features[start:end]
        preds = model(batch_x).squeeze(-1)
        all_preds.append(preds)

    all_preds = torch.cat(all_preds)
    ss_res = ((labels - all_preds) ** 2).sum()
    ss_tot = ((labels - labels.mean()) ** 2).sum()
    return 1.0 - (ss_res / ss_tot).item()


@torch.no_grad()
def evaluate_mae(model, features, labels, batch_size=256):
      """Mean Absolute Error (bu/acre). Supplementary metric."""
    model.eval()
    all_preds = []
    N = features.shape[0]

    for start in range(0, N, batch_size):
              end = min(start + batch_size, N)
        batch_x = features[start:end]
        preds = model(batch_x).squeeze(-1)
        all_preds.append(preds)

    all_preds = torch.cat(all_preds)
    return (all_preds - labels).abs().mean().item()


# ---------------------------------------------------------------------------
# Main: download and prepare everything
# ---------------------------------------------------------------------------

if __name__ == "__main__":
      parser = argparse.ArgumentParser(description="Prepare data for autoresearch-envsci")
    parser.add_argument("--years", type=int, default=15,
                                                help="Number of years of data to download (default: 15, back from 2023)")
    parser.add_argument("--nass-api-key", type=str, default=None,
                                                help="USDA NASS API key (or set NASS_API_KEY env var)")
    args = parser.parse_args()

    api_key = args.nass_api_key or os.environ.get("NASS_API_KEY")
    if not api_key:
              print("ERROR: USDA NASS API key required.")
        print("  Get a free key at: https://quickstats.nass.usda.gov/api/")
        print("  Then: export NASS_API_KEY=your_key_here")
        print("  Or pass: --nass-api-key your_key_here")
        sys.exit(1)

    end_year = VAL_YEAR_END
    start_year = end_year - args.years + 1

    print(f"Cache directory: {CACHE_DIR}")
    print(f"Year range: {start_year}-{end_year}")
    print()

    # Step 1: Download NASA POWER data for each state centroid
    os.makedirs(DATA_DIR, exist_ok=True)
    power_cache_path = os.path.join(DATA_DIR, "power_cache.json")

    if os.path.exists(power_cache_path):
              print("NASA POWER: loading cached data...")
        with open(power_cache_path) as f:
                      power_cache = json.load(f)
else:
        print("NASA POWER: downloading climate data...")
        power_cache = {}
        for state_fips, (lat, lon) in STATE_CENTROIDS.items():
                      print(f"  Downloading state {state_fips} ({lat}, {lon})...")
            data = download_power_data(lat, lon, start_year, end_year)
            if data:
                              power_cache[state_fips] = data
                              print(f"  State {state_fips}: OK ({len(data)} parameters)")
else:
                print(f"  State {state_fips}: FAILED")
            time.sleep(1)  # rate limiting

        with open(power_cache_path, "w") as f:
                      json.dump(power_cache, f)
        print(f"NASA POWER: cached to {power_cache_path}")
    print()

    # Step 2: Download USDA NASS yield data
    nass_cache_path = os.path.join(DATA_DIR, "nass_yields.json")

    if os.path.exists(nass_cache_path):
              print("USDA NASS: loading cached yield data...")
        with open(nass_cache_path) as f:
                      raw_records = json.load(f)
else:
        print("USDA NASS: downloading county yield data...")
        raw_records = download_nass_yields(api_key, TARGET_CROP, start_year, end_year)
        with open(nass_cache_path, "w") as f:
                      json.dump(raw_records, f)
        print(f"USDA NASS: {len(raw_records)} records cached")
    print()

    # Step 3: Parse yields
    yield_data = parse_yield_records(raw_records)
    print(f"Parsed yields: {len(yield_data)} (county, year) pairs")

    # Step 4: Build feature tensors
    print("Building feature tensors...")
    os.makedirs(TENSOR_DIR, exist_ok=True)

    train_years = list(range(start_year, TRAIN_YEAR_END + 1))
    val_years = list(range(VAL_YEAR_START, VAL_YEAR_END + 1))

    train_features, train_labels, train_meta = build_tensors(
              power_cache, yield_data, train_years
    )
    val_features, val_labels, val_meta = build_tensors(
              power_cache, yield_data, val_years
    )

    if train_features is None or val_features is None:
              print("ERROR: Could not build tensors. Check data downloads.")
        sys.exit(1)

    print(f"Train: {train_features.shape[0]} samples, years {train_years[0]}-{train_years[-1]}")
    print(f"Val:   {val_features.shape[0]} samples, years {val_years[0]}-{val_years[-1]}")

    # Step 5: Compute normalization from training data only
    feat_mean, feat_std = compute_normalization(train_features)

    # Normalize both splits using training stats
    train_features_norm = (train_features - feat_mean) / feat_std
    val_features_norm = (val_features - feat_mean) / feat_std

    # Step 6: Save everything
    torch.save(train_features_norm, os.path.join(TENSOR_DIR, "train_features.pt"))
    torch.save(train_labels, os.path.join(TENSOR_DIR, "train_labels.pt"))
    torch.save(val_features_norm, os.path.join(TENSOR_DIR, "val_features.pt"))
    torch.save(val_labels, os.path.join(TENSOR_DIR, "val_labels.pt"))
    torch.save(feat_mean, os.path.join(TENSOR_DIR, "feature_mean.pt"))
    torch.save(feat_std, os.path.join(TENSOR_DIR, "feature_std.pt"))

    with open(os.path.join(TENSOR_DIR, "feature_names.json"), "w") as f:
              json.dump(POWER_PARAMS, f)

    with open(os.path.join(TENSOR_DIR, "counties_train.json"), "w") as f:
              json.dump(train_meta, f)

    with open(os.path.join(TENSOR_DIR, "counties_val.json"), "w") as f:
              json.dump(val_meta, f)

    print()
    print(f"Tensors saved to {TENSOR_DIR}")
    print(f"  train_features: {train_features_norm.shape}")
    print(f"  train_labels:   {train_labels.shape}")
    print(f"  val_features:   {val_features_norm.shape}")
    print(f"  val_labels:     {val_labels.shape}")
    print()
    print("Done! Ready to train.")

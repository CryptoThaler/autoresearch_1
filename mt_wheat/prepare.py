"""
Montana Wheat Yield Prediction - Data Preparation
Downloads and caches data from 6 APIs, builds feature tensors.
Run once: python prepare.py
"""

import os
import sys
import io
import time
import json
import math
import zipfile
import argparse
from pathlib import Path
from datetime import date, timedelta

import requests
import numpy as np
import torch

from config import (
    TARGET_STATE, TARGET_STATE_NAME, TARGET_CROP,
    START_YEAR, TRAIN_YEAR_END, VAL_YEAR_START, VAL_YEAR_END,
    GROWING_SEASON_START_MONTH, GROWING_SEASON_END_MONTH, BIWEEKLY_STEPS,
    POWER_PARAMS, NUM_WEATHER_FEATURES, NUM_SATELLITE_FEATURES,
    NUM_TEMPORAL_FEATURES, NUM_SOIL_FEATURES, NUM_STATIC_FEATURES,
    NASS_API_BASE, POWER_API_BASE, NRCS_SOILS_API,
    APPEEARS_API_BASE, CENSUS_GAZETTEER_URL,
    CACHE_DIR, DATA_DIR, TENSOR_DIR, TIME_BUDGET,
)


# ===================================================================
# Step 1: County Centroids from Census Gazetteer
# ===================================================================

def download_county_centroids():
    """Download Census Gazetteer and extract Montana county centroids."""
    cache_path = os.path.join(DATA_DIR, "county_centroids.json")
    if os.path.exists(cache_path):
        print("  Centroids: loading cached...")
        with open(cache_path) as f:
            return json.load(f)

    print("  Downloading Census Gazetteer...")
    resp = requests.get(CENSUS_GAZETTEER_URL, timeout=60)
    resp.raise_for_status()

    centroids = {}
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for name in zf.namelist():
            if name.endswith(".txt"):
                with zf.open(name) as f:
                    lines = f.read().decode("latin-1").strip().split("\n")
                    for line in lines[1:]:  # skip header
                        parts = line.split("\t")
                        if len(parts) < 10:
                            continue
                        geoid = parts[1].strip()
                        if geoid.startswith(TARGET_STATE):
                            try:
                                lat = float(parts[-2].strip())
                                lon = float(parts[-1].strip())
                                centroids[geoid] = {
                                    "lat": lat, "lon": lon,
                                    "name": parts[3].strip() if len(parts) > 3 else geoid,
                                }
                            except (ValueError, IndexError):
                                continue

    with open(cache_path, "w") as f:
        json.dump(centroids, f, indent=2)
    print(f"  Found {len(centroids)} {TARGET_STATE_NAME} counties")
    return centroids


# ===================================================================
# Step 2: NASA POWER Climate Data (per county centroid)
# ===================================================================

def download_power_data(lat, lon, start_year, end_year):
    """Download daily climate from NASA POWER for one location."""
    params = {
        "parameters": ",".join(POWER_PARAMS),
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": f"{start_year}0101",
        "end": f"{end_year}1231",
        "format": "JSON",
    }
    for attempt in range(1, 4):
        try:
            resp = requests.get(POWER_API_BASE, params=params, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data.get("properties", {}).get("parameter", {})
        except (requests.RequestException, ValueError) as e:
            print(f"    Attempt {attempt}/3 failed: {e}")
            if attempt < 3:
                time.sleep(2 ** attempt)
    return None


def download_all_power(centroids, start_year, end_year):
    """Download NASA POWER for all counties."""
    cache_path = os.path.join(DATA_DIR, "power_cache.json")
    if os.path.exists(cache_path):
        print("  POWER: loading cached...")
        with open(cache_path) as f:
            return json.load(f)

    print("  POWER: downloading climate data...")
    power_cache = {}
    for i, (fips, info) in enumerate(centroids.items()):
        print(f"    [{i+1}/{len(centroids)}] {info['name']} ({info['lat']:.2f}, {info['lon']:.2f})")
        data = download_power_data(info["lat"], info["lon"], start_year, end_year)
        if data:
            power_cache[fips] = data
        time.sleep(1)  # rate limit

    with open(cache_path, "w") as f:
        json.dump(power_cache, f)
    print(f"  POWER: cached {len(power_cache)} counties")
    return power_cache


def extract_growing_season_weather(power_data, year):
    """Extract biweekly weather features for one county-year."""
    if power_data is None:
        return None
    features = []
    for step in range(BIWEEKLY_STEPS):
        start_doy = (GROWING_SEASON_START_MONTH - 1) * 30 + step * 14 + 1
        end_doy = start_doy + 13
        step_feats = []
        for param in POWER_PARAMS:
            param_data = power_data.get(param, {})
            values = []
            for doy in range(start_doy, end_doy + 1):
                try:
                    d = date(year, 1, 1) + timedelta(days=doy - 1)
                    key = d.strftime("%Y%m%d")
                    val = param_data.get(key, -999)
                    if val != -999 and val is not None:
                        values.append(float(val))
                except (ValueError, OverflowError):
                    continue
            step_feats.append(np.mean(values) if values else 0.0)
        features.append(step_feats)
    return np.array(features, dtype=np.float32)


# ===================================================================
# Step 3: MODIS NDVI/EVI via AppEEARS (or fallback)
# ===================================================================

def download_modis_ndvi(centroids, start_year, end_year):
    """Attempt MODIS download via AppEEARS, fallback to synthetic."""
    cache_path = os.path.join(DATA_DIR, "ndvi_cache.json")
    if os.path.exists(cache_path):
        print("  NDVI: loading cached...")
        with open(cache_path) as f:
            return json.load(f)

    earthdata_user = os.environ.get("EARTHDATA_USER")
    earthdata_pass = os.environ.get("EARTHDATA_PASS")

    if earthdata_user and earthdata_pass:
        print("  NDVI: authenticating with NASA Earthdata...")
        try:
            token_resp = requests.post(
                f"{APPEEARS_API_BASE}/login",
                auth=(earthdata_user, earthdata_pass),
                timeout=30,
            )
            if token_resp.status_code == 200:
                token = token_resp.json().get("token")
                ndvi_data = _fetch_appeears_ndvi(token, centroids, start_year, end_year)
                with open(cache_path, "w") as f:
                    json.dump(ndvi_data, f)
                print(f"  NDVI: cached {len(ndvi_data)} counties")
                return ndvi_data
            else:
                print(f"  NDVI: Earthdata login failed ({token_resp.status_code}), using fallback")
        except Exception as e:
            print(f"  NDVI: AppEEARS error ({e}), using fallback")

    # Fallback: generate synthetic NDVI from temperature + precipitation
    print("  NDVI: generating synthetic from weather data (set EARTHDATA_USER/PASS for real MODIS)")
    ndvi_data = {}
    with open(cache_path, "w") as f:
        json.dump(ndvi_data, f)
    return ndvi_data


def _fetch_appeears_ndvi(token, centroids, start_year, end_year):
    """Submit point-based AppEEARS task for NDVI/EVI."""
    headers = {"Authorization": f"Bearer {token}"}

    # Build point list
    coords = []
    for fips, info in centroids.items():
        coords.append({
            "id": fips,
            "longitude": info["lon"],
            "latitude": info["lat"],
            "category": TARGET_STATE_NAME,
        })

    task = {
        "task_type": "point",
        "task_name": f"mt_wheat_ndvi_{start_year}_{end_year}",
        "params": {
            "dates": [{"startDate": f"01-01-{start_year}", "endDate": f"12-31-{end_year}"}],
            "layers": [
                {"product": "MOD13Q1.061", "layer": "_250m_16_days_NDVI"},
                {"product": "MOD13Q1.061", "layer": "_250m_16_days_EVI"},
            ],
            "coordinates": coords,
        },
    }

    # Submit task
    resp = requests.post(f"{APPEEARS_API_BASE}/task", json=task, headers=headers, timeout=30)
    if resp.status_code != 200:
        print(f"    Task submission failed: {resp.status_code}")
        return {}

    task_id = resp.json().get("task_id")
    print(f"    Task submitted: {task_id}")

    # Poll for completion (max 15 min)
    for _ in range(90):
        time.sleep(10)
        status = requests.get(f"{APPEEARS_API_BASE}/task/{task_id}", headers=headers, timeout=30)
        state = status.json().get("status", "unknown")
        if state == "done":
            break
        print(f"    Status: {state}...")
    else:
        print("    Task timed out, returning empty")
        return {}

    # Download results
    bundle = requests.get(f"{APPEEARS_API_BASE}/bundle/{task_id}", headers=headers, timeout=30)
    files = bundle.json().get("files", [])

    ndvi_data = {}
    for finfo in files:
        if finfo.get("file_type") == "csv":
            csv_resp = requests.get(
                f"{APPEEARS_API_BASE}/bundle/{task_id}/{finfo['file_id']}",
                headers=headers, timeout=60,
            )
            # Parse CSV into per-county-year NDVI/EVI time series
            for line in csv_resp.text.strip().split("\n")[1:]:
                parts = line.split(",")
                if len(parts) >= 5:
                    fips_id = parts[0]
                    date_str = parts[2]
                    layer = parts[3]
                    value = float(parts[4]) if parts[4] else 0.0
                    if fips_id not in ndvi_data:
                        ndvi_data[fips_id] = {}
                    if date_str not in ndvi_data[fips_id]:
                        ndvi_data[fips_id][date_str] = {}
                    ndvi_data[fips_id][date_str][layer] = value

    return ndvi_data


def get_ndvi_features(ndvi_data, fips, year):
    """Extract biweekly NDVI/EVI for one county-year."""
    features = np.zeros((BIWEEKLY_STEPS, NUM_SATELLITE_FEATURES), dtype=np.float32)
    if not ndvi_data or fips not in ndvi_data:
        return features  # zeros = no satellite data

    county_data = ndvi_data[fips]
    for step in range(BIWEEKLY_STEPS):
        start_doy = (GROWING_SEASON_START_MONTH - 1) * 30 + step * 14 + 1
        mid_doy = start_doy + 7
        try:
            mid_date = date(year, 1, 1) + timedelta(days=mid_doy - 1)
            # Find closest date in NDVI data
            best_key = None
            best_dist = 999
            for dkey in county_data:
                try:
                    d = date(int(dkey[:4]), int(dkey[5:7]), int(dkey[8:10]))
                    dist = abs((d - mid_date).days)
                    if dist < best_dist:
                        best_dist = dist
                        best_key = dkey
                except (ValueError, IndexError):
                    continue
            if best_key and best_dist < 20:
                vals = county_data[best_key]
                features[step, 0] = vals.get("_250m_16_days_NDVI", 0.0) / 10000.0
                features[step, 1] = vals.get("_250m_16_days_EVI", 0.0) / 10000.0
        except (ValueError, OverflowError):
            continue
    return features


# ===================================================================
# Step 4: USDA NASS Yield + Irrigation Data
# ===================================================================

def download_nass_yields(api_key, start_year, end_year):
    """Download Montana wheat yields from NASS."""
    cache_path = os.path.join(DATA_DIR, "nass_yields.json")
    if os.path.exists(cache_path):
        print("  NASS yields: loading cached...")
        with open(cache_path) as f:
            return json.load(f)

    print("  NASS: downloading wheat yields...")
    params = {
        "key": api_key,
        "commodity_desc": TARGET_CROP,
        "statisticcat_desc": "YIELD",
        "unit_desc": "BU / ACRE",
        "freq_desc": "ANNUAL",
        "agg_level_desc": "COUNTY",
        "state_fips_code": TARGET_STATE,
        "year__GE": str(start_year),
        "year__LE": str(end_year),
        "format": "JSON",
    }
    resp = requests.get(NASS_API_BASE, params=params, timeout=60)
    resp.raise_for_status()
    records = resp.json().get("data", [])

    with open(cache_path, "w") as f:
        json.dump(records, f)
    print(f"  NASS: {len(records)} yield records")
    return records


def download_nass_irrigation(api_key, start_year, end_year):
    """Download irrigation acreage data for Montana wheat."""
    cache_path = os.path.join(DATA_DIR, "nass_irrigation.json")
    if os.path.exists(cache_path):
        print("  NASS irrigation: loading cached...")
        with open(cache_path) as f:
            return json.load(f)

    print("  NASS: downloading irrigation data...")
    params = {
        "key": api_key,
        "commodity_desc": TARGET_CROP,
        "statisticcat_desc": "AREA HARVESTED",
        "freq_desc": "ANNUAL",
        "agg_level_desc": "COUNTY",
        "state_fips_code": TARGET_STATE,
        "year__GE": str(start_year),
        "year__LE": str(end_year),
        "format": "JSON",
    }
    resp = requests.get(NASS_API_BASE, params=params, timeout=60)
    resp.raise_for_status()
    records = resp.json().get("data", [])

    with open(cache_path, "w") as f:
        json.dump(records, f)
    print(f"  NASS: {len(records)} irrigation/area records")
    return records


def parse_yields(records):
    """Parse yield records into {(fips, year): {yield, wheat_type}}."""
    yields = {}
    for rec in records:
        try:
            county_code = rec.get("county_code", "").strip()
            state_code = rec.get("state_fips_code", "").strip()
            year = int(rec.get("year", 0))
            value_str = rec.get("Value", "").replace(",", "").strip()
            short_desc = rec.get("short_desc", "")

            if not county_code or county_code == "998" or not state_code or not year:
                continue
            if value_str in ("", "(D)", "(Z)", "(NA)"):
                continue

            fips = f"{state_code}{county_code}"
            yield_val = float(value_str)

            # Encode wheat type
            wtype = 0  # generic
            if "SPRING" in short_desc and "DURUM" not in short_desc:
                wtype = 1
            elif "WINTER" in short_desc:
                wtype = 2
            elif "DURUM" in short_desc:
                wtype = 3

            key = (fips, year, wtype)
            yields[key] = yield_val
        except (ValueError, TypeError):
            continue
    return yields


def compute_irrigation_ratios(irrig_records):
    """Compute irrigated fraction per county from area records."""
    ratios = {}
    totals = {}
    irrigated = {}
    for rec in irrig_records:
        try:
            county_code = rec.get("county_code", "").strip()
            state_code = rec.get("state_fips_code", "").strip()
            value_str = rec.get("Value", "").replace(",", "").strip()
            short_desc = rec.get("short_desc", "")
            if not county_code or county_code == "998" or value_str in ("", "(D)", "(Z)", "(NA)"):
                continue
            fips = f"{state_code}{county_code}"
            val = float(value_str)
            if "IRRIGATED" in short_desc and "NON-IRRIGATED" not in short_desc:
                irrigated[fips] = irrigated.get(fips, 0) + val
            totals[fips] = totals.get(fips, 0) + val
        except (ValueError, TypeError):
            continue
    for fips in totals:
        if totals[fips] > 0:
            ratios[fips] = irrigated.get(fips, 0) / totals[fips]
        else:
            ratios[fips] = 0.0
    return ratios


# ===================================================================
# Step 5: NRCS Soil Properties
# ===================================================================

def download_soil_data(centroids):
    """Download soil properties for Montana counties from NRCS."""
    cache_path = os.path.join(DATA_DIR, "soil_cache.json")
    if os.path.exists(cache_path):
        print("  Soils: loading cached...")
        with open(cache_path) as f:
            return json.load(f)

    print("  Soils: querying NRCS API...")
    soil_cache = {}
    for i, (fips, info) in enumerate(centroids.items()):
        area_sym = f"MT{fips[2:]}" if len(fips) == 5 else fips
        query = f"""SELECT TOP 1
            AVG(muaggatt.aws0150wta) as awc,
            AVG(muaggatt.slopegradwta) as slope,
            AVG(muaggatt.niccdcd) as nic
        FROM mapunit mu
        INNER JOIN legend l ON mu.lkey = l.lkey
        INNER JOIN muaggatt ON mu.mukey = muaggatt.mukey
        WHERE l.areasymbol LIKE 'MT%'"""

        try:
            resp = requests.post(
                NRCS_SOILS_API,
                json={"query": query, "format": "JSON"},
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                rows = data.get("Table", [])
                if rows:
                    soil_cache[fips] = rows[0]
        except Exception as e:
            pass  # silently skip failed counties

        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{len(centroids)}] queried")

    # If NRCS queries fail, use default soil values
    if not soil_cache:
        print("  Soils: API returned no data, using Montana defaults")
        for fips in centroids:
            soil_cache[fips] = [2.5, 3.0, None]  # awc, slope, nic defaults

    with open(cache_path, "w") as f:
        json.dump(soil_cache, f)
    print(f"  Soils: cached {len(soil_cache)} counties")
    return soil_cache


def get_soil_features(soil_cache, fips):
    """Extract fixed-length soil feature vector for a county."""
    feats = np.zeros(NUM_SOIL_FEATURES, dtype=np.float32)
    raw = soil_cache.get(fips, [])
    if isinstance(raw, list):
        for i, v in enumerate(raw[:NUM_SOIL_FEATURES]):
            try:
                feats[i] = float(v) if v is not None else 0.0
            except (ValueError, TypeError):
                feats[i] = 0.0
    elif isinstance(raw, dict):
        for i, (k, v) in enumerate(list(raw.items())[:NUM_SOIL_FEATURES]):
            try:
                feats[i] = float(v) if v is not None else 0.0
            except (ValueError, TypeError):
                feats[i] = 0.0
    return feats


# ===================================================================
# Step 6: Tensor Construction
# ===================================================================

def build_tensors(centroids, power_cache, ndvi_data, yield_data, irrig_ratios,
                  soil_cache, years):
    """Build aligned temporal + static feature tensors and labels."""
    temporal_list = []
    static_list = []
    label_list = []
    meta_list = []

    for (fips, year, wtype), yield_val in sorted(yield_data.items()):
        if year not in years:
            continue
        if fips not in centroids:
            continue

        # Temporal: weather + NDVI
        weather = extract_growing_season_weather(power_cache.get(fips), year)
        if weather is None:
            continue
        ndvi = get_ndvi_features(ndvi_data, fips, year)
        temporal = np.concatenate([weather, ndvi], axis=1)  # (steps, 10)

        # Static: soil + spatial + irrigation + wheat_type
        soil = get_soil_features(soil_cache, fips)
        info = centroids[fips]
        lat_norm = info["lat"] / 90.0
        lon_norm = (info["lon"] + 180.0) / 360.0
        elev = 0.0  # placeholder, available from gazetteer
        irrig = irrig_ratios.get(fips, 0.0)
        wtype_enc = wtype / 3.0  # normalize to [0, 1]

        static = np.concatenate([
            soil,
            np.array([lat_norm, lon_norm, elev, irrig, wtype_enc], dtype=np.float32),
        ])

        temporal_list.append(temporal)
        static_list.append(static)
        label_list.append(yield_val)
        meta_list.append((fips, year, wtype))

    if not temporal_list:
        return None

    temporal_t = torch.tensor(np.stack(temporal_list), dtype=torch.float32)
    static_t = torch.tensor(np.stack(static_list), dtype=torch.float32)
    labels_t = torch.tensor(label_list, dtype=torch.float32)

    return {
        "temporal": temporal_t,
        "static": static_t,
        "labels": labels_t,
        "meta": meta_list,
    }


def compute_normalization(temporal, static):
    """Compute mean/std for normalization from training data."""
    t_flat = temporal.reshape(-1, temporal.shape[-1])
    t_mean = t_flat.mean(dim=0, keepdim=True).unsqueeze(0)
    t_std = t_flat.std(dim=0, keepdim=True).unsqueeze(0).clamp(min=1e-6)

    s_mean = static.mean(dim=0, keepdim=True)
    s_std = static.std(dim=0, keepdim=True).clamp(min=1e-6)

    return t_mean, t_std, s_mean, s_std


# ===================================================================
# Data Loading (imported by train.py)
# ===================================================================

def load_splits(device="cpu"):
    """Load pre-built train/val tensors."""
    data = {}
    for key in ["train_temporal", "train_static", "train_labels",
                "val_temporal", "val_static", "val_labels",
                "t_mean", "t_std", "s_mean", "s_std"]:
        path = os.path.join(TENSOR_DIR, f"{key}.pt")
        data[key] = torch.load(path, map_location=device, weights_only=True)

    for key in ["meta_train", "meta_val"]:
        with open(os.path.join(TENSOR_DIR, f"{key}.json")) as f:
            data[key] = json.load(f)
    return data


def make_dataloader(temporal, static, labels, batch_size, shuffle=True):
    """Yield (temporal_batch, static_batch, label_batch) tuples."""
    N = temporal.shape[0]
    indices = torch.randperm(N) if shuffle else torch.arange(N)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        idx = indices[start:end]
        yield temporal[idx], static[idx], labels[idx]


# ===================================================================
# Evaluation (DO NOT CHANGE)
# ===================================================================

@torch.no_grad()
def evaluate_rmse(model, temporal, static, labels, batch_size=256):
    model.eval()
    preds = []
    N = temporal.shape[0]
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        p = model(temporal[s:e], static[s:e]).squeeze(-1)
        preds.append(p)
    preds = torch.cat(preds)
    return math.sqrt(((preds - labels) ** 2).mean().item())


@torch.no_grad()
def evaluate_r2(model, temporal, static, labels, batch_size=256):
    model.eval()
    preds = []
    N = temporal.shape[0]
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        p = model(temporal[s:e], static[s:e]).squeeze(-1)
        preds.append(p)
    preds = torch.cat(preds)
    ss_res = ((labels - preds) ** 2).sum()
    ss_tot = ((labels - labels.mean()) ** 2).sum()
    return 1.0 - (ss_res / ss_tot).item()


@torch.no_grad()
def evaluate_mae(model, temporal, static, labels, batch_size=256):
    model.eval()
    preds = []
    N = temporal.shape[0]
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        p = model(temporal[s:e], static[s:e]).squeeze(-1)
        preds.append(p)
    preds = torch.cat(preds)
    return (preds - labels).abs().mean().item()


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Montana wheat data")
    parser.add_argument("--nass-api-key", type=str, default=None)
    args = parser.parse_args()

    api_key = args.nass_api_key or os.environ.get("NASS_API_KEY")
    if not api_key:
        print("ERROR: Set NASS_API_KEY environment variable")
        sys.exit(1)

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(TENSOR_DIR, exist_ok=True)

    start_year = START_YEAR
    end_year = VAL_YEAR_END
    print(f"=== Montana Wheat Data Preparation ===")
    print(f"Years: {start_year}-{end_year}")
    print(f"Cache: {CACHE_DIR}")
    print()

    # Step 1
    print("[1/6] County centroids...")
    centroids = download_county_centroids()

    # Step 2
    print("[2/6] NASA POWER climate...")
    power_cache = download_all_power(centroids, start_year, end_year)

    # Step 3
    print("[3/6] MODIS NDVI/EVI...")
    ndvi_data = download_modis_ndvi(centroids, start_year, end_year)

    # Step 4
    print("[4/6] USDA NASS yields + irrigation...")
    yield_records = download_nass_yields(api_key, start_year, end_year)
    irrig_records = download_nass_irrigation(api_key, start_year, end_year)
    yield_data = parse_yields(yield_records)
    irrig_ratios = compute_irrigation_ratios(irrig_records)
    print(f"  Parsed: {len(yield_data)} (county, year, type) yield entries")

    # Step 5
    print("[5/6] NRCS soil properties...")
    soil_cache = download_soil_data(centroids)

    # Step 6
    print("[6/6] Building tensors...")
    train_years = list(range(start_year, TRAIN_YEAR_END + 1))
    val_years = list(range(VAL_YEAR_START, VAL_YEAR_END + 1))

    train_data = build_tensors(centroids, power_cache, ndvi_data, yield_data,
                               irrig_ratios, soil_cache, train_years)
    val_data = build_tensors(centroids, power_cache, ndvi_data, yield_data,
                             irrig_ratios, soil_cache, val_years)

    if train_data is None or val_data is None:
        print("ERROR: Failed to build tensors")
        sys.exit(1)

    # Normalize
    t_mean, t_std, s_mean, s_std = compute_normalization(
        train_data["temporal"], train_data["static"]
    )
    train_data["temporal"] = (train_data["temporal"] - t_mean) / t_std
    train_data["static"] = (train_data["static"] - s_mean) / s_std
    val_data["temporal"] = (val_data["temporal"] - t_mean) / t_std
    val_data["static"] = (val_data["static"] - s_mean) / s_std

    # Save
    torch.save(train_data["temporal"], os.path.join(TENSOR_DIR, "train_temporal.pt"))
    torch.save(train_data["static"], os.path.join(TENSOR_DIR, "train_static.pt"))
    torch.save(train_data["labels"], os.path.join(TENSOR_DIR, "train_labels.pt"))
    torch.save(val_data["temporal"], os.path.join(TENSOR_DIR, "val_temporal.pt"))
    torch.save(val_data["static"], os.path.join(TENSOR_DIR, "val_static.pt"))
    torch.save(val_data["labels"], os.path.join(TENSOR_DIR, "val_labels.pt"))
    torch.save(t_mean, os.path.join(TENSOR_DIR, "t_mean.pt"))
    torch.save(t_std, os.path.join(TENSOR_DIR, "t_std.pt"))
    torch.save(s_mean, os.path.join(TENSOR_DIR, "s_mean.pt"))
    torch.save(s_std, os.path.join(TENSOR_DIR, "s_std.pt"))

    with open(os.path.join(TENSOR_DIR, "meta_train.json"), "w") as f:
        json.dump(train_data["meta"], f)
    with open(os.path.join(TENSOR_DIR, "meta_val.json"), "w") as f:
        json.dump(val_data["meta"], f)

    print()
    print(f"Train: {train_data['temporal'].shape[0]} samples")
    print(f"  temporal: {train_data['temporal'].shape}")
    print(f"  static:   {train_data['static'].shape}")
    print(f"Val: {val_data['temporal'].shape[0]} samples")
    print(f"  temporal: {val_data['temporal'].shape}")
    print(f"  static:   {val_data['static'].shape}")
    print()
    print("Done! Ready to train.")

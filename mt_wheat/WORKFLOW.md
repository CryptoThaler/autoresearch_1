# Montana Wheat Yield Prediction — M4 Pro Workflow

## Quick Start (M4 Pro Mac)
```bash
git clone https://github.com/CryptoThaler/autoresearch_1.git
cd autoresearch_1/mt_wheat
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision numpy requests pandas scipy

# Set API keys
export NASS_API_KEY=your_key_here
# For satellite data, also need NASA Earthdata credentials:
export EARTHDATA_USER=your_username
export EARTHDATA_PASS=your_password

# Phase 1: Download & cache all data (~20 min first run)
python prepare.py

# Phase 2: Train baseline
python train.py

# Phase 3: Autoresearch loop (agent modifies train.py)
# Each iteration: modify train.py -> python train.py -> check val_rmse
```

## API Registration Checklist
1. [x] USDA NASS API Key — https://quickstats.nass.usda.gov/api (instant, free)
2. [ ] NASA Earthdata Account — https://urs.earthdata.nasa.gov/users/new (free registration)
3. [x] NRCS Soils Data Access — no key needed
4. [x] NASA POWER — no key needed
5. [x] US Census Gazetteer — no key needed, direct download

## Data Pipeline (prepare.py)
### Step 1: County Centroids (Census Gazetteer)
- Downloads national county gazetteer ZIP
- Extracts Montana counties (FIPS 30xxx)
- Maps each county to lat/lon centroid
- **54 counties with coordinates**

### Step 2: NASA POWER Climate Data
- Queries per-county-centroid (not state-level!)
- 8 daily weather parameters: T2M, T2M_MAX, T2M_MIN, PRECTOTCORR, ALLSKY_SFC_SW_DWN, RH2M, WS2M, T2MDEW
- Aggregated to 12 biweekly time steps (Apr-Sep)
- **54 API calls, cached to JSON**

### Step 3: MODIS NDVI/EVI (NASA AppEEARS)
- Submits area extraction for all Montana counties
- MOD13Q1.061 product (250m, 16-day NDVI/EVI composites)
- Extracts county-mean NDVI and EVI per biweekly step
- **1 batch request, ~10 min processing on NASA servers**
- Falls back to synthetic features if no Earthdata account yet

### Step 4: USDA NASS Yield Data
- County-level wheat yields (bu/acre) 2000-2023
- Separate queries for Spring Wheat, Winter Wheat, Durum
- Also pulls irrigated vs. non-irrigated breakdowns
- Also pulls irrigated acreage ratios
- **~7,450 yield records across 54 counties**

### Step 5: NRCS Soil Properties
- SQL queries to SoilsDataAccess REST API
- Per-county aggregated soil: AWC, organic matter, clay/sand/silt %, pH, K-factor, CEC, drainage class
- **54 queries, cached to JSON**

### Step 6: Feature Tensor Construction
- Temporal features: (N, 12, 10) — 12 biweekly steps x (8 weather + 2 satellite)
- Static features: (N, 15) — 10 soil + lat + lon + elevation + irrig_ratio + wheat_type_encoded
- Labels: (N,) — yield in bu/acre
- Normalize using training data stats
- Train/val split by year (<=2020 train, 2021-2023 val)

## Model Architecture (train.py)
- Temporal branch: 1D Conv or GRU over 12 time steps
- Static branch: MLP on soil + spatial features
- Fusion: concatenate temporal summary + static features -> prediction head
- County embedding (learnable) for unobserved local factors
- Year embedding for technology trend
- Device: `torch.device("mps")` on M4 Pro

## Commodity Switching
To switch from Montana Wheat to another target:
1. Edit `config.py`: change TARGET_STATE, TARGET_CROP, growing season dates
2. Re-run `python prepare.py` (new data downloads)
3. `python train.py` (retrain)

Examples:
- Montana Barley: TARGET_CROP="BARLEY", same state/season
- Kansas Wheat: TARGET_STATE="20", same crop
- Iowa Corn: TARGET_STATE="19", TARGET_CROP="CORN", season Apr-Oct

"""
Montana Wheat Yield Prediction - Configuration
All constants, API endpoints, and feature definitions.
"""

# ---------------------------------------------------------------------------
# Target Configuration
# ---------------------------------------------------------------------------
TARGET_STATE = "30"          # Montana FIPS
TARGET_STATE_NAME = "Montana"
TARGET_CROP = "WHEAT"
TARGET_COMMODITY_TYPES = [
    "WHEAT, SPRING, (EXCL DURUM)",
    "WHEAT, WINTER",
    "WHEAT, SPRING, DURUM",
]
TIME_BUDGET = 300            # training time budget (seconds)

# ---------------------------------------------------------------------------
# Temporal Configuration
# ---------------------------------------------------------------------------
START_YEAR = 2000
TRAIN_YEAR_END = 2020
VAL_YEAR_START = 2021
VAL_YEAR_END = 2023

# Growing season for Montana wheat: April-August (shorter than corn)
GROWING_SEASON_START_MONTH = 4   # April
GROWING_SEASON_END_MONTH = 9     # September
BIWEEKLY_STEPS = 12             # ~24 weeks Apr-Sep

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
NASS_API_BASE = "https://quickstats.nass.usda.gov/api/api_GET/"
POWER_API_BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"
NRCS_SOILS_API = "https://SDMDataAccess.sc.egov.usda.gov/Tabular/post.rest"
APPEEARS_API_BASE = "https://appeears.earthdatacloud.nasa.gov/api"
CENSUS_GAZETTEER_URL = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_counties_national.zip"

# ---------------------------------------------------------------------------
# NASA POWER Climate Parameters (8 features)
# ---------------------------------------------------------------------------
POWER_PARAMS = [
    "T2M",               # temperature at 2m (C)
    "T2M_MAX",           # daily max temperature (C)
    "T2M_MIN",           # daily min temperature (C)
    "PRECTOTCORR",       # precipitation (mm/day)
    "ALLSKY_SFC_SW_DWN", # solar radiation (MJ/m2/day)
    "RH2M",              # relative humidity (%)
    "WS2M",              # wind speed (m/s)
    "T2MDEW",            # dewpoint temperature (C)
]

# ---------------------------------------------------------------------------
# Satellite Features (MODIS via AppEEARS)
# ---------------------------------------------------------------------------
MODIS_PRODUCTS = {
    "MOD13Q1.061": ["_250m_16_days_NDVI", "_250m_16_days_EVI"],
}

# ---------------------------------------------------------------------------
# Soil Features (NRCS)
# ---------------------------------------------------------------------------
SOIL_FEATURES = [
    "awc",         # available water capacity (cm/cm)
    "om",          # organic matter (%)
    "claypct",     # clay percentage
    "sandpct",     # sand percentage
    "siltpct",     # silt percentage
    "ph",          # soil pH
    "kfact",       # erosion K-factor
    "cec",         # cation exchange capacity
    "drainclass",  # drainage class (encoded)
    "cropindex",   # crop productivity index
]

# ---------------------------------------------------------------------------
# Feature Dimensions
# ---------------------------------------------------------------------------
NUM_WEATHER_FEATURES = len(POWER_PARAMS)       # 8
NUM_SATELLITE_FEATURES = 2                      # NDVI, EVI
NUM_TEMPORAL_FEATURES = NUM_WEATHER_FEATURES + NUM_SATELLITE_FEATURES  # 10
NUM_SOIL_FEATURES = len(SOIL_FEATURES)          # 10
NUM_STATIC_FEATURES = NUM_SOIL_FEATURES + 5     # soil + lat/lon/elev/irrig_ratio/wheat_type
NUM_TOTAL_FEATURES = BIWEEKLY_STEPS * NUM_TEMPORAL_FEATURES + NUM_STATIC_FEATURES

# ---------------------------------------------------------------------------
# Cache Directories
# ---------------------------------------------------------------------------
import os
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "mt-wheat-yield")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TENSOR_DIR = os.path.join(CACHE_DIR, "tensors")

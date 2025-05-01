import pandas as pd
import numpy as np
import os
import joblib
import gc
import warnings
import asyncio # For locking
from fastapi import FastAPI, HTTPException, Body, BackgroundTasks
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any, Optional, Tuple
from datetime import date, datetime, timedelta
from contextlib import asynccontextmanager
from filelock import FileLock
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
import pytz 

try: import xgboost as xgb
except ImportError: print("xgboost not installed"); exit()
try: import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError: print("Optuna not installed"); optuna = None

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
import traceback

# Configuration
MODEL_DIR = "saved_models_xgb"
MODEL_PIPELINE_BASE_PATH = os.path.join(MODEL_DIR, "xgb_price_pipeline")
FEATURES_PATH = os.path.join(MODEL_DIR, "xgb_features_list.joblib") # Expecting XGB feature list

DATA_DIR = "data"
WEATHER_TRAIN_PATH = os.path.join(DATA_DIR, "weather_train_data.csv")
PRICE_TRAIN_PATH = os.path.join(DATA_DIR, "price_train_data.csv")
INCOMING_WEATHER_PATH = os.path.join(DATA_DIR, "incoming_weather_data.csv")
INCOMING_PRICE_PATH = os.path.join(DATA_DIR, "incoming_price_data.csv")

# Retraining Settings
RETRAIN_INTERVAL_HOURS = 24
N_OPTUNA_TRIALS_RETRAIN = 15
OPTUNA_TIMEOUT_RETRAIN = 450

# Minimum history needed to calculate lags
MIN_HISTORY_WEEKS = 15 # max lag of 12 + buffer


model_pipelines: Dict[int, Any] = {}
feature_list: Optional[List[str]] = None


scheduler: Optional[AsyncIOScheduler] = None
retraining_in_progress: bool = False
last_retrain_time: Optional[datetime] = None
last_retrain_status: str = "Not run yet"
model_lock = asyncio.Lock()

TARGET_COL = None # Set during data loading
stored_categorical_features = [] 

# Pydantic Models
class PredictRequest(BaseModel):
    crop: str
    region: str

class PredictionItem(BaseModel):
    prediction_index: int
    date: date
    price: float

class PredictResponse(BaseModel):
    crop: str
    region: str
    predictions: List[PredictionItem]

class WeatherDetails(BaseModel):
    rainfall: Optional[float] = None
    humidity: Optional[float] = None
    temp: Optional[float] = None

class WeatherDataPost(BaseModel):
    date: date
    region: str
    weatherData: WeatherDetails

class PriceDetails(BaseModel):
    price: float

class PriceDataPost(BaseModel):
    date: date
    crop: str
    region: str
    priceData: PriceDetails


# Function to clean column names
def clean_api_col_names(cols):
    new_cols = []
    for col in cols:
        new_col = str(col).lower()
        # Keep only alphanumeric characters and underscores
        new_col = ''.join(e for e in new_col if e.isalnum() or e == '_')

        while '__' in new_col:
            new_col = new_col.replace('__', '_')

        new_col = new_col.strip('_')
        new_cols.append(new_col)
    return new_cols

# Data Loading and traning functions

def load_and_preprocess_data_agg(for_predict=False):

    log_prefix = "Predict Data" if for_predict else "Retraining [XGB]"
    print(f"{log_prefix}: Loading available data...")
    all_weather_dfs = []
    all_price_dfs = []

    # Load original
    try:
        dfw_o = pd.read_csv(WEATHER_TRAIN_PATH, parse_dates=['Date'])
        dfp_o = pd.read_csv(PRICE_TRAIN_PATH, parse_dates=['Date'])
        dfw_o.columns = clean_api_col_names(dfw_o.columns)
        dfp_o.columns = clean_api_col_names(dfp_o.columns)

        if 'date' not in dfw_o.columns or 'date' not in dfp_o.columns:
            raise ValueError("Orig missing date")
        dfw_o['date'] = pd.to_datetime(dfw_o['date'])
        dfp_o['date'] = pd.to_datetime(dfp_o['date'])
        all_weather_dfs.append(dfw_o)
        all_price_dfs.append(dfp_o)

        if not for_predict:
            print(f"Loaded original W={len(dfw_o)}, P={len(dfp_o)}")
    except Exception as e:
        raise RuntimeError(f"{log_prefix}: Cannot load original data: {e}")
    
    # Load incoming weather
    if os.path.exists(INCOMING_WEATHER_PATH):
        try:
            with FileLock(INCOMING_WEATHER_PATH+".lock"):
                dfw_n=pd.read_csv(INCOMING_WEATHER_PATH,parse_dates=['date'])
                dfw_n.columns=clean_api_col_names(dfw_n.columns)
            if 'date' in dfw_n.columns:
                dfw_n['date']=pd.to_datetime(dfw_n['date'])
                all_weather_dfs.append(dfw_n)
                print(f"{log_prefix}: Loaded incoming weather: {len(dfw_n)} rows")
        except Exception as e:
            print(f"Warn: Load incoming weather failed: {e}")

    # Load incoming price
    if os.path.exists(INCOMING_PRICE_PATH):
         try:
            with FileLock(INCOMING_PRICE_PATH+".lock"): dfp_n=pd.read_csv(INCOMING_PRICE_PATH,parse_dates=['date']); dfp_n.columns=clean_api_col_names(dfp_n.columns);
            if 'date' in dfp_n.columns:
                dfp_n['date']=pd.to_datetime(dfp_n['date'])
                if 'crop' in dfp_n.columns and 'commodity' not in dfp_n.columns: dfp_n=dfp_n.rename(columns={'crop':'commodity'})
                all_price_dfs.append(dfp_n); print(f"{log_prefix}: Loaded incoming price: {len(dfp_n)} rows")
         except Exception as e: print(f"Warn: Load incoming price failed: {e}")

    if not all_weather_dfs or not all_price_dfs:
        raise RuntimeError(f"{log_prefix}: No data available.")

    dfw_full = pd.concat(all_weather_dfs, ignore_index=True)
    dfp_full = pd.concat(all_price_dfs, ignore_index=True)
    del all_weather_dfs, all_price_dfs
    gc.collect()

    global TARGET_COL
    TARGET_COL = 'priceperunitsilverdrachmakg' # Ensure TARGET_COL is set

    if 'date' not in dfw_full.columns or 'date' not in dfp_full.columns:
        raise ValueError("Combined missing 'date'.")

    print(f"{log_prefix}: Handling duplicates...")
    initial_w = len(dfw_full)
    initial_p = len(dfp_full)

    if not all(c in dfw_full.columns for c in ['date', 'region']):
        raise ValueError("Weather missing keys pre-dedup.")
    dfw_full = dfw_full.drop_duplicates(['date', 'region'], keep='last') # Keep latest entry if duplicate date/region
    if initial_w > len(dfw_full):
        print(f"{log_prefix}: Removed {initial_w - len(dfw_full)} weather dups.")

    if 'commodity' not in dfp_full.columns:
        if 'crop' in dfp_full.columns:
            dfp_full = dfp_full.rename(columns={'crop': 'commodity'})
        else:
            raise ValueError("Price missing 'commodity'/'crop'.")

    if not all(c in dfp_full.columns for c in ['date', 'region', 'commodity']):
        raise ValueError("Price missing keys pre-dedup.")
    dfp_full = dfp_full.drop_duplicates(['date', 'region', 'commodity'], keep='last') # Keep latest entry
    if initial_p > len(dfp_full):
        print(f"{log_prefix}: Removed {initial_p - len(dfp_full)} price dups.")

    # Fill missing columns
    for col, d in [('temperaturek', np.nan), ('rainfallmm', np.nan), ('humidity', np.nan), ('cropyieldimpactscore', np.nan)]:
        if col not in dfw_full.columns:
            dfw_full[col] = d
    if 'type' not in dfp_full.columns:
        dfp_full['type'] = "Unknown"

    print(f"{log_prefix}: Merging...")
    if not all(c in dfw_full.columns for c in ['date', 'region']) or not all(c in dfp_full.columns for c in ['date', 'region']):
        raise ValueError("Missing merge keys.")

    df_m = pd.merge(dfw_full, dfp_full, on=['date', 'region'], how='inner')
    if df_m.empty:
        print(f"{log_prefix} Warn: Merged empty.")
        return None

    print(f"{log_prefix}: Sorting...")
    if not all(c in df_m.columns for c in ['region', 'commodity', 'date']):
        raise ValueError("Missing sort keys.")
    df_m = df_m.sort_values(['region', 'commodity', 'date']).reset_index(drop=True)

    print(f"{log_prefix}: Aggregated data ready. Shape: {df_m.shape}")
    del dfw_full, dfp_full
    gc.collect()
    return df_m

def engineer_features(df, max_forecast_horizon=4, generate_targets=True):
    
    mode = "Train" if generate_targets else "Predict"
    print(f"Feature Eng [{mode}]: Starting...")
    if df is None or df.empty:
        print(f"Feature Eng [{mode}]: Input DataFrame is empty or None.")
        return None, [] if generate_targets else None

    if TARGET_COL is None or TARGET_COL not in df.columns:
        print(f"Error: TARGET_COL '{TARGET_COL}' not found in DataFrame for feature engineering.")
        return None, [] if generate_targets else None

    df_feat = df.copy()

    # Ensure 'date' is datetime
    if 'date' not in df_feat.columns or not pd.api.types.is_datetime64_any_dtype(df_feat['date']):
        print(f"Feature Eng [{mode}]: 'date' column missing or not datetime. Attempting conversion.")
        try:
            df_feat['date'] = pd.to_datetime(df_feat['date'])
        except Exception as e:
            print(f"Feature Eng [{mode}]: Error converting 'date' to datetime: {e}")
            return None, [] if generate_targets else None

    # Sort data before creating lags
    df_feat = df_feat.sort_values(['region', 'commodity', 'date']).reset_index(drop=True)

    # Time Features
    print(f"Feature Eng [{mode}]: Creating time features...")
    df_feat['year'] = df_feat['date'].dt.year
    df_feat['month'] = df_feat['date'].dt.month
    df_feat['weekofyear'] = df_feat['date'].dt.isocalendar().week.astype(int)
    df_feat['dayofweek'] = df_feat['date'].dt.dayofweek

    # Lag Features
    print(f"Feature Eng [{mode}]: Creating lag/rolling features...")
    cols_to_lag = [TARGET_COL, 'temperaturek', 'rainfallmm', 'humidity', 'cropyieldimpactscore']
    lags = [1, 2, 3, 4, 8, 12] # Weeks
    rolls = [4, 8, 12] # Weeks

    # Filter cols_to_lag 
    cols_to_lag_present = [col for col in cols_to_lag if col in df_feat.columns]
    if not cols_to_lag_present:
        print(f"Feature Eng [{mode}]: Warning - None of the specified lag columns found.")
    else:
        print(f"Feature Eng [{mode}]: Lagging columns: {cols_to_lag_present}")

    # Group once for efficiency
    grouped = df_feat.groupby(['region', 'commodity'], observed=True, group_keys=False) 

    for col in cols_to_lag_present:
        # Lags 
        for lag in lags:
            df_feat[f'{col}_lag_{lag}w'] = grouped[col].shift(lag)

        shifted_col = grouped[col].shift(1)
        for win in rolls:
            min_p = max(1, win // 2) 
            df_feat[f'{col}_roll_mean_{win}w'] = shifted_col.rolling(window=win, min_periods=min_p).mean()
            df_feat[f'{col}_roll_std_{win}w'] = shifted_col.rolling(window=win, min_periods=min_p).std()

    del grouped, shifted_col 
    gc.collect()

    target_cols = []
    if generate_targets:
        print("Feature Eng [Train]: Generating targets...")

        # Group for target generation shift
        grouped_target = df_feat.groupby(['region', 'commodity'], observed=True, group_keys=False)
        for h in range(1, max_forecast_horizon + 1):
            t_name = f'target_{TARGET_COL}_{h}w_ahead'
            target_cols.append(t_name)

            # Shift target backwards
            df_feat[t_name] = grouped_target[TARGET_COL].shift(-h)
        del grouped_target
        gc.collect()

        # Drop rows where the longest horizon target is NaN 
        longest_target = f'target_{TARGET_COL}_{max_forecast_horizon}w_ahead'
        if longest_target in df_feat.columns:
            rows_before = len(df_feat)
            df_feat = df_feat.dropna(subset=[longest_target])
            rows_after = len(df_feat)
            print(f"Feature Eng [Train]: Dropped {rows_before - rows_after} rows due to missing target '{longest_target}'.")
        else:
             print(f"Feature Eng [Train]: Warning - Longest target column '{longest_target}' not found for dropping NaNs.")


    # Exclude target columns from filling if they exist
    feature_columns = [col for col in df_feat.columns if col not in target_cols and col != 'date'] # Keep date
    feature_columns_exist = [col for col in feature_columns if col in df_feat.columns]

    nan_counts = df_feat[feature_columns_exist].isnull().sum()
    total_nans = nan_counts.sum()

    if total_nans > 0:
        print(f"Feature Eng [{mode}]: Filling {total_nans} NaNs in feature columns with 0...")
        
        df_feat[feature_columns_exist] = df_feat[feature_columns_exist].fillna(0) 
    else:
        print(f"Feature Eng [{mode}]: No NaNs found in feature columns.")

    # check for categorical types 
    cat_cols = df_feat.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if col not in ['region', 'commodity', 'type']: 
             print(f"Feature Eng [{mode}]: Unexpected object/category column '{col}'. Converting to string.")
             df_feat[col] = df_feat[col].astype(str)

    print(f"Feature Eng [{mode}] complete. Final shape: {df_feat.shape}")

    if generate_targets:
        return df_feat, target_cols # Return tuple for training
    else:
        # For prediction return the features DataFrame
        return df_feat

# Training Components 
def select_features_and_target(df, target_col_name, all_target_cols): 

    if df is None or target_col_name not in df.columns:
        return None, None, None
    y = df[target_col_name]
    cols_exclude = all_target_cols + ['date']
    if TARGET_COL in df.columns:
        cols_exclude.append(TARGET_COL)
    f_list = [col for col in df.columns if col not in cols_exclude]
    f_list = [f for f in f_list if f in df.columns] # Ensure features exist

    if not f_list:
        return None, None, None
    X = df[f_list]

    if not os.path.exists(FEATURES_PATH): # Save only if doesn't exist
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            joblib.dump(f_list, FEATURES_PATH)
            print("XGB: Saved feature list.")
        except Exception as e:
            print(f"XGB Error saving features: {e}")
    return X, y, f_list

def build_preprocessor(X): 

    if X is None or X.empty:
        return None
    cat_feats = X.select_dtypes(['object', 'category']).columns.tolist()
    num_feats = X.select_dtypes(np.number).columns.tolist()
    global stored_categorical_features
    stored_categorical_features = cat_feats.copy()

    if not cat_feats and not num_feats:
        return None
    transformers = []

    if num_feats:
        transformers.append(('num', StandardScaler(), num_feats))
    if cat_feats:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_feats))
    if not transformers:
        return None
    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
    return preprocessor

def xgboost_objective(trial, X, y, preprocessor):
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': trial.suggest_float('eta', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'seed': 42,
        'nthread': -1
    }
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    split_idx = int(len(X) * 0.8)
    X_train_p, y_train_p = X.iloc[:split_idx], y.iloc[:split_idx]
    X_val_p, y_val_p = X.iloc[split_idx:], y.iloc[split_idx:]

    if X_val_p.empty:
        return float('inf')

    try:
        preprocessor_trial = clone(preprocessor)
        preprocessor_fitted = preprocessor_trial.fit(X_train_p)
        X_train_proc = preprocessor_fitted.transform(X_train_p)
        X_val_proc = preprocessor_fitted.transform(X_val_p)
    except Exception as e:
        print(f"XGB Optuna Preprocessing Error: {e}")
        return float('inf')

    model = xgb.XGBRegressor(**xgb_params, n_estimators=1000, early_stopping_rounds=50)

    try:
        model.fit(X_train_proc, y_train_p, eval_set=[(X_val_proc, y_val_p)], verbose=False)
    except Exception as e:
        print(f"XGB Optuna Fitting Error: {e}")
        return float('inf')

    preds = model.predict(X_val_proc)
    rmse = np.sqrt(mean_squared_error(y_val_p, preds))
    trial.report(rmse, step=0)

    if trial.should_prune():
        raise optuna.TrialPruned()

    return rmse

def optimize_and_train_xgb_pipeline(X_train, y_train, preprocessor, forecast_horizon, n_trials, timeout):

    pipeline_save_path = f"{MODEL_PIPELINE_BASE_PATH}_{forecast_horizon}w.joblib"
    print(f"Retrain [XGB]: Optimizing & Training H{forecast_horizon}w...")
    if X_train is None or y_train is None or preprocessor is None:
        return None, -1
    best_params = {}
    best_value = -1
    if optuna:
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
        try:
            study.optimize(lambda trial: xgboost_objective(trial, X_train, y_train, preprocessor), n_trials=n_trials, timeout=timeout)
        except Exception as e:
            print(f"XGB Optuna error H{forecast_horizon}: {e}. Defaults.")
            best_params = {}
            best_value = float('inf')
        else:
            best_params = study.best_params
            best_value = study.best_value
            print(f"XGB Optuna H{forecast_horizon} Best RMSE: {best_value:.4f}")
    else:
        print("Optuna not installed, using defaults.")

    final_params = best_params.copy()
    final_params.setdefault('objective', 'reg:squarederror')
    final_params.setdefault('eval_metric', 'rmse')
    final_params.setdefault('seed', 42 + forecast_horizon)
    final_params.setdefault('nthread', -1)
    final_params.setdefault('n_estimators', 1000)

    final_model = xgb.XGBRegressor(**final_params)

    try:
        preprocessor_fitted = preprocessor.fit(X_train, y_train)
    except Exception as e:
        print(f"XGB Error fit preproc H{forecast_horizon}: {e}")
        return None, -1

    pipeline = Pipeline(steps=[('preprocessor', preprocessor_fitted), ('regressor', final_model)])

    try:
        pipeline.fit(X_train, y_train)
    except Exception as e:
        print(f"XGB Error fit pipeline H{forecast_horizon}: {e}")
        return None, -1

    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(pipeline, pipeline_save_path)
    except Exception as e:
        print(f"XGB Error save pipeline H{forecast_horizon}: {e}")
        return None, -1

    print(f"XGB: Saved H{forecast_horizon} pipeline to '{pipeline_save_path}'")
    del pipeline, X_train, y_train, preprocessor, final_model, study
    gc.collect()
    return pipeline_save_path, best_value

# Retraining Job Function 
async def run_retraining_job():

    global retraining_in_progress, model_pipelines, feature_list, last_retrain_time, last_retrain_status
    if retraining_in_progress:
        print("Retraining job: Already in progress.")
        return

    retraining_in_progress = True
    last_retrain_status = "Running"
    print("\n--- Starting XGBoost Retraining Job ---")
    newly_trained_pipelines = {}
    training_successful = True
    start_time = datetime.now(pytz.utc)
    current_feature_list = None

    try:
        aggregated_data = load_and_preprocess_data_agg()
        if aggregated_data is None:
            raise ValueError("Data aggregation failed.")

        # Engineer features *with targets* for training
        featured_data, target_cols = engineer_features(aggregated_data, generate_targets=True)
        if featured_data is None:
            raise ValueError("Feature engineering failed.")
        del aggregated_data
        gc.collect()

        # Load or Define Feature List
        if os.path.exists(FEATURES_PATH):
            current_feature_list = joblib.load(FEATURES_PATH)
        elif feature_list:
            current_feature_list = feature_list
        else:
            print("Warning: Feature list file not found. Will generate.")

        # Train loop
        for h in range(1, 5):
            target_col_name = f'target_{TARGET_COL}_{h}w_ahead'
            if target_col_name not in featured_data.columns:
                continue

            X_train, y_train, temp_feat_list = select_features_and_target(featured_data, target_col_name, target_cols)
            if X_train is None:
                training_successful = False
                break

            # Ensure feature list consistency
            if current_feature_list:
                X_train = X_train[current_feature_list] # Use consistent list
            elif temp_feat_list:
                current_feature_list = temp_feat_list
                feature_list = current_feature_list # Save first generated list globally
            else:
                print("Error: Cannot determine feature list.")
                training_successful = False
                break

            preprocessor_unfitted = build_preprocessor(X_train)
            if preprocessor_unfitted is None:
                training_successful = False
                break

            # Call XGBoost training function
            model_path, _ = optimize_and_train_xgb_pipeline(
                X_train, y_train, preprocessor_unfitted,
                forecast_horizon=h,
                n_trials=N_OPTUNA_TRIALS_RETRAIN,
                timeout=OPTUNA_TIMEOUT_RETRAIN
            )

            if model_path:
                newly_trained_pipelines[h] = joblib.load(model_path)
            else:
                training_successful = False
                break

            del X_train, y_train, preprocessor_unfitted
            gc.collect()

        del featured_data
        gc.collect()

    except Exception as e:
        print(f"ERROR during XGB retraining: {e}")
        training_successful = False
        traceback.print_exc()
        last_retrain_status = f"Failed: {e}"

    finally:
        end_time = datetime.now(pytz.utc)
        if training_successful and len(newly_trained_pipelines) == 4:
            print("XGB Retraining successful. Updating models in memory...")
            async with model_lock:
                global model_pipelines
                model_pipelines.clear()
                model_pipelines.update(newly_trained_pipelines)
                last_retrain_status = f"Success at {end_time.isoformat()}"
                last_retrain_time = end_time
                print("Global XGBoost models updated.")
        else:
            print("XGB Retraining failed/incomplete. Models NOT updated.")
            last_retrain_status = f"Failed/Incomplete at {end_time.isoformat()}"

        retraining_in_progress = False
        gc.collect()
        print(f"--- XGBoost Retraining Job Finished (Duration: {end_time - start_time}) ---")


# Get Latest Features for Rolling Forecast

def get_latest_features_for_item(region: str, crop: str, full_data: pd.DataFrame) -> Optional[Tuple[pd.DataFrame, pd.Timestamp]]: # Return Tuple (features, latest_date)

    print(f"Predict Helper: Getting features for {region} - {crop}")
    commodity_internal = crop
    item_data = full_data[(full_data['region'] == region) & (full_data['commodity'] == commodity_internal)].copy()

    if item_data.empty:
        print(f"Predict Helper: No data found.")
        return None, None
    latest_date = item_data['date'].max() # Get latest date *before* feature engineering
    print(f"Predict Helper: Found {len(item_data)} rows. Latest date: {latest_date.strftime('%Y-%m-%d')}")
    if len(item_data) < MIN_HISTORY_WEEKS:
        print(f"Predict Helper: Insufficient history.")
        return None, None

    item_features_df = engineer_features(item_data, generate_targets=False)
    if item_features_df is None or item_features_df.empty:
        print(f"Predict Helper: Feature engineering failed.")
        return None, None

    # Get the row corresponding to the latest date 
    if 'date' not in item_features_df.columns:
         print("Predict Helper: Error - 'date' column missing after feature engineering.")
         return None, None
    latest_feature_row_full = item_features_df[item_features_df['date'] == latest_date]

    if latest_feature_row_full.empty:
        print(f"Predict Helper: Could not find features for latest date {latest_date}. Using last row.")
        latest_feature_row_full = item_features_df.iloc[-1:]
        if latest_feature_row_full.empty:
            print("Predict Helper: Feature DataFrame empty.")
            return None, None
        # Update latest_date if had to use fallback
        latest_date = latest_feature_row_full['date'].iloc[0]

    # Ensure Columns Match Trained Features 
    if not feature_list:
        raise RuntimeError("Predict Helper: Global feature_list not loaded.")
    try:
        features_to_predict = latest_feature_row_full[feature_list].copy()
    except KeyError as e:
        print(f"Predict Helper: Error - Feature mismatch: {e}")
        return None, None

    # Ensure correct types
    for col in features_to_predict.select_dtypes(include=['object']).columns:
        if col in ['region', 'commodity', 'type']:
            features_to_predict[col] = features_to_predict[col].astype(str)

    print(f"Predict Helper: Generated feature row for prediction.")
    return features_to_predict.head(1), latest_date


# FastAPI Lifespan loads XGBoost models
@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Application startup: Loading XGBoost resources...")
    global model_pipelines, feature_list, scheduler, last_retrain_status, last_retrain_time
    models_loaded_count = 0
    try:
        if os.path.exists(FEATURES_PATH):
            feature_list = joblib.load(FEATURES_PATH)
            print(f"Loaded XGB feature list ({len(feature_list)} features).")
        else:
            print(f"Warning: XGB Feature list not found at {FEATURES_PATH}")
        for h in range(1, 5):
            model_path = f"{MODEL_PIPELINE_BASE_PATH}_{h}w.joblib"
            if os.path.exists(model_path):
                try:
                    model_pipelines[h] = joblib.load(model_path)
                    print(f"Loaded initial XGB model: {h}w")
                    models_loaded_count += 1
                except Exception as load_e:
                    print(f"Error loading XGB model {h}w: {load_e}")
            else:
                print(f"Warning: Initial XGB model missing: {model_path}")
        if models_loaded_count < 4:
            print(f"Warning: Loaded only {models_loaded_count}/4 initial XGB models.")

    except Exception as e:
        print(f"FATAL STARTUP ERROR loading models/features: {e}")
        model_pipelines = {}
        feature_list = None
    print("Setting up APScheduler for XGBoost retraining...")
    scheduler = AsyncIOScheduler(timezone=str(pytz.utc))
    scheduler.add_job(
        run_retraining_job,
        trigger=IntervalTrigger(hours=RETRAIN_INTERVAL_HOURS, timezone=pytz.utc),
        id="retraining_job",
        name="XGBoost Retraining",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=3600
    )

    scheduler.start()
    print(f"Scheduler started. Retraining every {RETRAIN_INTERVAL_HOURS} hours.")

    last_retrain_status = "Scheduled"
    yield # App runs
    print("Application shutdown: Stopping scheduler...")

    if scheduler and scheduler.running:
        scheduler.shutdown()
    print("Cleaning up globals...")
    model_pipelines = {}
    feature_list = None
    scheduler = None
    gc.collect()

# FastAPI App Instance
app = FastAPI(title="Crop Price Prediction API [XGBoost v2 Rolling]", version="2.0.1", lifespan=lifespan)

# API Endpoints 
@app.get("/api/status", tags=["Status"])
async def get_status():

    next_run = None
    job_status = "Scheduler not active"

    if scheduler and scheduler.running:
        job = scheduler.get_job("retraining_job")
        job_status = "Scheduled" if job else "Not scheduled"
        if job and job.next_run_time:
            next_run = job.next_run_time.isoformat()
    current_retrain_status = last_retrain_status if not retraining_in_progress else "Running"
    return {
        "status": "ok",
        "active_model_type": "XGBoost",
        "models_loaded_horizons": list(model_pipelines.keys()),
        "features_loaded": feature_list is not None,
        "retraining_status": current_retrain_status,
        "retraining_in_progress": retraining_in_progress,
        "last_retrain_time": last_retrain_time.isoformat() if last_retrain_time else None,
        "scheduled_next_run": next_run
    }

# /api/predict Endpoint for ROLLING Forecast 
@app.post("/api/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_multi_step_price(request: PredictRequest):

    if not feature_list: raise HTTPException(status_code=503, detail="Feature list not loaded. Cannot generate features.")
    async with model_lock: current_pipelines = model_pipelines.copy() # Read models under lock
    if len(current_pipelines) < 4: raise HTTPException(status_code=503, detail="Missing required forecast models.")

    print(f"\n-> Predict request: Region='{request.region}', Crop='{request.crop}'")

    # Load ALL current data
    try:
        full_data = load_and_preprocess_data_agg(for_predict=True)
        if full_data is None: raise ValueError("Failed to load aggregated data for prediction.")
    except Exception as e:
        print(f"Error loading aggregated data during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal error loading data for prediction.")

    # Get latest features
    try:
        features_df, latest_date = get_latest_features_for_item(request.region, request.crop, full_data)
        del full_data; gc.collect() # Free memory

        if features_df is None or features_df.empty or latest_date is None:
            raise HTTPException(status_code=404, detail=f"Cannot predict: No data or insufficient history found for Region '{request.region}' and Crop '{request.crop}'. Minimum {MIN_HISTORY_WEEKS} weeks required, or feature generation failed.")

        print(f"-> Features generated for latest date: {latest_date.strftime('%Y-%m-%d')}")

    except HTTPException as he: raise he # Propagate 404
    except Exception as e:
        print(f"Error generating features for prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal error generating features.")

    # Predict across horizons using the generated feature row
    predictions_list: List[PredictionItem] = []
    print("-> Predicting horizons...")

    with warnings.catch_warnings():
         warnings.simplefilter("ignore", category=UserWarning) # Suppress XGB feature name warnings
         for h in range(1, 5):
             # Use the latest_date 
             forecast_dt = latest_date + pd.Timedelta(weeks=h)
             pipeline_h = current_pipelines.get(h)
             if not pipeline_h: print(f"Warn: Model {h}w missing!"); continue
             try:
                  pred = pipeline_h.predict(features_df) # Predict using the single row
                  value = round(float(pred[0]), 2)
                  predictions_list.append(PredictionItem(prediction_index=h-1, date=forecast_dt.date(), price=value))
             except Exception as pred_e: print(f"Error predicting H{h}: {pred_e}")

    return PredictResponse(crop=request.crop, region=request.region, predictions=predictions_list)

# POST /api/data/weather Endpoint for Weather Data Ingestion
@app.post("/api/data/weather", status_code=201, tags=["Data Ingestion"])
async def add_weather_data(data: WeatherDataPost = Body(...)):

    print(f"\n-> Received weather: Region={data.region}, Date={data.date}")
    record = {"date": data.date, "region": data.region, "temperaturek": data.weatherData.temp,
              "rainfallmm": data.weatherData.rainfall, "humidity": data.weatherData.humidity, "cropyieldimpactscore": None }
    new_data_df = pd.DataFrame([record]); new_data_df['date'] = pd.to_datetime(new_data_df['date'])

    lock_path = INCOMING_WEATHER_PATH + ".lock"

    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with FileLock(lock_path):
            file_exists = os.path.exists(INCOMING_WEATHER_PATH)
            new_data_df.to_csv(INCOMING_WEATHER_PATH, mode='a', header=not file_exists, index=False)
        return {"message": "Weather data stored successfully"}
    except Exception as save_e:
        print(f"Error saving weather data: {save_e}")
        raise HTTPException(status_code=500, detail="Failed to store weather data.")

# POST /api/data/prices Endpoint for Price Data Ingestion
@app.post("/api/data/prices", status_code=201, tags=["Data Ingestion"])
async def add_price_data(data: PriceDataPost = Body(...)):

    print(f"\n-> Received price: Region={data.region}, Crop={data.crop}, Date={data.date}")
    record = {"date": data.date, "region": data.region, "commodity": data.crop,
              "priceperunitsilverdrachmakg": data.priceData.price, "type": "Unknown"}
    new_data_df = pd.DataFrame([record]); new_data_df['date'] = pd.to_datetime(new_data_df['date'])

    lock_path = INCOMING_PRICE_PATH + ".lock"

    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with FileLock(lock_path):
            file_exists = os.path.exists(INCOMING_PRICE_PATH)
            new_data_df.to_csv(INCOMING_PRICE_PATH, mode='a', header=not file_exists, index=False)
        return {"message": "Price data stored successfully"}
    except Exception as save_e:
        print(f"Error saving price data: {save_e}")
        raise HTTPException(status_code=500, detail="Failed to store price data.")

# POST /api/retrain Endpoint for Manual Retraining Trigger
@app.post("/api/retrain", status_code=202, tags=["Maintenance"])
async def trigger_manual_retraining(background_tasks: BackgroundTasks):

     global retraining_in_progress

     if retraining_in_progress: raise HTTPException(status_code=409, detail="Retraining is already in progress.")

     print("Manual XGBoost retraining trigger received.")

     background_tasks.add_task(run_retraining_job)
     return {"message": "XGBoost retraining process initiated in the background."}

# To Run 
# --------------------------------------------------------------------------- #
# uvicorn main:app --reload --host 0.0.0.0 --port 8000

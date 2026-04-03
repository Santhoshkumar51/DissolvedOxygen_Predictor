import numpy as np
import pandas as pd
import joblib
import os
from tensorflow.keras.models import load_model
from model import SEQUENCE_LEN, SumOverTime

MODEL_PATH    = "models/bisru_model.keras"
SCALER_PATH   = "models/scaler.pkl"
FEATURES_PATH = "models/selected_features.pkl"
TARGET        = "dissolved_oxygen"
DO_THRESHOLD  = 5.0
ALL_FEATURES  = ["temperature", "pH", "BOD", "ammonia", "nitrate", "nitrogen"]
RAW_PATH      = "data/raw/Combined_dataset.csv"

COLUMN_ALIASES = {
    "temperature (cel)"                : "temperature",
    "temp"                             : "temperature",
    "water temperature"                : "temperature",
    "ph (ph units)"                    : "pH",
    "ph"                               : "pH",
    "biochemical oxygen demand (mg/l)" : "BOD",
    "bod"                              : "BOD",
    "bod (mg/l)"                       : "BOD",
    "ammonia (mg/l)"                   : "ammonia",
    "ammonia"                          : "ammonia",
    "nh3"                              : "ammonia",
    "nitrate (mg/l)"                   : "nitrate",
    "nitrate"                          : "nitrate",
    "no3"                              : "nitrate",
    "nitrogen (mg/l)"                  : "nitrogen",
    "nitrogen"                         : "nitrogen",
    "dissolved oxygen (mg/l)"          : "dissolved_oxygen",
    "dissolved oxygen"                 : "dissolved_oxygen",
    "do (mg/l)"                        : "dissolved_oxygen",
    "do"                               : "dissolved_oxygen",
    "date"                             : "timestamp",
}

# Meta columns: lowercase key → standard name used in result DataFrame
META_LOOKUP = {
    "country"        : "country",
    "area"           : "area",
    "waterbody type" : "waterbody_type",
    "waterbody_type" : "waterbody_type",
}


def load_artifacts():
    model    = load_model(MODEL_PATH, custom_objects={"SumOverTime": SumOverTime})
    scaler   = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, scaler, features


def normalize_columns(df: pd.DataFrame):
    renamed = {col: COLUMN_ALIASES.get(col.strip().lower(), col.strip().lower())
               for col in df.columns}
    df      = df.rename(columns=renamed)
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    return df, missing


def extract_meta(raw_df: pd.DataFrame) -> dict:
    meta = {}

    for col in raw_df.columns:
        # normalize aggressively
        key = (
            col.strip()
               .lower()
               .replace("_", " ")
               .replace("\xa0", " ")   # handles hidden spaces
               .strip()
        )

        if key in META_LOOKUP:
            meta_name = META_LOOKUP[key]
            meta[meta_name] = raw_df[col].reset_index(drop=True)

    print("META FOUND:", meta.keys())  # debug

    return meta


def preprocess_upload(df: pd.DataFrame, scaler) -> pd.DataFrame:
    df = df.copy()
    df[ALL_FEATURES] = df[ALL_FEATURES].fillna(df[ALL_FEATURES].median())
    cols_to_scale    = ALL_FEATURES + ([TARGET] if TARGET in df.columns else [])
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    return df


def make_sequences(df: pd.DataFrame, features: list):
    values = df[features].values
    return np.array(
        [values[i : i + SEQUENCE_LEN] for i in range(len(df) - SEQUENCE_LEN)],
        dtype="float32"
    )


def inverse_do(scaler, values: np.ndarray) -> np.ndarray:
    dummy = np.zeros((len(values), len(ALL_FEATURES) + 1))
    dummy[:, -1] = values.flatten()
    return scaler.inverse_transform(dummy)[:, -1]


def predict(csv_path: str):
    model, scaler, features = load_artifacts()

    raw_df      = pd.read_csv(csv_path, low_memory=False)
    df, missing = normalize_columns(raw_df.copy())

    if missing:
        return None, missing

    # ── Grab Country / Area / Waterbody Type from raw BEFORE any processing ──
    meta = extract_meta(raw_df)   # dict: {"country": Series, "area": Series, ...}

    # Raw numeric values for scatter charts (un-scaled)
    chart_cols = [c for c in ALL_FEATURES + [TARGET] if c in df.columns]
    raw_values = df[chart_cols].copy()

    df_scaled = preprocess_upload(df, scaler)

    if len(df_scaled) < SEQUENCE_LEN + 1:
        raise ValueError(f"Need at least {SEQUENCE_LEN + 1} rows. Got {len(df_scaled)}.")

    X          = make_sequences(df_scaled, features)
    preds_norm = model.predict(X, verbose=0).flatten()
    preds_mgL  = inverse_do(scaler, preds_norm)

    actual_col = df_scaled[TARGET].values[SEQUENCE_LEN:] if TARGET in df_scaled.columns else None
    actual_mgL = inverse_do(scaler, actual_col) if actual_col is not None else None

    result = pd.DataFrame({
        "index"        : range(len(preds_mgL)),
        "predicted_DO" : np.round(preds_mgL, 3),
        "alert"        : preds_mgL < DO_THRESHOLD,
    })

    # Attach meta columns — slice to match prediction rows (offset by SEQUENCE_LEN)
    aligned_meta = pd.DataFrame(meta)

    aligned_meta = aligned_meta.iloc[SEQUENCE_LEN:].reset_index(drop=True)
    aligned_meta = aligned_meta.iloc[:len(result)]

    for col in aligned_meta.columns:
        result[col] = aligned_meta[col].values

    if actual_mgL is not None:
        result["actual_DO"] = np.round(actual_mgL, 3)

    # Attach raw feature values for charts
    raw_slice = raw_values.iloc[SEQUENCE_LEN:].reset_index(drop=True)
    for col in chart_cols:
        if col not in result.columns:
            result[f"raw_{col}"] = raw_slice[col].values

    return result, []


if __name__ == "__main__":
    raw = pd.read_csv(RAW_PATH, low_memory=False).dropna().head(500)
    print("RAW COLUMNS:", list(raw.columns))
    tmp = "data/processed/_sample.csv"
    raw.to_csv(tmp, index=False)
    result, missing = predict(tmp)
    if missing:
        print(f"[error] Missing columns: {missing}")
    else:
        show = ["country","area","waterbody_type","predicted_DO","actual_DO"]
        show = [c for c in show if c in result.columns]
        print(result[show].head(10).to_string(index=False))
        print(f"\n[alerts] {result['alert'].sum()} / {len(result)} below {DO_THRESHOLD} mg/L")
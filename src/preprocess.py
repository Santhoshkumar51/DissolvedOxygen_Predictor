import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

RAW_PATH       = "data/raw/Combined_dataset.csv"
PROCESSED_PATH = "data/processed/cleaned_data.csv"
SCALER_PATH    = "models/scaler.pkl"

COLUMN_MAP = {
    "Temperature (cel)"                  : "temperature",
    "pH (ph units)"                      : "pH",
    "Biochemical Oxygen Demand (mg/l)"   : "BOD",
    "Ammonia (mg/l)"                     : "ammonia",
    "Nitrate (mg/l)"                     : "nitrate",
    "Nitrogen (mg/l)"                    : "nitrogen",
    "Dissolved Oxygen (mg/l)"            : "dissolved_oxygen",
    "Date"                               : "timestamp",
    "Waterbody Type"                     : "waterbody_type",
    "Country"                            : "country",
    "Area"                               : "area",
}

FEATURES  = ["temperature", "pH", "BOD", "ammonia", "nitrate", "nitrogen"]
META_COLS = ["country", "area", "waterbody_type", "timestamp"]
TARGET    = "dissolved_oxygen"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df.rename(columns=COLUMN_MAP)
    keep = FEATURES + [TARGET] + META_COLS
    df   = df[[c for c in keep if c in df.columns]]
    print(f"[load]  {len(df):,} rows  |  columns: {df.columns.tolist()}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=[TARGET])
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())
    for col in FEATURES + [TARGET]:
        df = df[df[col] >= 0]
    if len(df) > 200_000:
        df = df.sample(n=200_000, random_state=42).reset_index(drop=True)
        print(f"[clean] Sampled down to 200,000 rows")
    print(f"[clean] {before - len(df):,} rows removed  →  {len(df):,} remaining")
    return df.reset_index(drop=True)


def normalize(df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    scaler = MinMaxScaler()
    # Only scale numeric feature+target columns — meta cols stay as strings
    if fit:
        df[FEATURES + [TARGET]] = scaler.fit_transform(df[FEATURES + [TARGET]])
        os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        print(f"[norm]  Scaler saved → {SCALER_PATH}")
    else:
        scaler = joblib.load(SCALER_PATH)
        df[FEATURES + [TARGET]] = scaler.transform(df[FEATURES + [TARGET]])
    return df


def split_data(df: pd.DataFrame, train=0.7, val=0.15):
    n = len(df)
    t = int(n * train)
    v = int(n * (train + val))
    train_df, val_df, test_df = df.iloc[:t], df.iloc[t:v], df.iloc[v:]
    print(f"[split] train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")
    return train_df, val_df, test_df


def run():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = load_data(RAW_PATH)
    df = clean_data(df)
    df = normalize(df, fit=True)

    train_df, val_df, test_df = split_data(df)

    train_df.to_csv("data/processed/train.csv",        index=False)
    val_df.to_csv("data/processed/val.csv",            index=False)
    test_df.to_csv("data/processed/test.csv",          index=False)
    df.to_csv(PROCESSED_PATH,                          index=False)

    print(f"\n[done]  All files saved with columns: {df.columns.tolist()}")


if __name__ == "__main__":
    run()
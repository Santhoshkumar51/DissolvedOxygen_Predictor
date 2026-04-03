import numpy as np
import pandas as pd
import joblib
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model import build_model, SEQUENCE_LEN

TRAIN_PATH = "data/processed/train.csv"
VAL_PATH   = "data/processed/val.csv"
MODEL_PATH = "models/bisru_model.keras"
FEATURES   = joblib.load("models/selected_features.pkl")
TARGET     = "dissolved_oxygen"


def make_sequences(df, features, target, seq_len):
    X, y = [], []
    values = df[features].values
    labels = df[target].values
    for i in range(len(df) - seq_len):
        X.append(values[i : i + seq_len])
        y.append(labels[i + seq_len])
    return np.array(X, dtype="float32"), np.array(y, dtype="float32")


def run():
    print(f"[info]  Using features: {FEATURES}")

    train_df = pd.read_csv(TRAIN_PATH)
    val_df   = pd.read_csv(VAL_PATH)

    print("[prep]  Building sequences...")
    X_train, y_train = make_sequences(train_df, FEATURES, TARGET, SEQUENCE_LEN)
    X_val,   y_val   = make_sequences(val_df,   FEATURES, TARGET, SEQUENCE_LEN)
    print(f"        X_train: {X_train.shape}  X_val: {X_val.shape}")

    model = build_model(n_features=len(FEATURES))

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1),
    ]

    print("\n[train] Starting training...\n")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=256,
        callbacks=callbacks,
    )

    print(f"\n[done]  Best model saved → {MODEL_PATH}")


if __name__ == "__main__":
    run()
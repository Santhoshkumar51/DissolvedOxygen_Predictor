import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import joblib
import os

TRAIN_PATH   = "data/processed/train.csv"
MODEL_PATH   = "models/lgbm_selector.pkl"
PLOT_PATH    = "data/processed/feature_importance.png"

FEATURES = ["temperature", "pH", "BOD", "ammonia", "nitrate", "nitrogen"]
TARGET   = "dissolved_oxygen"


def load_train():
    df = pd.read_csv(TRAIN_PATH)
    X  = df[FEATURES]
    y  = df[TARGET]
    print(f"[load]  {len(df):,} training rows")
    return X, y


def train_lgbm(X, y):
    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        verbose=-1
    )
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    print(f"[lgbm]  Model saved → {MODEL_PATH}")
    return model


def get_top_features(model, top_n=4):
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]

    print("\n[importance] Feature ranking:")
    for i, idx in enumerate(indices):
        print(f"  {i+1}. {FEATURES[idx]:<15}  score: {importances[idx]:.1f}")

    top_features = [FEATURES[i] for i in indices[:top_n]]
    print(f"\n[select] Top {top_n} features: {top_features}")

    # Save selected features list
    joblib.dump(top_features, "models/selected_features.pkl")

    return top_features, importances, indices


def plot_importance(importances, indices):
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    sorted_features = [FEATURES[i] for i in indices]
    sorted_scores   = importances[indices]

    plt.figure(figsize=(7, 4))
    colors = ["#1D9E75" if i < 4 else "#B4B2A9" for i in range(len(sorted_features))]
    plt.barh(sorted_features[::-1], sorted_scores[::-1], color=colors[::-1])
    plt.xlabel("Importance score")
    plt.title("Feature importance (LightGBM)")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=120)
    plt.close()
    print(f"[plot]  Saved → {PLOT_PATH}")


def run():
    os.makedirs("models", exist_ok=True)
    X, y              = load_train()
    model             = train_lgbm(X, y)
    top_features, imp, idx = get_top_features(model, top_n=4)
    plot_importance(imp, idx)
    print("\n[done]  Feature selection complete.")
    print(f"        BiSRU model will use: {top_features}")


if __name__ == "__main__":
    run()
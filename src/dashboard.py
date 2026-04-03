import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys

sys.path.append(os.path.dirname(__file__))
from predict import predict, ALL_FEATURES, TARGET, DO_THRESHOLD

st.set_page_config(page_title="DO Predictor", layout="wide", page_icon="💧")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Gradient background */
.stApp {
    background: linear-gradient(135deg, #0b1e33 0%, #0e3d52 40%, #0b2d3a 70%, #07121c 100%);
    background-attachment: fixed;
}
/* Main content area */
.block-container { padding-top: 2rem; }

/* Metric cards */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(100,220,200,0.2);
    border-radius: 12px;
    padding: 16px 20px;
    backdrop-filter: blur(6px);
}
[data-testid="metric-container"] label { color: #7ecfbf !important; font-size: 13px !important; }
[data-testid="metric-container"] [data-testid="metric-value"] { color: #e8f8f5 !important; font-size: 28px !important; }
[data-testid="metric-container"] [data-testid="metric-delta"] { color: #f0a500 !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071828 0%, #0d2e40 100%);
    border-right: 1px solid rgba(100,220,200,0.15);
}
[data-testid="stSidebar"] * { color: #c8eae3 !important; }

/* Headings */
h1 { color: #5de8cc !important; font-size: 2rem !important; letter-spacing: -0.5px; }
h2, h3 { color: #7ecfbf !important; }

/* Divider */
hr { border-color: rgba(100,220,200,0.2) !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid rgba(100,220,200,0.15); border-radius: 10px; }

/* Success / error / info boxes */
.stSuccess { background: rgba(29,158,117,0.15) !important; border-color: #1d9e75 !important; color: #9ff0d0 !important; }
.stError   { background: rgba(226,75,74,0.15)  !important; border-color: #e24b4a !important; color: #ffb3b3 !important; }
.stInfo    { background: rgba(55,138,221,0.12) !important; border-color: #378add !important; color: #b3d8ff !important; }

/* Caption */
.stCaption { color: #7ecfbf !important; opacity: 0.7; }

/* Download button */
.stDownloadButton button {
    background: rgba(29,158,117,0.2) !important;
    border: 1px solid #1d9e75 !important;
    color: #9ff0d0 !important;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


# ── Chart style helper ────────────────────────────────────────────────────────
def chart_style(ax, title=""):
    ax.set_facecolor("#0b1e2d")
    ax.tick_params(colors="#7ecfbf", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor((100/255, 200/255, 180/255, 0.2))
    ax.xaxis.label.set_color("#7ecfbf")
    ax.yaxis.label.set_color("#7ecfbf")
    if title:
        ax.set_title(title, color="#5de8cc", fontsize=11, pad=8)


def make_fig(rows=1, cols=1, h=4, w=12):
    fig, axes = plt.subplots(rows, cols, figsize=(w, h))
    fig.patch.set_facecolor("#0b1e2d")
    return fig, axes


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 💧 Dissolved Oxygen Prediction System")
st.caption("Aquaculture Water Quality Management — BiSRU + Attention Model")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Upload Data")
    uploaded  = st.file_uploader("Upload sensor CSV", type=["csv"])
    threshold = st.slider("Alert threshold (mg/L)", 2.0, 8.0, 5.0, 0.05)
    st.markdown("---")
    with st.expander("Accepted column names"):
        st.caption("Your CSV can use any of these — auto-detected.")
        for field, examples in {
            "Temperature" : "Temperature (cel) · temp",
            "pH"          : "pH (ph units) · ph",
            "BOD"         : "Biochemical Oxygen Demand (mg/l) · bod",
            "Ammonia"     : "Ammonia (mg/l) · nh3",
            "Nitrate"     : "Nitrate (mg/l) · no3",
            "Nitrogen"    : "Nitrogen (mg/l)",
            "DO (target)" : "Dissolved Oxygen (mg/l) · do",
        }.items():
            st.markdown(f"**{field}** — `{examples}`")

if uploaded is None:
    st.info("Upload a CSV file from the sidebar to get started.")
    st.stop()

tmp_path = "data/processed/_upload_tmp.csv"
os.makedirs("data/processed", exist_ok=True)
with open(tmp_path, "wb") as f:
    f.write(uploaded.read())

# Read meta columns directly from the saved file — guaranteed, independent of predict()
_raw_upload = pd.read_csv(tmp_path, low_memory=False)
_META_MAP   = {"country": "country", "area": "area", "waterbody type": "waterbody_type"}
_meta_cols  = {}
for _c in _raw_upload.columns:
    _k = _c.strip().lower()
    if _k in _META_MAP:
        _meta_cols[_META_MAP[_k]] = _raw_upload[_c].reset_index(drop=True)

with st.spinner("Running prediction..."):
    try:
        result, missing = predict(tmp_path)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

# ── Missing columns error ─────────────────────────────────────────────────────
if missing:
    st.error(f"Missing required columns: `{'`, `'.join(missing)}`")
    st.info("Open **Accepted column names** in the sidebar to see supported variations.")
    st.stop()

result["alert"] = result["predicted_DO"] < threshold
alerts   = int(result["alert"].sum())
total    = len(result)
avg_pred = result["predicted_DO"].mean()
min_pred = result["predicted_DO"].min()
max_pred = result["predicted_DO"].max()

# ── Metric cards ──────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total predictions", f"{total:,}")
c2.metric("Avg DO", f"{avg_pred:.2f} mg/L")
c3.metric("Min DO", f"{min_pred:.2f} mg/L")
c4.metric("Max DO", f"{max_pred:.2f} mg/L")
c5.metric("Alerts", f"{alerts:,}", delta=f"{alerts/total*100:.1f}% of readings", delta_color="inverse")

st.divider()

if alerts > 0:
    st.error(f"⚠️  {alerts} readings predicted below {threshold} mg/L — immediate attention required.")
else:
    st.success("✅  All predicted DO levels are within the safe range.")

# ── Country / Area filters ────────────────────────────────────────────────────
has_country = "country" in result.columns
has_area    = "area"    in result.columns

filtered = result.copy()
if has_country or has_area:
    st.markdown("### Filter by location")
    f1, f2 = st.columns(2)
    if has_country:
        countries   = ["All"] + sorted(filtered["country"].dropna().unique().tolist())
        sel_country = f1.selectbox("Country", countries)
        if sel_country != "All":
            filtered = filtered[filtered["country"] == sel_country]
    if has_area:
        areas    = ["All"] + sorted(filtered["area"].dropna().unique().tolist())
        sel_area = f2.selectbox("Area", areas)
        if sel_area != "All":
            filtered = filtered[filtered["area"] == sel_area]

plot_df = filtered.iloc[:600].reset_index(drop=True)

# ── Chart 1: DO Trend ─────────────────────────────────────────────────────────
st.markdown("### Dissolved oxygen trend")
fig, ax = make_fig(h=4)
ax.plot(plot_df.index, plot_df["predicted_DO"], color="#1d9e75", linewidth=1.6,
        label="Predicted DO")
if "actual_DO" in plot_df.columns:
    ax.plot(plot_df.index, plot_df["actual_DO"], color="#378add", linewidth=1.2,
            alpha=0.7, label="Actual DO")
ax.axhline(threshold, color="#e24b4a", linestyle="--", linewidth=1.2,
           label=f"Threshold ({threshold} mg/L)")
ax.fill_between(plot_df.index, 0, threshold, color="#e24b4a", alpha=0.07)
ax.set_xlabel("Reading index")
ax.set_ylabel("DO (mg/L)")
ax.legend(frameon=False, labelcolor="#c8eae3")
chart_style(ax)
plt.tight_layout()
st.pyplot(fig); plt.close()

# ── Chart 2 & 3: DO Distribution + Alert breakdown ───────────────────────────
st.markdown("### Distribution & alert breakdown")
col1, col2 = st.columns(2)

with col1:
    fig, ax = make_fig(h=3.5, w=6)
    vals = filtered["predicted_DO"].values
    ax.hist(vals, bins=40, color="#1d9e75", alpha=0.8, edgecolor="none")
    ax.axvline(threshold, color="#e24b4a", linestyle="--", linewidth=1.2,
               label=f"Threshold {threshold} mg/L")
    ax.axvline(vals.mean(), color="#f0a500", linestyle="-", linewidth=1.2,
               label=f"Mean {vals.mean():.2f} mg/L")
    ax.set_xlabel("Predicted DO (mg/L)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False, labelcolor="#c8eae3", fontsize=9)
    chart_style(ax, "DO distribution")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

with col2:
    fig, ax = make_fig(h=3.5, w=6)
    safe_n  = total - alerts
    sizes   = [safe_n, alerts]
    colors  = ["#1d9e75", "#e24b4a"]
    labels  = [f"Safe\n{safe_n:,}", f"Alert\n{alerts:,}"]
    wedges, texts = ax.pie(sizes, colors=colors, startangle=90,
                           wedgeprops=dict(width=0.55, edgecolor="#0b1e2d", linewidth=2))
    for txt in texts:
        txt.set_color("#c8eae3")
    ax.legend(labels, loc="center", frameon=False, labelcolor="#c8eae3", fontsize=10)
    chart_style(ax, "Safe vs alert readings")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ── Chart 4 & 5: Feature scatter plots ───────────────────────────────────────
raw_features = [c for c in ["raw_temperature","raw_pH","raw_ammonia","raw_nitrate"] if c in filtered.columns]

if raw_features and "actual_DO" in filtered.columns:
    st.markdown("### Feature vs dissolved oxygen")
    cols4 = st.columns(min(4, len(raw_features)))
    labels_map = {
        "raw_temperature": "Temperature (°C)",
        "raw_pH"         : "pH",
        "raw_ammonia"    : "Ammonia (mg/L)",
        "raw_nitrate"    : "Nitrate (mg/L)",
    }
    for i, feat in enumerate(raw_features):
        with cols4[i]:
            fig, ax = make_fig(h=3, w=3)
            ax.scatter(filtered[feat].values[:600],
                       filtered["actual_DO"].values[:600],
                       color="#378add", alpha=0.35, s=8, edgecolors="none")
            ax.set_xlabel(labels_map.get(feat, feat), fontsize=9)
            ax.set_ylabel("DO (mg/L)", fontsize=9)
            chart_style(ax, labels_map.get(feat, feat))
            plt.tight_layout()
            st.pyplot(fig); plt.close()

# ── Chart 6: Actual vs Predicted ─────────────────────────────────────────────
if "actual_DO" in filtered.columns:
    st.markdown("### Predicted vs actual DO")
    fig, ax = make_fig(h=4, w=8)
    sample = filtered.sample(min(600, len(filtered)), random_state=42)
    ax.scatter(sample["actual_DO"], sample["predicted_DO"],
               color="#5de8cc", alpha=0.4, s=12, edgecolors="none")
    mn = min(sample["actual_DO"].min(), sample["predicted_DO"].min())
    mx = max(sample["actual_DO"].max(), sample["predicted_DO"].max())
    ax.plot([mn, mx], [mn, mx], color="#f0a500", linestyle="--",
            linewidth=1.2, label="Perfect prediction")
    ax.set_xlabel("Actual DO (mg/L)")
    ax.set_ylabel("Predicted DO (mg/L)")
    ax.legend(frameon=False, labelcolor="#c8eae3")
    chart_style(ax, "Actual vs predicted DO")
    plt.tight_layout()
    col_l, col_r = st.columns([2, 1])
    col_l.pyplot(fig); plt.close()

    with col_r:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae  = mean_absolute_error(filtered["actual_DO"], filtered["predicted_DO"])
        rmse = np.sqrt(mean_squared_error(filtered["actual_DO"], filtered["predicted_DO"]))
        # r2   = r2_score(filtered["actual_DO"], filtered["predicted_DO"])
        st.markdown("#### Model metrics")
        st.metric("MAE",  f"{mae:.3f} mg/L")
        st.metric("RMSE", f"{rmse:.3f} mg/L")
        # st.metric("R²",   f"{r2:.4f}")

# ── Feature importance ────────────────────────────────────────────────────────
if os.path.exists("data/processed/feature_importance.png"):
    st.markdown("### Feature importance (LightGBM)")
    _, ci, _ = st.columns([1, 2, 1])
    ci.image("data/processed/feature_importance.png")

# ── Predictions table ─────────────────────────────────────────────────────────
st.markdown("### Predictions table")
display_df = filtered.copy()

# st.write(display_df.columns)
# # Inject meta columns directly from the uploaded file (bypasses any pipeline gap)
# for _col, _series in _meta_cols.items():
#     if _col not in display_df.columns:
#         display_df[_col] = _series.reindex(display_df.index).values

display_df["status"] = display_df["alert"].map({True: "⚠️ Alert", False: "✅ Safe"})
display_df = display_df.drop(columns=["alert"] + [c for c in display_df.columns if c.startswith("raw_")], errors="ignore")

priority = ["country", "area", "waterbody_type", "index", "predicted_DO", "actual_DO", "status"]
ordered  = [c for c in priority if c in display_df.columns]
rest     = [c for c in display_df.columns if c not in ordered]
display_df = display_df[ordered + rest]

st.dataframe(display_df, use_container_width=True, height=320)

csv_bytes = filtered.to_csv(index=False).encode()
st.download_button("⬇ Download predictions CSV", csv_bytes,
                   file_name="do_predictions.csv", mime="text/csv")
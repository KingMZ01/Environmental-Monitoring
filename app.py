# app.py - Multi-page AQI Prediction & Model Performance Streamlit App
import os
import glob
import joblib
import json
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------------
# Utility: find a file by patterns (returns first match or None)
# -------------------------
def find_file(patterns):
    """Search working dir and subfolders for first matching file pattern(s)."""
    for p in patterns:
        matches = glob.glob(p, recursive=True)
        if matches:
            matches.sort(key=os.path.getmtime, reverse=True)
            return matches[0]
    return None

# -------------------------
# Load model + feature list robustly
# -------------------------
@st.cache_resource
def load_model_and_features():
    # possible filenames to search
    model_candidates = [
        "xgb_pm25_1h.joblib",
        "Models/xgb_pm25_1h.joblib",
        "xgb*.joblib",
        "models/*.joblib",
        "*.joblib"
    ]
    feat_candidates = [
        "feature_list.csv",
        "Data/feature_list.csv",
        "feature_list*.csv",
        "*.csv"
    ]

    model_path = find_file(model_candidates)
    feat_path = find_file(feat_candidates)

    if model_path is None or feat_path is None:
        # don't fail — return Nones; UI will ask user to upload
        return None, None, None, None

    # load model
    model = joblib.load(model_path)

    # load features robustly: CSV or JSON
    try:
        raw = pd.read_csv(feat_path, header=None)[0].astype(str).str.strip().tolist()
    except Exception:
        # try JSON
        try:
            with open(feat_path, "r") as f:
                raw = json.load(f)
        except Exception:
            raw = []

    # clean artifacts often saved by mistake
    feature_cols = [c for c in raw if c and c not in ["0", "pm2.5", "pm25", ""]]

    return model, feature_cols, model_path, feat_path

model, feature_cols, model_path, feat_path = load_model_and_features()

# -------------------------
# AQI conversion (EPA breakpoints)
# -------------------------
breakpoints = [
    (0.0, 12.0, 0, 50, "Good", "#4CAF50"),
    (12.1, 35.4, 51, 100, "Moderate", "#FFEB3B"),
    (35.5, 55.4, 101, 150, "Unhealthy for Sensitive Groups", "#FF9800"),
    (55.5, 150.4, 151, 200, "Unhealthy", "#F44336"),
    (150.5, 250.4, 201, 300, "Very Unhealthy", "#9C27B0"),
    (250.5, 500.4, 301, 500, "Hazardous", "#7E0023"),
]

def aqi_from_pm25(pm25):
    pm = float(pm25)
    for (c_low, c_high, aqi_low, aqi_high, cat, color) in breakpoints:
        if c_low <= pm <= c_high:
            aqi = ((aqi_high - aqi_low) / (c_high - c_low)) * (pm - c_low) + aqi_low
            return int(round(aqi)), cat, color
    return None, "Unknown", "#9E9E9E"

# -------------------------
# Helpers: predict with XGBoost / wrapper
# -------------------------
def xgb_predict(model_obj, X_df):
    """
    Accepts a joblib-loaded XGBoost model (Booster or sklearn XGBRegressor).
    Returns a 1d numpy array of predictions.
    """
    # If model is Booster
    if isinstance(model_obj, xgb.Booster) or model_obj.__class__.__name__ == "Booster":
        dmat = xgb.DMatrix(X_df)
        return model_obj.predict(dmat)
    # If sklearn wrapper (XGBRegressor)
    if hasattr(model_obj, "predict"):
        return np.asarray(model_obj.predict(X_df))
    raise TypeError("Unsupported model object type: %s" % type(model_obj))

def try_feature_align(df, feature_cols):
    """
    Ensure df contains all columns in feature_cols.
    If missing, add with zeros and warn.
    Returns aligned df in the requested column order.
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        st.warning(f"Missing features in input (added with zeros): {missing}")
        for c in missing:
            df[c] = 0
    return df[feature_cols]

# -------------------------
# UI: Page layout and nav
# -------------------------
st.set_page_config(page_title="AQI Dashboard", layout="wide", page_icon="🌍")

st.sidebar.title("AQI Dashboard")
page = st.sidebar.radio("Go to", ["AQI Prediction", "Model Performance", "About / Data"])

# Allow manual upload of model & feature_list if auto-load failed
if model is None or not feature_cols:
    with st.sidebar.expander("Model / Features not auto-found — upload here"):
        uploaded_model = st.file_uploader("Upload joblib model (.joblib)", type=["joblib", "pkl"])
        uploaded_feat = st.file_uploader("Upload feature list (.csv or .json)", type=["csv", "json"])
        if uploaded_model is not None:
            # save to temp and load
            tmp_model_path = os.path.join(".", uploaded_model.name)
            with open(tmp_model_path, "wb") as f:
                f.write(uploaded_model.getbuffer())
            try:
                model = joblib.load(tmp_model_path)
                st.success("Model loaded")
            except Exception as e:
                st.error(f"Failed loading model: {e}")
        if uploaded_feat is not None:
            tmp_feat_path = os.path.join(".", uploaded_feat.name)
            with open(tmp_feat_path, "wb") as f:
                f.write(uploaded_feat.getbuffer())
            try:
                if uploaded_feat.name.lower().endswith(".json"):
                    with open(tmp_feat_path) as f:
                        raw = json.load(f)
                else:
                    raw = pd.read_csv(tmp_feat_path, header=None)[0].astype(str).str.strip().tolist()
                feature_cols = [c for c in raw if c and c not in ["0", "pm2.5"]]
                st.success("Feature list loaded")
            except Exception as e:
                st.error(f"Failed reading feature list: {e}")

# -------------------------
# Page: About / Data
# -------------------------
if page == "About / Data":
    st.title("About this App")
    st.markdown("""
    **Air Quality Prediction** — predicts next-hour **PM2.5** and converts to **AQI** (EPA breakpoints).
    
    App pages:
    - **AQI Prediction**: manual or CSV batch predictions (visual results + gauge).
    - **Model Performance**: view evaluation metrics, feature importance, and diagnostic plots.
    """)
    st.markdown("### Repo / Files detected")
    st.write("Model path:", model_path)
    st.write("Feature list path:", feat_path)
    st.markdown("### Notes & Instructions")
    st.markdown("""
    - If you trained the model locally, save `xgb_pm25_1h.joblib` and `feature_list.csv` into the app folder (or models/).
    - For production, deploy the same model artifacts and keep `feature_list.csv` consistent (same order).
    - The app expects the features the model was trained on. If your input lacks lag/rolling features, supply them or the app will fill zeros (less accurate).
    """)

# -------------------------
# Page: AQI Prediction
# -------------------------
elif page == "AQI Prediction":
    st.title("🔮 AQI Prediction")

    with st.expander("ℹ️ What are PM2.5 and AQI?"):
        st.markdown("""
    - **PM2.5**: Fine particulate matter smaller than 2.5 microns.  
      It can penetrate deep into the lungs and bloodstream, causing health issues.  

    - **AQI (Air Quality Index)**: A standardized scale (0–500) that converts pollutant levels into categories like *Good*, *Moderate*, *Unhealthy*, etc.  
    """)

    st.markdown("Choose input mode: `Manual Input` (quick) or `Upload CSV` (batch).")

    mode = st.radio("Input mode:", ["Manual Input", "Upload CSV"], horizontal=True)

    if mode == "Manual Input":
        st.subheader("Manual Input")
        # Provide a minimal (friendly) set of inputs; we will fill or zero other features
        c1, c2, c3 = st.columns(3)
        with c1:
            DEWP = st.number_input("Dew Point (°C)", value=5.0, step=0.1)
            TEMP = st.number_input("Temperature (°C)", value=20.0, step=0.1)
        with c2:
            PRES = st.number_input("Pressure (hPa)", value=1010.0, step=0.1)
            Iws = st.number_input("Wind Speed (m/s)", value=2.0, step=0.1)
        with c3:
            Is = st.number_input("Cumulated Snow Hours (Is)", value=0, step=1)
            Ir = st.number_input("Cumulated Rain Hours (Ir)", value=0, step=1)

        now = datetime.now()
        hour = st.slider("Hour (0-23)", 0, 23, now.hour)
        dow = st.selectbox("Day of week (0=Mon)", list(range(7)), index=now.weekday())
        month = st.selectbox("Month (1-12)", list(range(1,13)), index=now.month - 1)

        # Build sample DataFrame with provided & fallback values
        sample = pd.DataFrame([{
            "DEWP": DEWP, "TEMP": TEMP, "PRES": PRES, "Iws": Iws, "Is": Is, "Ir": Ir,
            "hour": hour, "dow": dow, "month": month,
            "hour_sin": np.sin(2*np.pi*hour/24), "hour_cos": np.cos(2*np.pi*hour/24),
        }])

        # fill common lag/roll features with zeros (user should provide real ones if available)
        if feature_cols:
            for c in feature_cols:
                if c not in sample.columns:
                    sample[c] = 0.0
            sample = sample[feature_cols]  # align order

        if st.button("Predict"):
            if model is None or not feature_cols:
                st.error("Model or feature list not available. Please upload artifacts via sidebar.")
            else:
                try:
                    preds = xgb_predict(model, sample)
                    pred_pm = float(preds[0])
                    pred_aqi, pred_cat, color = aqi_from_pm25(pred_pm)

                    # Colored result card
                    st.markdown(
                        f"""
                        <div style="padding:18px;border-radius:12px;background-color:{color};color:white;text-align:center;">
                            <h2> AQI: {pred_aqi} — {pred_cat} </h2>
                            <h4>Predicted PM2.5: {pred_pm:.2f} µg/m³</h4>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Plotly gauge
                    gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=pred_aqi,
                        title={'text': "Air Quality Index (AQI)"},
                        gauge={
                            'axis': {'range': [0, 500]},
                            'bar': {'color': color},
                            'steps': [
                                {'range': [0,50], 'color': "#4CAF50"},
                                {'range': [51,100], 'color': "#FFEB3B"},
                                {'range': [101,150], 'color': "#FF9800"},
                                {'range': [151,200], 'color': "#F44336"},
                                {'range': [201,300], 'color': "#9C27B0"},
                                {'range': [301,500], 'color': "#7E0023"}
                            ],
                        }
                    ))
                    st.plotly_chart(gauge, use_container_width=True)

                    # PM2.5 vs AQI curve
                    st.markdown("#### PM2.5 → AQI mapping (EPA)")
                    pm_range = np.linspace(0, 300, 300)
                    aqi_vals = [aqi_from_pm25(pm)[0] for pm in pm_range]
                    fig, ax = plt.subplots(figsize=(8,4))
                    ax.plot(pm_range, aqi_vals, color="tab:blue", lw=2, label="AQI Mapping")
                    ax.scatter([pred_pm], [pred_aqi], color="red", s=80, label="Prediction")
                    ax.set_xlabel("PM2.5 (µg/m³)")
                    ax.set_ylabel("AQI")
                    ax.set_title("PM2.5 vs AQI (EPA)")
                    ax.legend()
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    else:
        st.subheader("Batch Input (CSV)")
        st.markdown("Upload a CSV that already contains the features the model expects (same columns & order as feature_list).")
        uploaded = st.file_uploader("CSV file", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.write("Preview:", df.head())
            if st.button("Run batch prediction"):
                if model is None or not feature_cols:
                    st.error("Model or feature list missing. Upload via sidebar.")
                else:
                    try:
                        df_aligned = try_feature_align(df.copy(), feature_cols)
                        preds = xgb_predict(model, df_aligned)
                        df_result = df_aligned.copy()
                        df_result["pred_pm2.5"] = np.round(preds, 2)
                        # add AQI & category
                        aqi_list, cat_list, _ = [], [], []
                        for pm in df_result["pred_pm2.5"]:
                            a, c, col = aqi_from_pm25(pm)
                            aqi_list.append(a); cat_list.append(c)
                        df_result["pred_AQI"] = aqi_list
                        df_result["AQI_category"] = cat_list

                        st.success("Batch predictions done.")
                        st.dataframe(df_result.head(200))

                        csv = df_result.to_csv(index=False).encode("utf-8")
                        st.download_button("Download predictions CSV", csv, "aqi_predictions.csv", "text/csv")
                    except Exception as e:
                        st.error(f"Batch prediction failed: {e}")

# -------------------------
# Page: Model Performance
# -------------------------
elif page == "Model Performance":
    st.title("📈 Model Performance")

    st.markdown("Provide a CSV with `true` and `pred` columns or let the app try to load evaluation artifacts.")

    # Try to auto-find evaluation CSVs
    eval_candidate = find_file(["eval_results.csv", "y_true_y_pred.csv", "y_test_pred.csv", "*eval*.csv"])
    uploaded_eval = st.file_uploader("Upload evaluation CSV (columns: true,pred,index(optional))", type=["csv"])

    df_eval = None
    if uploaded_eval is not None:
        df_eval = pd.read_csv(uploaded_eval)
    elif eval_candidate is not None:
        try:
            df_eval = pd.read_csv(eval_candidate)
            st.info(f"Loaded evaluation file: {eval_candidate}")
        except Exception:
            df_eval = None

    if df_eval is None:
        st.warning("No evaluation data loaded. You can upload a CSV with ground-truth and predictions to view metrics/plots.")
    else:
        # Expect columns: true, pred (if different names try to guess)
        cols = [c.lower() for c in df_eval.columns]
        true_col = None; pred_col = None
        for name in ["true","y_true","actual","target","y"]:
            if name in cols:
                true_col = df_eval.columns[cols.index(name)]; break
        for name in ["pred","y_pred","yhat","prediction","yhat"]:
            if name in cols:
                pred_col = df_eval.columns[cols.index(name)]; break
        # fallback to first two numeric columns
        if true_col is None or pred_col is None:
            numeric_cols = df_eval.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                true_col, pred_col = numeric_cols[0], numeric_cols[1]
            else:
                st.error("Couldn't find numeric true/pred columns. Please upload a CSV with 'true' and 'pred' columns.")
                st.stop()

        y_true = df_eval[true_col].values
        y_pred = df_eval[pred_col].values

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (np.where(y_true==0, 1e-8, y_true)))) * 100
        r2 = r2_score(y_true, y_pred)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("RMSE", f"{rmse:.3f}")
        col2.metric("MAE", f"{mae:.3f}")
        col3.metric("MAPE", f"{mape:.2f}%")
        col4.metric("R²", f"{r2:.3f}")

        st.markdown("### True vs Predicted (scatter)")
        fig, ax = plt.subplots(figsize=(6,6))
        ax.scatter(y_true, y_pred, alpha=0.4, s=20)
        mn = min(min(y_true), min(y_pred)); mx = max(max(y_true), max(y_pred))
        ax.plot([mn,mx],[mn,mx], 'r--', lw=1)
        ax.set_xlabel("True PM2.5")
        ax.set_ylabel("Predicted PM2.5")
        st.pyplot(fig)

        st.markdown("### Residuals distribution")
        resid = y_true - y_pred
        fig, ax = plt.subplots(figsize=(8,4))
        ax.hist(resid, bins=50, color="tab:gray")
        ax.set_xlabel("Residual (true - pred)")
        st.pyplot(fig)

        st.markdown("### Time-series sample (first 200 rows)")
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(y_true[:200], label="True")
        ax.plot(y_pred[:200], label="Predicted")
        ax.legend()
        st.pyplot(fig)

        # Feature importance from model if available
        st.markdown("### Feature importance (model)")
        if model is None:
            st.info("Model not loaded — upload it via the sidebar to view feature importances.")
        else:
            try:
                # Try sklearn feature_importances_
                if hasattr(model, "feature_importances_"):
                    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
                else:
                    # Booster: get_score returns {'f0':value,...}
                    booster = model if isinstance(model, xgb.Booster) else model.get_booster()
                    raw = booster.get_score(importance_type="gain")
                    # map f0->feature name
                    mapped = {}
                    for k,v in raw.items():
                        if k.startswith("f") and k[1:].isdigit():
                            idx = int(k[1:]); name = feature_cols[idx] if idx < len(feature_cols) else k
                            mapped[name] = v
                        else:
                            mapped[k] = v
                    imp = pd.Series(mapped).sort_values(ascending=False)
                topn = imp.head(20)
                fig, ax = plt.subplots(figsize=(8,6))
                topn.plot(kind="barh", ax=ax)
                ax.invert_yaxis()
                ax.set_xlabel("Importance")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Could not compute feature importance: {e}")

# End of app

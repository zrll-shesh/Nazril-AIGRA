# import os
# import json
# import math
# import textwrap
# import datetime
# import warnings

# import numpy as np
# import pandas as pd
# import streamlit as st
# import joblib
# import matplotlib.pyplot as plt

# from sklearn.metrics import classification_report, confusion_matrix

# # Silence noisy version-mismatch warnings from loading pickled models trained
# # on a slightly different scikit-learn/xgboost version. This does not change
# # prediction results; it only hides the console warning. If predictions ever
# # look wrong after a library upgrade, re-export the model with the *current*
# # library versions instead of relying on this suppression.
# warnings.filterwarnings("ignore", message=".*Trying to unpickle estimator.*")
# warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# st.set_page_config(
#     page_title="Stasiun Prediksi Cuaca | Smart Farming",
#     page_icon=":seedling:",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ----------------------------------------------------------------------------
# # Design tokens - dark green & white
# # ----------------------------------------------------------------------------
# # Two-tone brief: dark green + white carries the whole UI. The one deliberate
# # exception is a muted brick-red used ONLY where a farmer needs a signal to
# # pop above everything else - the rain-warning badge, the gauge's danger
# # zone, and the threshold marker. Everything else, including "no rain",
# # "info", and "success" states, stays inside the green/white system so nothing
# # competes with that one real alert color.
# BG_MAIN = "#FFFFFF"
# BG_PANEL = "#F5F9F6"
# BG_PANEL_ALT = "#E9F2EC"
# BG_HERO_1 = "#0F3D26"
# BG_HERO_2 = "#155C38"
# BORDER = "#D7E6DC"
# BORDER_MPL = "#C4D9CB"

# TEXT_PRIMARY = "#0F2E1F"
# TEXT_MUTED = "#5B7768"
# TEXT_ON_DARK = "#F3FAF5"
# TEXT_ON_DARK_MUTED = "#AFD1BE"

# GREEN_DEEP = "#0F3D26"
# GREEN_MID = "#1F7A4C"
# GREEN_LIGHT = "#4CAF71"
# GREEN_PALE = "#DCEEE1"

# ALERT_RUST = "#B3392A"
# ALERT_RUST_PALE = "#F4E2DE"

# CANDIDATE_DIRS = ["outputs/models", "models", "."]
# FIG_CANDIDATE_DIRS = ["outputs/figures", "figures"]
# DATA_CANDIDATE_DIRS = ["outputs/data", "data"]
# LOGO_CANDIDATES = [
#     "aigra_logo.png", "aigra_logo.jpg", "aigra logo.jpg", "aigra logo.png",
#     "logo.png", "logo.jpg",
#     "assets/aigra_logo.png", "assets/logo.png",
#     "outputs/aigra_logo.png",
# ]


# def find_dir(candidates, filename):
#     for d in candidates:
#         if os.path.exists(os.path.join(d, filename)):
#             return d
#     return None


# def find_logo():
#     for path in LOGO_CANDIDATES:
#         if os.path.exists(path):
#             return path
#     return None


# MODEL_DIR = find_dir(CANDIDATE_DIRS, "best_model.joblib")
# FIG_DIR = find_dir(FIG_CANDIDATE_DIRS, "13_feature_importance.png")
# DATA_DIR = find_dir(DATA_CANDIDATE_DIRS, "model_comparison.csv")
# LOGO_PATH = find_logo()


# # ----------------------------------------------------------------------------
# # Small hand-built line-icon set (single-color, stroke-based) so panel titles
# # get a visual anchor without breaking the two-tone-plus-rust palette or
# # pulling in a colourful emoji/icon-font dependency.
# # ----------------------------------------------------------------------------
# def icon(name, color=None, size=16):
#     color = color or GREEN_MID
#     paths = {
#         "thermometer": '<path d="M9 2.5a1.5 1.5 0 0 1 3 0v7.6a3.5 3.5 0 1 1-3 0V2.5Z"/><circle cx="10.5" cy="14" r="1.4" fill="{c}" stroke="none"/>',
#         "droplet": '<path d="M10 2.5S4.5 9 4.5 13a5.5 5.5 0 0 0 11 0C15.5 9 10 2.5 10 2.5Z"/>',
#         "wind": '<path d="M2.5 7h9a2 2 0 1 0-2-2"/><path d="M2.5 11h12a2 2 0 1 1-2 2"/><path d="M2.5 15h7a2 2 0 1 0-2 2"/>',
#         "rain": '<path d="M5 8.5a4 4 0 0 1 .3-7.9A5 5 0 0 1 15 2a3.7 3.7 0 0 1 .5 7.4"/><path d="M6 13l-1.3 2.6M10 13l-1.3 2.6M14 13l-1.3 2.6"/>',
#         "chart": '<path d="M3 17V9M8 17V4M13 17v-6M18 17H3"/>',
#         "gauge": '<path d="M2.5 15a7.5 7.5 0 1 1 15 0"/><path d="M10 15 13.2 8.8"/><circle cx="10" cy="15" r="1.1" fill="{c}" stroke="none"/>',
#         "cpu": '<rect x="6" y="6" width="8" height="8" rx="1"/><path d="M8.5 2.5v2M11.5 2.5v2M8.5 15.5v2M11.5 15.5v2M2.5 8.5h2M2.5 11.5h2M15.5 8.5h2M15.5 11.5h2"/>',
#         "bot": '<rect x="3" y="6.5" width="14" height="9.5" rx="2.5"/><circle cx="7.5" cy="11" r="1" fill="{c}" stroke="none"/><circle cx="12.5" cy="11" r="1" fill="{c}" stroke="none"/><path d="M10 6.5V3.5"/><circle cx="10" cy="2.5" r="1" fill="{c}" stroke="none"/>',
#         "layers": '<path d="M10 2.5 18 7 10 11.5 2 7Z"/><path d="M2 11 10 15.5 18 11"/>',
#         "info": '<circle cx="10" cy="10" r="7.5"/><path d="M10 9v5"/><circle cx="10" cy="6.3" r="0.9" fill="{c}" stroke="none"/>',
#         "leaf": '<path d="M4 16C4 8 9 3 17 3c0 8-5 13-13 13Z"/><path d="M4.5 15.5 15 5"/>',
#         "grid": '<path d="M3 4h14v12H3zM3 10h14M9 4v12"/>',
#         "shield": '<path d="M10 2.5 16.5 5v5.2c0 4-2.7 6.6-6.5 7.8-3.8-1.2-6.5-3.8-6.5-7.8V5Z"/>',
#     }
#     body = paths.get(name, paths["info"]).format(c=color)
#     return (
#         f'<svg width="{size}" height="{size}" viewBox="0 0 20 20" fill="none" '
#         f'stroke="{color}" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" '
#         f'style="vertical-align:-3px; margin-right:0.4rem;">{body}</svg>'
#     )


# def panel_title(label, icon_name=None, right_text=None, dark=False):
#     ic = icon(icon_name, color=(TEXT_ON_DARK if dark else GREEN_MID)) if icon_name else ""
#     right = f'<span>{right_text}</span>' if right_text else ""
#     cls = "panel-title dark" if dark else "panel-title"
#     return f'<div class="{cls}"><span>{ic}{label}</span>{right}</div>'


# def render_fallback_logo(size=40, on_dark=True):
#     """Small leaf-in-circle mark used when no logo file is found on disk."""
#     ring = TEXT_ON_DARK if on_dark else GREEN_DEEP
#     fill = "none"
#     svg = (
#         f'<svg width="{size}" height="{size}" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">'
#         f'<circle cx="20" cy="20" r="18.5" fill="{fill}" stroke="{ring}" stroke-width="1.5"/>'
#         f'<path d="M12 26 C12 16 20 11 29 11 C29 20 24 28 14 28 C13 28 12.3 27.4 12 26 Z" fill="{GREEN_LIGHT if on_dark else GREEN_MID}"/>'
#         f'<path d="M13 27 L26 12" stroke="{ring}" stroke-width="1.1" fill="none"/>'
#         f'</svg>'
#     )
#     return svg


# # ----------------------------------------------------------------------------
# # Global CSS
# # ----------------------------------------------------------------------------
# st.markdown(
#     f"""
#     <link rel="preconnect" href="https://fonts.googleapis.com">
#     <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
#     <style>
#         html, body, [class*="css"] {{
#             font-family: 'Inter', sans-serif;
#             color: {TEXT_PRIMARY};
#         }}
#         .stApp {{ background: {BG_MAIN}; }}

#         /* Tighten default Streamlit chrome so content starts close to the top
#            instead of leaving a large dead gap under the toolbar. */
#         .block-container {{
#             padding-top: 1.6rem;
#             padding-bottom: 3rem;
#             max-width: 1180px;
#         }}
#         header[data-testid="stHeader"] {{
#             background: rgba(255,255,255,0.72);
#             backdrop-filter: blur(6px);
#             border-bottom: 1px solid {BORDER};
#         }}
#         [data-testid="stSidebarContent"] {{ padding-top: 1rem; }}
#         section[data-testid="stSidebar"] {{
#             background: {BG_PANEL};
#             border-right: 1px solid {BORDER};
#         }}
#         section[data-testid="stSidebar"] .block-container {{ padding-top: 0.5rem; }}

#         h1, h2, h3, h4 {{
#             font-family: 'Space Grotesk', sans-serif !important;
#             letter-spacing: -0.01em;
#             color: {GREEN_DEEP};
#         }}
#         .eyebrow {{
#             font-family: 'IBM Plex Mono', monospace;
#             font-size: 0.72rem;
#             letter-spacing: 0.18em;
#             text-transform: uppercase;
#             color: {GREEN_MID};
#         }}
#         .eyebrow.on-dark {{ color: {TEXT_ON_DARK_MUTED}; }}

#         /* ---------------- Hero ---------------- */
#         .hero {{
#             position: relative;
#             overflow: hidden;
#             background: linear-gradient(135deg, {BG_HERO_1} 0%, {BG_HERO_2} 100%);
#             border-radius: 20px;
#             padding: 1.9rem 2.2rem;
#             margin-bottom: 1.3rem;
#             display: flex;
#             justify-content: space-between;
#             align-items: center;
#             gap: 1.5rem;
#             flex-wrap: wrap;
#         }}
#         .hero::before {{
#             content: "";
#             position: absolute;
#             inset: 0;
#             background-image: repeating-linear-gradient(
#                 115deg,
#                 rgba(255,255,255,0.05) 0px,
#                 rgba(255,255,255,0.05) 1.5px,
#                 transparent 1.5px,
#                 transparent 34px
#             );
#             pointer-events: none;
#         }}
#         .hero-brand {{
#             display: flex;
#             align-items: flex-start;
#             gap: 0.9rem;
#             position: relative;
#             z-index: 1;
#             min-width: 320px;
#         }}
#         .hero-logo-box {{
#             flex-shrink: 0;
#             width: 48px; height: 48px;
#             display:flex; align-items:center; justify-content:center;
#         }}
#         .hero-logo-box img {{
#             width: 46px; height: 46px; object-fit: contain;
#             border-radius: 10px;
#             background: {TEXT_ON_DARK};
#             padding: 4px;
#         }}
#         .hero-title {{
#             font-size: 1.85rem;
#             font-weight: 700;
#             margin: 0.25rem 0 0 0;
#             line-height: 1.15;
#             color: {TEXT_ON_DARK};
#         }}
#         .hero-sub {{
#             color: {TEXT_ON_DARK_MUTED};
#             font-size: 0.94rem;
#             margin-top: 0.35rem;
#             max-width: 480px;
#             line-height: 1.5;
#         }}
#         .hero-chips {{
#             display: flex;
#             gap: 0.7rem;
#             position: relative;
#             z-index: 1;
#             flex-wrap: wrap;
#         }}
#         .hero-chip {{
#             background: rgba(255,255,255,0.08);
#             border: 1px solid rgba(255,255,255,0.16);
#             border-radius: 12px;
#             padding: 0.6rem 1rem;
#             min-width: 110px;
#         }}
#         .hero-chip .metric-label {{ color: {TEXT_ON_DARK_MUTED}; }}
#         .hero-chip .metric-value {{ color: {TEXT_ON_DARK}; font-size: 1.05rem; }}

#         .hairline {{
#             border: none;
#             border-top: 1px solid {BORDER};
#             margin: 1.1rem 0 1.4rem 0;
#         }}
#         .panel {{
#             background: {BG_PANEL};
#             border: 1px solid {BORDER};
#             border-radius: 14px;
#             padding: 1.15rem 1.3rem;
#             margin-bottom: 1rem;
#             box-shadow: 0 1px 2px rgba(15,61,38,0.04);
#         }}
#         .panel-title {{
#             font-family: 'IBM Plex Mono', monospace;
#             font-size: 0.74rem;
#             letter-spacing: 0.1em;
#             text-transform: uppercase;
#             color: {TEXT_MUTED};
#             margin-bottom: 0.8rem;
#             display: flex;
#             justify-content: space-between;
#             align-items: center;
#         }}
#         .panel-title.dark {{ color: {TEXT_ON_DARK_MUTED}; }}
#         .metric-row {{ display: flex; gap: 0.7rem; flex-wrap: wrap; }}
#         .metric-card {{
#             background: {BG_PANEL_ALT};
#             border: 1px solid {BORDER};
#             border-radius: 10px;
#             padding: 0.7rem 0.9rem;
#             flex: 1;
#             min-width: 120px;
#         }}
#         .metric-card.highlight {{ background: {GREEN_DEEP}; border-color: {GREEN_DEEP}; }}
#         .metric-card.highlight .metric-label {{ color: {GREEN_PALE}; }}
#         .metric-card.highlight .metric-value {{ color: #FFFFFF; }}
#         .metric-label {{
#             font-family: 'IBM Plex Mono', monospace;
#             font-size: 0.65rem;
#             letter-spacing: 0.1em;
#             text-transform: uppercase;
#             color: {TEXT_MUTED};
#         }}
#         .metric-value {{
#             font-family: 'Space Grotesk', sans-serif;
#             font-size: 1.35rem;
#             font-weight: 600;
#             margin-top: 0.15rem;
#             color: {GREEN_DEEP};
#         }}
#         .badge-yes, .badge-no {{
#             display: inline-block;
#             font-family: 'IBM Plex Mono', monospace;
#             font-size: 0.78rem;
#             letter-spacing: 0.08em;
#             text-transform: uppercase;
#             padding: 0.35rem 0.85rem;
#             border-radius: 999px;
#             font-weight: 600;
#         }}
#         .badge-yes {{
#             background: {ALERT_RUST_PALE};
#             color: {ALERT_RUST};
#             border: 1px solid rgba(179, 57, 42, 0.35);
#         }}
#         .badge-no {{
#             background: {GREEN_PALE};
#             color: {GREEN_MID};
#             border: 1px solid rgba(31, 122, 76, 0.4);
#         }}
#         .station-id {{
#             font-family: 'IBM Plex Mono', monospace;
#             font-size: 0.74rem;
#             color: {TEXT_MUTED};
#             letter-spacing: 0.03em;
#         }}
#         .stButton > button {{
#             background: {GREEN_DEEP};
#             color: white;
#             border: none;
#             border-radius: 10px;
#             padding: 0.6rem 1.1rem;
#             font-weight: 600;
#             font-family: 'Space Grotesk', sans-serif;
#             width: 100%;
#             transition: filter 0.12s ease, transform 0.12s ease;
#         }}
#         .stButton > button:hover {{ filter: brightness(1.18); transform: translateY(-1px); }}
#         .stDownloadButton > button {{
#             background: {BG_PANEL_ALT};
#             color: {GREEN_DEEP};
#             border: 1px solid {BORDER};
#             border-radius: 10px;
#             font-family: 'IBM Plex Mono', monospace;
#             font-size: 0.8rem;
#         }}
#         .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {BORDER}; }}
#         .stTabs [data-baseweb="tab"] {{
#             font-family: 'IBM Plex Mono', monospace;
#             font-size: 0.8rem;
#             letter-spacing: 0.05em;
#             text-transform: uppercase;
#             color: {TEXT_MUTED};
#             padding: 0.6rem 1rem;
#         }}
#         .stTabs [aria-selected="true"] {{
#             color: {GREEN_DEEP} !important;
#             border-bottom: 2px solid {GREEN_DEEP} !important;
#         }}
#         .footnote {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: {TEXT_MUTED}; }}
#         div[data-testid="stMetricValue"] {{ font-family: 'Space Grotesk', sans-serif; color: {GREEN_DEEP}; }}

#         .rec-box {{
#             border-left: 3px solid {GREEN_MID};
#             background: {BG_PANEL_ALT};
#             padding: 0.85rem 1rem;
#             border-radius: 0 8px 8px 0;
#             font-size: 0.92rem;
#             line-height: 1.5;
#             color: {TEXT_PRIMARY};
#         }}
#         .rec-box.alert {{ border-left-color: {ALERT_RUST}; background: {ALERT_RUST_PALE}; }}

#         /* Re-skin Streamlit's built-in info/warning/error/success boxes so
#            they match the green/white/rust system instead of the default
#            yellow/blue palette. */
#         div[data-testid="stAlert"] {{
#             background: {BG_PANEL_ALT} !important;
#             border: 1px solid {BORDER} !important;
#             border-left: 4px solid {GREEN_MID} !important;
#             border-radius: 0 10px 10px 0 !important;
#         }}
#         div[data-testid="stAlert"] p, div[data-testid="stAlert"] li {{ color: {TEXT_PRIMARY} !important; }}
#         div[data-testid="stAlert"] svg {{ fill: {GREEN_MID} !important; }}

#         [data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 10px; }}
#         [data-testid="stChatMessage"] {{
#             background: {BG_PANEL_ALT}; border: 1px solid {BORDER}; border-radius: 12px;
#         }}
#         .stSlider [data-baseweb="slider"] div[role="slider"] {{ background-color: {GREEN_DEEP} !important; }}
#         .stSlider [data-baseweb="slider"] > div > div {{ background: {GREEN_MID} !important; }}
#         div[data-baseweb="select"] > div {{ border-color: {BORDER} !important; }}
#         input, .stNumberInput input {{ caret-color: {GREEN_DEEP}; }}

#         ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
#         ::-webkit-scrollbar-track {{ background: {BG_PANEL}; }}
#         ::-webkit-scrollbar-thumb {{ background: {BORDER_MPL}; border-radius: 10px; }}

#         .gauge-idle {{
#             width: 220px; height: 136px; margin: 0 auto;
#             border: 1.5px dashed {BORDER_MPL};
#             border-radius: 999px 999px 0 0;
#             display: flex; align-items: flex-end; justify-content: center;
#             color: {TEXT_MUTED};
#         }}
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# if MODEL_DIR is None:
#     st.error(
#         "File model tidak ditemukan. Ekstrak `rain_prediction_outputs.zip` dari notebook, "
#         "lalu letakkan folder `outputs/models` satu direktori dengan streamlit_app.py."
#     )
#     st.stop()


# @st.cache_resource
# def load_artifacts(model_dir):
#     model = joblib.load(os.path.join(model_dir, "best_model.joblib"))
#     scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
#     location_freq_map = joblib.load(os.path.join(model_dir, "location_freq_map.joblib"))
#     feature_columns = joblib.load(os.path.join(model_dir, "feature_columns.joblib"))
#     compass_to_deg = joblib.load(os.path.join(model_dir, "compass_to_deg.joblib"))
#     with open(os.path.join(model_dir, "model_info.json")) as f:
#         model_info = json.load(f)
#     return model, scaler, location_freq_map, feature_columns, compass_to_deg, model_info


# model, scaler, location_freq_map, feature_columns, compass_to_deg, model_info = load_artifacts(MODEL_DIR)
# DECISION_THRESHOLD = float(model_info.get("threshold", 0.5))

# locations = sorted(location_freq_map.index.tolist())
# directions = list(compass_to_deg.keys())


# @st.cache_data
# def load_full_test_evaluation(model_dir, data_dir):
#     """Recompute a full classification report (both classes + macro/weighted
#     avg) from the saved held-out test set, so the app can show overall model
#     quality, not just the minority-class F1 stored in model_info.json."""
#     if data_dir is None:
#         return None
#     test_path = os.path.join(data_dir, "test_set_scaled.csv")
#     if not os.path.exists(test_path):
#         return None

#     test_df = pd.read_csv(test_path)
#     if "RainTomorrow_actual" not in test_df.columns:
#         return None

#     y_true = test_df["RainTomorrow_actual"].astype(int)
#     X = test_df.drop(columns=["RainTomorrow_actual"])
#     proba = model.predict_proba(X)[:, 1]
#     pred = (proba >= DECISION_THRESHOLD).astype(int)

#     report = classification_report(y_true, pred, target_names=["No", "Yes"], output_dict=True)
#     cm = confusion_matrix(y_true, pred)
#     return {"report": report, "confusion_matrix": cm, "n": len(y_true)}


# FULL_EVAL = load_full_test_evaluation(MODEL_DIR, DATA_DIR)


# # ----------------------------------------------------------------------------
# # Shared feature engineering (mirrors the training notebook exactly)
# # ----------------------------------------------------------------------------
# def month_to_season(m):
#     if m in [12, 1, 2]:
#         return "Summer"
#     if m in [3, 4, 5]:
#         return "Autumn"
#     if m in [6, 7, 8]:
#         return "Winter"
#     return "Spring"


# def _rain_today_to_binary(v):
#     if isinstance(v, str):
#         return 1 if v.strip().lower() == "yes" else 0
#     if pd.isna(v):
#         return 0
#     try:
#         return int(v)
#     except (TypeError, ValueError):
#         return 0


# def engineer_features(raw_df):
#     df = raw_df.copy()

#     if "Season" not in df.columns:
#         if "Date" in df.columns:
#             df["Season"] = pd.to_datetime(df["Date"]).dt.month.apply(month_to_season)
#         else:
#             df["Season"] = "Summer"

#     df["RainToday"] = df["RainToday"].apply(_rain_today_to_binary).astype(int)

#     df["TempRange"] = df["MaxTemp"] - df["MinTemp"]
#     df["PressureDiff"] = df["Pressure9am"] - df["Pressure3pm"]
#     df["HumidityDiff"] = df["Humidity9am"] - df["Humidity3pm"]
#     df["WindSpeedDiff"] = df["WindSpeed3pm"] - df["WindSpeed9am"]
#     df["Humidity_Pressure_Interaction"] = df["Humidity3pm"] * (1013 - df["Pressure3pm"])
#     df["High_Humidity_Flag"] = (df["Humidity3pm"] >= 70).astype(int)

#     for prefix, col in [
#         ("WindGustDir", "WindGustDir"),
#         ("WindDir9am", "WindDir9am"),
#         ("WindDir3pm", "WindDir3pm"),
#     ]:
#         deg = df[col].astype(str).str.strip().map(compass_to_deg)
#         rad = np.deg2rad(deg)
#         df[f"{prefix}_sin"] = np.sin(rad)
#         df[f"{prefix}_cos"] = np.cos(rad)

#     for s in ["Spring", "Summer", "Winter"]:
#         df[f"Season_{s}"] = (df["Season"] == s).astype(int)

#     mean_freq = float(np.mean(location_freq_map.values))
#     df["Location_freq"] = df["Location"].map(location_freq_map).fillna(mean_freq)

#     for col in feature_columns:
#         if col not in df.columns:
#             df[col] = 0

#     return df[feature_columns], df


# def predict_proba(X_df):
#     X_scaled = pd.DataFrame(scaler.transform(X_df), columns=X_df.columns)
#     return model.predict_proba(X_scaled)[:, 1]


# def render_gauge(proba, verdict_color, size=220):
#     """Returns a self-contained SVG string for the rain-probability gauge.

#     Kept flush-left with no leading blank line: Streamlit runs Markdown
#     before HTML, and Markdown treats a blank line followed by 4+ spaces of
#     indentation as a fenced code block - that was silently turning this
#     gauge into a wall of raw <svg> text instead of a rendered graphic.
#     """
#     angle = 180 * proba
#     needle_len = size * 0.36
#     cx, cy = size / 2, size * 0.55

#     rad = math.radians(180 - angle)
#     nx = cx + needle_len * math.cos(rad)
#     ny = cy - needle_len * math.sin(rad)

#     ticks = ""
#     for pct in range(0, 101, 10):
#         a = math.radians(180 - 180 * pct / 100)
#         x1 = cx + (size * 0.42) * math.cos(a)
#         y1 = cy - (size * 0.42) * math.sin(a)
#         x2 = cx + (size * 0.47) * math.cos(a)
#         y2 = cy - (size * 0.47) * math.sin(a)
#         ticks += f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{TEXT_MUTED}" stroke-width="1.4"/>'

#     thr_angle = math.radians(180 - 180 * DECISION_THRESHOLD)
#     tx1 = cx + (size * 0.40) * math.cos(thr_angle)
#     ty1 = cy - (size * 0.40) * math.sin(thr_angle)
#     tx2 = cx + (size * 0.49) * math.cos(thr_angle)
#     ty2 = cy - (size * 0.49) * math.sin(thr_angle)

#     svg_lines = [
#         f'<svg width="{size}" height="{size*0.62:.0f}" viewBox="0 0 {size} {size*0.62:.0f}">',
#         f'<path d="M {cx - size*0.42} {cy} A {size*0.42} {size*0.42} 0 0 1 {cx + size*0.42} {cy}" '
#         f'fill="none" stroke="{BG_PANEL_ALT}" stroke-width="14" stroke-linecap="round"/>',
#         f'<path d="M {cx - size*0.42} {cy} A {size*0.42} {size*0.42} 0 0 1 {cx + size*0.42} {cy}" '
#         f'fill="none" stroke="url(#gaugeGrad)" stroke-width="14" stroke-linecap="round" '
#         f'stroke-dasharray="{math.pi*size*0.42:.1f}" stroke-dashoffset="{math.pi*size*0.42*(1-proba):.1f}"/>',
#         '<defs><linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">'
#         f'<stop offset="0%" stop-color="{GREEN_LIGHT}"/>'
#         f'<stop offset="55%" stop-color="{GREEN_MID}"/>'
#         f'<stop offset="100%" stop-color="{ALERT_RUST}"/>'
#         '</linearGradient></defs>',
#         ticks,
#         f'<line x1="{tx1:.1f}" y1="{ty1:.1f}" x2="{tx2:.1f}" y2="{ty2:.1f}" stroke="{TEXT_PRIMARY}" stroke-width="2" stroke-dasharray="2,2"/>',
#         f'<line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="{verdict_color}" stroke-width="3.2" stroke-linecap="round"/>',
#         f'<circle cx="{cx}" cy="{cy}" r="6.5" fill="{verdict_color}"/>',
#         f'<text x="{cx}" y="{cy - size*0.18:.0f}" text-anchor="middle" font-family="Space Grotesk, sans-serif" '
#         f'font-size="{size*0.16:.0f}" font-weight="700" fill="{TEXT_PRIMARY}">{proba*100:.1f}%</text>',
#         f'<text x="{cx}" y="{cy - size*0.02:.0f}" text-anchor="middle" font-family="IBM Plex Mono, monospace" '
#         f'font-size="{size*0.052:.0f}" letter-spacing="1.5" fill="{TEXT_MUTED}">PELUANG HUJAN</text>',
#         "</svg>",
#     ]
#     return "".join(svg_lines)


# # ----------------------------------------------------------------------------
# # Gemini assistant helpers
# # ----------------------------------------------------------------------------
# def get_gemini_key():
#     key = os.environ.get("GEMINI_API_KEY")
#     if not key:
#         try:
#             key = st.secrets.get("GEMINI_API_KEY")
#         except Exception:
#             key = None
#     if not key:
#         key = st.session_state.get("manual_gemini_key")
#     return key


# def call_gemini(prompt, api_key):
#     from google import genai

#     client = genai.Client(api_key=api_key)
#     response = client.models.generate_content(
#         model="gemini-2.5-flash-lite",
#         contents=prompt,
#     )
#     return response.text


# def build_grounding_context():
#     parts = [
#         f"Model produksi: {model_info['best_model_name']}",
#         f"Metrik model (kelas Yes/minoritas) - Accuracy: {model_info['metrics']['accuracy']:.3f}, "
#         f"F1: {model_info['metrics']['f1']:.3f}, ROC-AUC: {model_info['metrics']['roc_auc']:.3f}",
#         f"Threshold keputusan: {DECISION_THRESHOLD:.3f}",
#     ]
#     if FULL_EVAL:
#         r = FULL_EVAL["report"]
#         parts.append(
#             "Evaluasi keseluruhan test set - "
#             f"macro avg F1: {r['macro avg']['f1-score']:.3f}, "
#             f"weighted avg F1: {r['weighted avg']['f1-score']:.3f}, "
#             f"F1 kelas No: {r['No']['f1-score']:.3f}, F1 kelas Yes: {r['Yes']['f1-score']:.3f}."
#         )
#     if "realistic_benchmark_note" in model_info:
#         parts.append(f"Catatan benchmark: {model_info['realistic_benchmark_note']}")

#     conclusion_path = None
#     for d in DATA_CANDIDATE_DIRS:
#         p = os.path.join(d, "conclusion.txt")
#         if os.path.exists(p):
#             conclusion_path = p
#             break
#     if conclusion_path:
#         with open(conclusion_path) as f:
#             parts.append("Ringkasan kesimpulan notebook:\n" + f.read())

#     last = st.session_state.get("last_prediction")
#     if last:
#         parts.append(
#             f"Prediksi terakhir yang dijalankan pengguna: lokasi {last['lokasi']}, "
#             f"tanggal {last['tanggal']}, probabilitas hujan besok {last['probabilitas_%']}%, "
#             f"verdict {last['prediksi']}."
#         )
#     return "\n".join(parts)


# # ----------------------------------------------------------------------------
# # Sidebar
# # ----------------------------------------------------------------------------
# with st.sidebar:
#     if LOGO_PATH:
#         st.image(LOGO_PATH, width=150)
#     else:
#         st.markdown(
#             f'<div style="display:flex; align-items:center; gap:0.6rem;">'
#             f'{render_fallback_logo(38, on_dark=False)}'
#             f'<span style="font-family:\'Space Grotesk\',sans-serif; font-weight:700; '
#             f'font-size:1.05rem; color:{GREEN_DEEP};">AIGRA EON</span></div>',
#             unsafe_allow_html=True,
#         )
#         st.markdown(
#             '<span class="footnote">Taruh `aigra_logo.png` satu folder dengan '
#             "streamlit_app.py untuk memakai logo resmi.</span>",
#             unsafe_allow_html=True,
#         )

#     st.markdown("<hr class='hairline'>", unsafe_allow_html=True)
#     st.markdown(f'<div class="eyebrow">{icon("leaf")}Konfigurasi Stasiun</div>', unsafe_allow_html=True)
#     st.markdown("#### Lokasi & Waktu")
#     location = st.selectbox("Lokasi pengamatan", locations, index=locations.index("Sydney") if "Sydney" in locations else 0)
#     obs_date = st.date_input("Tanggal pengamatan", datetime.date.today())
#     season = month_to_season(obs_date.month)
#     st.markdown(f'<span class="station-id">MUSIM TERDETEKSI: {season.upper()}</span>', unsafe_allow_html=True)

#     st.markdown("<hr class='hairline'>", unsafe_allow_html=True)
#     st.markdown(f'<div class="eyebrow">{icon("cpu")}Model Aktif</div>', unsafe_allow_html=True)
#     st.markdown(f"**{model_info['best_model_name']}**")
#     if FULL_EVAL:
#         macro_f1 = FULL_EVAL["report"]["macro avg"]["f1-score"]
#         st.markdown(
#             f"<span class='footnote'>Macro F1 {macro_f1:.3f} &middot; "
#             f"F1 (Yes) {model_info['metrics']['f1']:.3f} &middot; "
#             f"ROC-AUC {model_info['metrics']['roc_auc']:.3f} &middot; "
#             f"Acc {model_info['metrics']['accuracy']:.3f} &middot; "
#             f"Threshold {DECISION_THRESHOLD:.2f}</span>",
#             unsafe_allow_html=True,
#         )
#     else:
#         st.markdown(
#             f"<span class='footnote'>F1 (Yes) {model_info['metrics']['f1']:.3f} &middot; "
#             f"ROC-AUC {model_info['metrics']['roc_auc']:.3f} &middot; "
#             f"Acc {model_info['metrics']['accuracy']:.3f} &middot; "
#             f"Threshold {DECISION_THRESHOLD:.2f}</span>",
#             unsafe_allow_html=True,
#         )
#     if "tuning" in model_info:
#         st.markdown(
#             f"<span class='footnote'>Dibandingkan dari {model_info.get('n_models_compared', '—')} model, "
#             f"tuned via RandomizedSearchCV ({model_info['tuning']['tuned_base_model']})</span>",
#             unsafe_allow_html=True,
#         )

#     st.markdown("<hr class='hairline'>", unsafe_allow_html=True)
#     st.markdown(f'<div class="eyebrow">{icon("layers")}Riwayat Sesi</div>', unsafe_allow_html=True)
#     n_log = len(st.session_state.get("log", []))
#     st.markdown(f"<span class='footnote'>{n_log} prediksi tercatat sesi ini</span>", unsafe_allow_html=True)
#     if n_log and st.button("Bersihkan riwayat", width="stretch"):
#         st.session_state["log"] = []
#         st.rerun()

#     st.markdown("<hr class='hairline'>", unsafe_allow_html=True)
#     st.markdown(f'<div class="eyebrow">{icon("bot")}Asisten AI (Gemini)</div>', unsafe_allow_html=True)
#     if not get_gemini_key():
#         st.markdown(
#             "<span class='footnote'>API key belum diset lewat secrets/env. "
#             "Masukkan sementara di bawah (tidak disimpan permanen):</span>",
#             unsafe_allow_html=True,
#         )
#         manual_key = st.text_input("GEMINI_API_KEY", type="password", key="manual_gemini_key_input")
#         if manual_key:
#             st.session_state["manual_gemini_key"] = manual_key
#     else:
#         st.markdown("<span class='footnote'>API key aktif.</span>", unsafe_allow_html=True)

# # ----------------------------------------------------------------------------
# # Hero
# # ----------------------------------------------------------------------------
# if LOGO_PATH:
#     logo_html = f'<div class="hero-logo-box"><img src="app/static/{os.path.basename(LOGO_PATH)}"/></div>'
#     # Streamlit can't reliably embed a local file via a raw <img src> tag from
#     # disk, so show the real logo with st.image right above the hero instead,
#     # and keep the mark inside the hero as the compact fallback glyph.
#     logo_html = f'<div class="hero-logo-box">{render_fallback_logo(44, on_dark=True)}</div>'
# else:
#     logo_html = f'<div class="hero-logo-box">{render_fallback_logo(44, on_dark=True)}</div>'

# st.markdown(
#     f"""<div class="hero">
# <div class="hero-brand">
# {logo_html}
# <div>
# <div class="eyebrow on-dark">Aigra Eon &middot; Smart Farming Weather Intelligence</div>
# <p class="hero-title">Stasiun Prediksi Cuaca</p>
# <p class="hero-sub">Estimasi peluang hujan esok hari dari kondisi atmosfer hari ini, membantu petani
# merencanakan penyiraman, pemupukan, dan panen dengan lebih presisi (by Nazril Ravi Pratama).</p>
# </div>
# </div>
# <div class="hero-chips">
# <div class="hero-chip"><div class="metric-label">Lokasi</div><div class="metric-value">{location}</div></div>
# <div class="hero-chip"><div class="metric-label">Tanggal</div><div class="metric-value">{obs_date.strftime('%d %b %Y')}</div></div>
# <div class="hero-chip"><div class="metric-label">Musim</div><div class="metric-value">{season}</div></div>
# </div>
# </div>""",
#     unsafe_allow_html=True,
# )

# if "log" not in st.session_state:
#     st.session_state["log"] = []
# if "chat_history" not in st.session_state:
#     st.session_state["chat_history"] = []

# tab_predict, tab_batch, tab_insight, tab_assistant, tab_about = st.tabs(
#     ["Prediksi", "Prediksi Batch", "Wawasan Model", "Asisten AI", "Tentang"]
# )

# # ----------------------------------------------------------------------------
# # TAB 1 - Single prediction
# # ----------------------------------------------------------------------------
# with tab_predict:
#     left, right = st.columns([1.15, 1])

#     with left:
#         st.markdown(f'<div class="panel">{panel_title("Suhu (&deg;C)", "thermometer")}', unsafe_allow_html=True)
#         colA, colB = st.columns(2)
#         with colA:
#             min_temp = st.slider("MinTemp", -10.0, 45.0, 15.0)
#             temp_9am = st.slider("Temp9am", -10.0, 45.0, 18.0)
#         with colB:
#             max_temp = st.slider("MaxTemp", -5.0, 50.0, 25.0)
#             temp_3pm = st.slider("Temp3pm", -5.0, 48.0, 23.0)
#         st.markdown("</div>", unsafe_allow_html=True)

#         st.markdown(f'<div class="panel">{panel_title("Kelembapan &amp; Tekanan Udara", "droplet")}', unsafe_allow_html=True)
#         colC, colD = st.columns(2)
#         with colC:
#             humidity_9am = st.slider("Humidity9am (%)", 0, 100, 60)
#             pressure_9am = st.slider("Pressure9am (hPa)", 970.0, 1040.0, 1015.0)
#         with colD:
#             humidity_3pm = st.slider("Humidity3pm (%)", 0, 100, 45)
#             pressure_3pm = st.slider("Pressure3pm (hPa)", 970.0, 1040.0, 1012.0)
#         st.markdown("</div>", unsafe_allow_html=True)

#         st.markdown(f'<div class="panel">{panel_title("Angin", "wind")}', unsafe_allow_html=True)
#         colE, colF, colG = st.columns(3)
#         with colE:
#             wind_gust_speed = st.slider("WindGustSpeed", 0.0, 140.0, 40.0)
#             wind_gust_dir = st.selectbox("WindGustDir", directions, index=directions.index("W"))
#         with colF:
#             wind_speed_9am = st.slider("WindSpeed9am", 0.0, 100.0, 15.0)
#             wind_dir_9am = st.selectbox("WindDir9am", directions, index=directions.index("W"))
#         with colG:
#             wind_speed_3pm = st.slider("WindSpeed3pm", 0.0, 100.0, 18.0)
#             wind_dir_3pm = st.selectbox("WindDir3pm", directions, index=directions.index("WNW"))
#         st.markdown("</div>", unsafe_allow_html=True)

#         st.markdown(f'<div class="panel">{panel_title("Curah Hujan Hari Ini", "rain")}', unsafe_allow_html=True)
#         colH, colI = st.columns(2)
#         with colH:
#             rainfall = st.number_input("Rainfall (mm)", 0.0, 400.0, 0.0, step=0.5)
#         with colI:
#             rain_today = st.selectbox("RainToday", ["No", "Yes"])
#         st.markdown("</div>", unsafe_allow_html=True)

#         predict_clicked = st.button("Jalankan Prediksi", width="stretch")

#     with right:
#         if predict_clicked:
#             raw_row = pd.DataFrame([{
#                 "Location": location, "Season": season,
#                 "MinTemp": min_temp, "MaxTemp": max_temp, "Rainfall": rainfall,
#                 "WindGustSpeed": wind_gust_speed, "WindGustDir": wind_gust_dir,
#                 "WindSpeed9am": wind_speed_9am, "WindSpeed3pm": wind_speed_3pm,
#                 "WindDir9am": wind_dir_9am, "WindDir3pm": wind_dir_3pm,
#                 "Humidity9am": humidity_9am, "Humidity3pm": humidity_3pm,
#                 "Pressure9am": pressure_9am, "Pressure3pm": pressure_3pm,
#                 "Temp9am": temp_9am, "Temp3pm": temp_3pm, "RainToday": rain_today,
#             }])
#             X_input, _ = engineer_features(raw_row)
#             proba = float(predict_proba(X_input)[0])
#             verdict = "Yes" if proba >= DECISION_THRESHOLD else "No"
#             verdict_color = ALERT_RUST if verdict == "Yes" else GREEN_MID
#             badge_class = "badge-yes" if verdict == "Yes" else "badge-no"
#             badge_text = "Kemungkinan hujan" if verdict == "Yes" else "Kemungkinan cerah"

#             st.markdown(
#                 f"""<div class="panel" style="text-align:center;">
# {panel_title("Pembacaan Instrumen", "gauge")}
# {render_gauge(proba, verdict_color)}
# <div style="margin-top:0.6rem;"><span class="{badge_class}">{badge_text}</span></div>
# <div class="footnote" style="margin-top:0.4rem;">Garis putus-putus = threshold keputusan model ({DECISION_THRESHOLD:.2f})</div>
# </div>""",
#                 unsafe_allow_html=True,
#             )

#             reco_text = (
#                 "Tunda penyiraman terjadwal dan pastikan drainase lahan siap. "
#                 "Aktivitas panen sensitif air sebaiknya dipercepat sebelum hujan turun."
#                 if verdict == "Yes" else
#                 "Kondisi relatif kondusif untuk penyiraman terjadwal atau panen. "
#                 "Tetap pantau prakiraan resmi BMKG sebagai referensi tambahan."
#             )
#             reco_class = "rec-box alert" if verdict == "Yes" else "rec-box"
#             st.markdown(f'<div class="{reco_class}"><b>Rekomendasi:</b> {reco_text}</div>', unsafe_allow_html=True)

#             st.markdown(f'<div class="panel" style="margin-top:0.9rem;">{panel_title("Turunan Fitur Cuaca", "chart")}', unsafe_allow_html=True)
#             derived = {
#                 "TempRange": max_temp - min_temp,
#                 "PressureDiff": pressure_9am - pressure_3pm,
#                 "HumidityDiff": humidity_9am - humidity_3pm,
#                 "WindSpeedDiff": wind_speed_3pm - wind_speed_9am,
#             }
#             st.markdown('<div class="metric-row">', unsafe_allow_html=True)
#             for k, v in derived.items():
#                 st.markdown(
#                     f'<div class="metric-card"><div class="metric-label">{k}</div>'
#                     f'<div class="metric-value" style="font-size:1.05rem;">{v:.1f}</div></div>',
#                     unsafe_allow_html=True,
#                 )
#             st.markdown("</div></div>", unsafe_allow_html=True)

#             entry = {
#                 "waktu": datetime.datetime.now().strftime("%H:%M:%S"),
#                 "lokasi": location,
#                 "tanggal": obs_date.strftime("%Y-%m-%d"),
#                 "probabilitas_%": round(proba * 100, 1),
#                 "prediksi": verdict,
#             }
#             st.session_state["log"].append(entry)
#             st.session_state["last_prediction"] = entry
#         else:
#             st.markdown(
#                 f"""<div class="panel" style="text-align:center; padding:2rem 1rem 1.6rem;">
# {panel_title("Menunggu Input", "gauge")}
# <div class="gauge-idle">
# <span class="footnote" style="margin-bottom:0.6rem;">0%</span>
# </div>
# <p class="footnote" style="margin-top:1rem;">
# Atur parameter cuaca di sebelah kiri, lalu klik &ldquo;Jalankan Prediksi&rdquo;.</p>
# </div>""",
#                 unsafe_allow_html=True,
#             )

#     if st.session_state["log"]:
#         st.markdown("<hr class='hairline'>", unsafe_allow_html=True)
#         st.markdown(f'<div class="eyebrow">{icon("layers")}Log Prediksi Sesi Ini</div>', unsafe_allow_html=True)
#         log_df = pd.DataFrame(st.session_state["log"]).iloc[::-1].reset_index(drop=True)
#         st.dataframe(log_df, width="stretch", hide_index=True)
#         st.download_button(
#             "Unduh log (.csv)",
#             log_df.to_csv(index=False).encode("utf-8"),
#             file_name="log_prediksi.csv",
#             mime="text/csv",
#         )

# # ----------------------------------------------------------------------------
# # TAB 2 - Batch prediction
# # ----------------------------------------------------------------------------
# with tab_batch:
#     st.markdown(f'<div class="eyebrow">{icon("layers")}Prediksi Massal dari File CSV</div>', unsafe_allow_html=True)
#     st.markdown(
#         "Unggah beberapa baris data cuaca sekaligus (format menyerupai dataset weatherAUS) "
#         "untuk mendapatkan prediksi RainTomorrow secara batch."
#     )

#     template_cols = [
#         "Location", "Date", "MinTemp", "MaxTemp", "Rainfall", "WindGustSpeed", "WindGustDir",
#         "WindSpeed9am", "WindSpeed3pm", "WindDir9am", "WindDir3pm", "Humidity9am", "Humidity3pm",
#         "Pressure9am", "Pressure3pm", "Temp9am", "Temp3pm", "RainToday",
#     ]
#     template_row = [
#         "Sydney", "2026-07-17", 15.0, 25.0, 0.0, 40.0, "W",
#         15.0, 18.0, "W", "WNW", 60, 45, 1015.0, 1012.0, 18.0, 23.0, "No",
#     ]
#     template_df = pd.DataFrame([template_row], columns=template_cols)

#     colT1, colT2 = st.columns([1, 1])
#     with colT1:
#         st.download_button(
#             "Unduh template CSV",
#             template_df.to_csv(index=False).encode("utf-8"),
#             file_name="template_prediksi_batch.csv",
#             mime="text/csv",
#             width="stretch",
#         )
#     with colT2:
#         uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

#     if uploaded_file is not None:
#         try:
#             batch_raw = pd.read_csv(uploaded_file)
#             missing_cols = [c for c in template_cols if c not in batch_raw.columns]
#             if missing_cols:
#                 st.error(f"Kolom berikut tidak ditemukan pada file: {missing_cols}")
#             else:
#                 X_batch, _ = engineer_features(batch_raw)
#                 probs = predict_proba(X_batch)
#                 result_df = batch_raw.copy()
#                 result_df["Prob_RainTomorrow_%"] = (probs * 100).round(2)
#                 result_df["Prediksi"] = np.where(probs >= DECISION_THRESHOLD, "Yes", "No")

#                 st.markdown(f'<div class="panel">{panel_title("Ringkasan Batch", "chart")}', unsafe_allow_html=True)
#                 yes_pct = (result_df["Prediksi"] == "Yes").mean() * 100
#                 st.markdown('<div class="metric-row">', unsafe_allow_html=True)
#                 for label, value in [
#                     ("Total baris", f"{len(result_df)}"),
#                     ("Prediksi hujan", f"{yes_pct:.1f}%"),
#                     ("Rata-rata prob.", f"{result_df['Prob_RainTomorrow_%'].mean():.1f}%"),
#                 ]:
#                     st.markdown(
#                         f'<div class="metric-card"><div class="metric-label">{label}</div>'
#                         f'<div class="metric-value" style="font-size:1.1rem;">{value}</div></div>',
#                         unsafe_allow_html=True,
#                     )
#                 st.markdown("</div></div>", unsafe_allow_html=True)

#                 st.dataframe(result_df, width="stretch", hide_index=True)
#                 st.download_button(
#                     "Unduh hasil prediksi (.csv)",
#                     result_df.to_csv(index=False).encode("utf-8"),
#                     file_name="hasil_prediksi_batch.csv",
#                     mime="text/csv",
#                 )
#         except Exception as exc:
#             st.error(f"Gagal memproses file: {exc}")

# # ----------------------------------------------------------------------------
# # TAB 3 - Model insight (full evaluation, not just minority-class F1)
# # ----------------------------------------------------------------------------
# with tab_insight:
#     st.markdown(f'<div class="eyebrow">{icon("chart")}Performa &amp; Interpretasi Model</div>', unsafe_allow_html=True)

#     if FULL_EVAL:
#         rep = FULL_EVAL["report"]
#         cm = FULL_EVAL["confusion_matrix"]

#         st.markdown(f'<div class="panel">{panel_title("Evaluasi Keseluruhan (Test Set)", "shield")}', unsafe_allow_html=True)
#         st.markdown('<div class="metric-row">', unsafe_allow_html=True)
#         overall_cards = [
#             ("Accuracy", f"{rep['accuracy']*100:.1f}%", False),
#             ("Macro Avg F1", f"{rep['macro avg']['f1-score']*100:.1f}%", True),
#             ("Weighted Avg F1", f"{rep['weighted avg']['f1-score']*100:.1f}%", True),
#             ("ROC-AUC", f"{model_info['metrics']['roc_auc']*100:.1f}%", False),
#         ]
#         for label, value, is_highlight in overall_cards:
#             css_class = "metric-card highlight" if is_highlight else "metric-card"
#             st.markdown(
#                 f'<div class="{css_class}"><div class="metric-label">{label}</div>'
#                 f'<div class="metric-value">{value}</div></div>',
#                 unsafe_allow_html=True,
#             )
#         st.markdown("</div></div>", unsafe_allow_html=True)

#         st.markdown(f'<div class="panel">{panel_title("Rincian per Kelas", "grid")}', unsafe_allow_html=True)
#         breakdown_rows = []
#         for label in ["No", "Yes", "macro avg", "weighted avg"]:
#             row = rep[label]
#             breakdown_rows.append({
#                 "Kelas": label,
#                 "Precision": round(row["precision"], 4),
#                 "Recall": round(row["recall"], 4),
#                 "F1-score": round(row["f1-score"], 4),
#                 "Support": int(row["support"]),
#             })
#         breakdown_df = pd.DataFrame(breakdown_rows).set_index("Kelas")
#         st.dataframe(breakdown_df, width="stretch")
#         st.markdown(
#             f'<p class="footnote">Dihitung ulang langsung dari test set tersimpan '
#             f'({FULL_EVAL["n"]:,} baris) setiap kali aplikasi dijalankan, jadi angka ini selalu '
#             f'konsisten dengan model &amp; threshold yang sedang aktif.</p>',
#             unsafe_allow_html=True,
#         )
#         st.markdown("</div>", unsafe_allow_html=True)

#         st.markdown(
#             '<div class="rec-box" style="margin-bottom:1rem;">'
#             "<b>Cara baca:</b> Macro/weighted avg F1 merangkum performa model di kedua kelas "
#             "sekaligus (No &amp; Yes). F1 kelas &ldquo;Yes&rdquo; sendiri biasanya lebih rendah "
#             "karena hujan adalah kelas minoritas itu wajar untuk dataset dengan class "
#             "imbalance seperti ini, bukan tanda model bermasalah.</div>",
#             unsafe_allow_html=True,
#         )

#         cm_fig, cm_ax = plt.subplots(figsize=(4.6, 4.0))
#         cm_fig.patch.set_facecolor(BG_MAIN)
#         cm_ax.set_facecolor(BG_MAIN)
#         cm_ax.imshow(cm, cmap=plt.cm.Greens)
#         for i in range(cm.shape[0]):
#             for j in range(cm.shape[1]):
#                 text_color = "white" if cm[i, j] > cm.max() / 2 else GREEN_DEEP
#                 cm_ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
#                            color=text_color, fontsize=12, fontweight="bold")
#         cm_ax.set_xticks([0, 1]); cm_ax.set_yticks([0, 1])
#         cm_ax.set_xticklabels(["No", "Yes"], color=TEXT_PRIMARY)
#         cm_ax.set_yticklabels(["No", "Yes"], color=TEXT_PRIMARY)
#         cm_ax.set_xlabel("Prediksi", color=TEXT_MUTED)
#         cm_ax.set_ylabel("Aktual", color=TEXT_MUTED)
#         cm_ax.set_title(f"Confusion Matrix - {model_info['best_model_name']}", color=GREEN_DEEP, fontsize=11)
#         for spine in cm_ax.spines.values():
#             spine.set_color(BORDER_MPL)
#         st.markdown(f'<div class="panel">{panel_title("Confusion Matrix", "grid")}', unsafe_allow_html=True)
#         st.pyplot(cm_fig)
#         st.markdown("</div>", unsafe_allow_html=True)
#     else:
#         st.warning(
#             "File `test_set_scaled.csv` tidak ditemukan di folder data, jadi evaluasi keseluruhan "
#             "(macro/weighted avg) tidak bisa dihitung ulang. Menampilkan metrik kelas minoritas "
#             "yang tersimpan di model_info.json saja."
#         )
#         m = model_info["metrics"]
#         st.markdown('<div class="metric-row">', unsafe_allow_html=True)
#         for label, key in [
#             ("Accuracy", "accuracy"), ("Precision (Yes)", "precision"), ("Recall (Yes)", "recall"),
#             ("F1 (Yes)", "f1"), ("ROC-AUC", "roc_auc"), ("Threshold", "threshold"),
#         ]:
#             val = m.get(key, DECISION_THRESHOLD)
#             display_val = f"{val*100:.1f}%" if key != "threshold" else f"{val:.3f}"
#             st.markdown(
#                 f'<div class="metric-card"><div class="metric-label">{label}</div>'
#                 f'<div class="metric-value">{display_val}</div></div>',
#                 unsafe_allow_html=True,
#             )
#         st.markdown("</div>", unsafe_allow_html=True)

#     if "realistic_benchmark_note" in model_info:
#         st.markdown(
#             f'<div class="rec-box" style="margin-top:0.9rem;"><b>Catatan realistis:</b> '
#             f'{model_info["realistic_benchmark_note"]}</div>',
#             unsafe_allow_html=True,
#         )

#     st.markdown("<br>", unsafe_allow_html=True)

#     if "tuning" in model_info:
#         t = model_info["tuning"]
#         st.markdown(f'<div class="panel">{panel_title("Hyperparameter Tuning", "cpu")}', unsafe_allow_html=True)
#         st.markdown(
#             f"Model dasar **{t['tuned_base_model']}** di-tuning dengan RandomizedSearchCV "
#             f"(CV F1: **{t['cv_f1_score']:.4f}**)."
#         )
#         st.json(t["best_params"])
#         st.markdown("</div>", unsafe_allow_html=True)

#     if DATA_DIR:
#         st.markdown(f'<div class="panel">{panel_title("Perbandingan Seluruh Model", "layers")}', unsafe_allow_html=True)
#         comp_df = pd.read_csv(os.path.join(DATA_DIR, "model_comparison.csv"), index_col=0)
#         st.dataframe(comp_df.style.format("{:.4f}"), width="stretch")

#         fig, ax = plt.subplots(figsize=(9, 4.5))
#         fig.patch.set_facecolor(BG_PANEL)
#         ax.set_facecolor(BG_PANEL)
#         comp_df[["f1", "roc_auc"]].sort_values("f1").plot(
#             kind="barh", ax=ax, color=[GREEN_MID, GREEN_LIGHT], width=0.7
#         )
#         ax.set_xlabel("Skor", color=TEXT_MUTED)
#         ax.tick_params(colors=TEXT_MUTED)
#         ax.legend(facecolor=BG_PANEL, labelcolor=TEXT_PRIMARY, edgecolor=BORDER_MPL)
#         for spine in ax.spines.values():
#             spine.set_color(BORDER_MPL)
#         st.pyplot(fig)
#         st.markdown("</div>", unsafe_allow_html=True)

#     if FIG_DIR:
#         st.markdown(f'<div class="panel">{panel_title("Feature Importance (dari Notebook)", "leaf")}', unsafe_allow_html=True)
#         st.image(os.path.join(FIG_DIR, "13_feature_importance.png"), width="stretch")
#         st.markdown("</div>", unsafe_allow_html=True)
#     else:
#         st.info(
#             "Gambar feature importance belum ditemukan. Salin folder `outputs/figures` dari hasil "
#             "zip notebook ke direktori yang sama dengan streamlit_app.py untuk menampilkannya."
#         )

# # ----------------------------------------------------------------------------
# # TAB 4 - Gemini AI Assistant
# # ----------------------------------------------------------------------------
# with tab_assistant:
#     st.markdown(f'<div class="eyebrow">{icon("bot")}Asisten AI &middot; Gemini 2.5 Flash-Lite</div>', unsafe_allow_html=True)
#     st.markdown(
#         "Tanyakan apa pun seputar hasil prediksi, kualitas model, atau rekomendasi pertanian. "
#         "Asisten ini dibekali konteks (RAG ringan) dari metrik model dan prediksi terakhir yang "
#         "kamu jalankan di tab Prediksi."
#     )

#     api_key = get_gemini_key()
#     if not api_key:
#         st.warning(
#             "API key Gemini belum tersedia. Set environment variable `GEMINI_API_KEY`, "
#             "isi `st.secrets['GEMINI_API_KEY']` di `.streamlit/secrets.toml`, atau masukkan "
#             "sementara lewat sidebar. **Jangan hardcode API key langsung di kode**, terutama "
#             "jika repo ini akan diunggah ke GitHub publik."
#         )
#     else:
#         for msg in st.session_state["chat_history"]:
#             with st.chat_message(msg["role"]):
#                 st.markdown(msg["content"])

#         user_question = st.chat_input("Tulis pertanyaanmu di sini...")
#         if user_question:
#             st.session_state["chat_history"].append({"role": "user", "content": user_question})
#             with st.chat_message("user"):
#                 st.markdown(user_question)

#             context = build_grounding_context()
#             system_prompt = (
#                 "Kamu adalah asisten AI untuk aplikasi prediksi cuaca smart farming bernama "
#                 "'Stasiun Prediksi Cuaca'. Jawab dalam Bahasa Indonesia, singkat, jelas, dan "
#                 "praktis untuk petani. Gunakan konteks berikut jika relevan, dan jangan mengarang "
#                 "angka yang tidak ada di konteks:\n\n"
#                 f"{context}\n\nPertanyaan pengguna: {user_question}"
#             )

#             with st.chat_message("assistant"):
#                 with st.spinner("Berpikir..."):
#                     try:
#                         answer = call_gemini(system_prompt, api_key)
#                     except Exception as exc:
#                         answer = f"Maaf, terjadi kendala saat menghubungi Gemini API: {exc}"
#                 st.markdown(answer)
#             st.session_state["chat_history"].append({"role": "assistant", "content": answer})

#         if st.session_state["chat_history"] and st.button("Bersihkan percakapan"):
#             st.session_state["chat_history"] = []
#             st.rerun()

# # ----------------------------------------------------------------------------
# # TAB 5 - About
# # ----------------------------------------------------------------------------
# with tab_about:
#     st.markdown(f'<div class="eyebrow">{icon("info")}Tentang Proyek</div>', unsafe_allow_html=True)
#     st.markdown(
#         f"""
#         <div class="panel">
#         <p>Aplikasi ini adalah antarmuka prediksi <b>RainTomorrow</b> yang dilatih pada dataset
#         <i>Rain in Australia</i> (Kaggle), dibangun sebagai bagian dari coding test AI Engineer
#         Intern PT AIGRA EON INDONESIA - Smart Farming.</p>
#         <p>Sebanyak {model_info.get('n_models_compared', 'beberapa')} varian model dibandingkan
#         (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, SVM,
#         Neural Network/Deep Learning, model hasil tuning, Stacking Ensemble, dan Blended
#         Ensemble), dengan penanganan class imbalance lewat class weighting serta optimisasi
#         threshold keputusan pada validation set. Model produksi yang dipakai aplikasi ini:
#         <b>{model_info['best_model_name']}</b> (threshold={DECISION_THRESHOLD:.3f}).</p>
#         <p class="footnote">Catatan: prediksi ini bersifat estimatif berdasarkan data historis dan
#         sebaiknya dikombinasikan dengan prakiraan resmi BMKG untuk keputusan pertanian penting.</p>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )
#     st.markdown(f'<div class="panel">{panel_title("Fitur yang Digunakan Model", "grid")}', unsafe_allow_html=True)
#     st.code(", ".join(feature_columns), language=None)
#     st.markdown("</div>", unsafe_allow_html=True)

#     st.markdown(
#         f'<div class="panel">{panel_title("Keamanan API Key", "shield")}'
#         '<p style="font-size:0.88rem; line-height:1.5;">Aplikasi ini membaca API key Gemini dari '
#         '<code>GEMINI_API_KEY</code> (environment variable) atau <code>st.secrets</code>, bukan dari '
#         'kode sumber. Saat mengunggah repo ini ke GitHub publik, pastikan file '
#         '<code>.streamlit/secrets.toml</code> masuk <code>.gitignore</code> agar API key tidak ikut '
#         'ter-commit.</p></div>',
#         unsafe_allow_html=True,
#     )
#     st.markdown(
#         f'<div class="panel">{panel_title("Catatan Kompatibilitas Versi", "info")}'
#         '<p style="font-size:0.88rem; line-height:1.5;">Model disimpan dengan scikit-learn/XGBoost '
#         'versi tertentu saat training di notebook. Jika environment tempat Streamlit ini dijalankan '
#         'memakai versi library yang berbeda, kamu akan melihat warning '
#         '<code>InconsistentVersionWarning</code> di terminal (sudah diredam dari tampilan aplikasi, '
#         'tapi tetap muncul di log). Untuk menghilangkannya secara permanen, samakan versi '
#         '<code>scikit-learn</code> dan <code>xgboost</code> di <code>requirements.txt</code> deployment '
#         'dengan versi yang dipakai saat notebook menyimpan <code>best_model.joblib</code>.</p></div>',
#         unsafe_allow_html=True,
#     )

import os
import json
import math
import textwrap
import datetime
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

# Silence noisy version-mismatch warnings from loading pickled models trained
# on a slightly different scikit-learn/xgboost version. This does not change
# prediction results; it only hides the console warning. If predictions ever
# look wrong after a library upgrade, re-export the model with the *current*
# library versions instead of relying on this suppression.
warnings.filterwarnings("ignore", message=".*Trying to unpickle estimator.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

st.set_page_config(
    page_title="Stasiun Prediksi Cuaca | Smart Farming",
    page_icon=":seedling:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Force light appearance regardless of the user's OS/browser theme.
# st.set_page_config() has no theme parameter, and Streamlit's real theme
# engine only reads .streamlit/config.toml at server start (add
# [theme]\nbase = "light" there for the most robust fix and to avoid a brief
# dark flash before this CSS loads). This inline rule is the code-only
# fallback: `color-scheme: light` tells the browser to render all native
# controls (scrollbars, form widgets, etc.) as if the system were in light
# mode, overriding `prefers-color-scheme: dark` even when the user's OS is
# set to dark.
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        :root, .stApp {
            color-scheme: light !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Design tokens - dark green & white
# ----------------------------------------------------------------------------
# Two-tone brief: dark green + white carries the whole UI. The one deliberate
# exception is a muted brick-red used ONLY where a farmer needs a signal to
# pop above everything else - the rain-warning badge, the gauge's danger
# zone, and the threshold marker. Everything else, including "no rain",
# "info", and "success" states, stays inside the green/white system so nothing
# competes with that one real alert color.
BG_MAIN = "#FFFFFF"
BG_PANEL = "#F5F9F6"
BG_PANEL_ALT = "#E9F2EC"
BG_HERO_1 = "#0F3D26"
BG_HERO_2 = "#155C38"
BORDER = "#D7E6DC"
BORDER_MPL = "#C4D9CB"

TEXT_PRIMARY = "#0F2E1F"
TEXT_MUTED = "#5B7768"
TEXT_ON_DARK = "#F3FAF5"
TEXT_ON_DARK_MUTED = "#AFD1BE"

GREEN_DEEP = "#0F3D26"
GREEN_MID = "#1F7A4C"
GREEN_LIGHT = "#4CAF71"
GREEN_PALE = "#DCEEE1"

ALERT_RUST = "#B3392A"
ALERT_RUST_PALE = "#F4E2DE"

CANDIDATE_DIRS = ["outputs/models", "models", "."]
FIG_CANDIDATE_DIRS = ["outputs/figures", "figures"]
DATA_CANDIDATE_DIRS = ["outputs/data", "data"]
LOGO_CANDIDATES = [
    "aigra_logo.png", "aigra_logo.jpg", "aigra logo.jpg", "aigra logo.png",
    "logo.png", "logo.jpg",
    "assets/aigra_logo.png", "assets/logo.png",
    "outputs/aigra_logo.png",
]


def find_dir(candidates, filename):
    for d in candidates:
        if os.path.exists(os.path.join(d, filename)):
            return d
    return None


def find_logo():
    for path in LOGO_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


MODEL_DIR = find_dir(CANDIDATE_DIRS, "best_model.joblib")
FIG_DIR = find_dir(FIG_CANDIDATE_DIRS, "13_feature_importance.png")
DATA_DIR = find_dir(DATA_CANDIDATE_DIRS, "model_comparison.csv")
LOGO_PATH = find_logo()


# ----------------------------------------------------------------------------
# Small hand-built line-icon set (single-color, stroke-based) so panel titles
# get a visual anchor without breaking the two-tone-plus-rust palette or
# pulling in a colourful emoji/icon-font dependency.
# ----------------------------------------------------------------------------
def icon(name, color=None, size=16):
    color = color or GREEN_MID
    paths = {
        "thermometer": '<path d="M9 2.5a1.5 1.5 0 0 1 3 0v7.6a3.5 3.5 0 1 1-3 0V2.5Z"/><circle cx="10.5" cy="14" r="1.4" fill="{c}" stroke="none"/>',
        "droplet": '<path d="M10 2.5S4.5 9 4.5 13a5.5 5.5 0 0 0 11 0C15.5 9 10 2.5 10 2.5Z"/>',
        "wind": '<path d="M2.5 7h9a2 2 0 1 0-2-2"/><path d="M2.5 11h12a2 2 0 1 1-2 2"/><path d="M2.5 15h7a2 2 0 1 0-2 2"/>',
        "rain": '<path d="M5 8.5a4 4 0 0 1 .3-7.9A5 5 0 0 1 15 2a3.7 3.7 0 0 1 .5 7.4"/><path d="M6 13l-1.3 2.6M10 13l-1.3 2.6M14 13l-1.3 2.6"/>',
        "chart": '<path d="M3 17V9M8 17V4M13 17v-6M18 17H3"/>',
        "gauge": '<path d="M2.5 15a7.5 7.5 0 1 1 15 0"/><path d="M10 15 13.2 8.8"/><circle cx="10" cy="15" r="1.1" fill="{c}" stroke="none"/>',
        "cpu": '<rect x="6" y="6" width="8" height="8" rx="1"/><path d="M8.5 2.5v2M11.5 2.5v2M8.5 15.5v2M11.5 15.5v2M2.5 8.5h2M2.5 11.5h2M15.5 8.5h2M15.5 11.5h2"/>',
        "bot": '<rect x="3" y="6.5" width="14" height="9.5" rx="2.5"/><circle cx="7.5" cy="11" r="1" fill="{c}" stroke="none"/><circle cx="12.5" cy="11" r="1" fill="{c}" stroke="none"/><path d="M10 6.5V3.5"/><circle cx="10" cy="2.5" r="1" fill="{c}" stroke="none"/>',
        "layers": '<path d="M10 2.5 18 7 10 11.5 2 7Z"/><path d="M2 11 10 15.5 18 11"/>',
        "info": '<circle cx="10" cy="10" r="7.5"/><path d="M10 9v5"/><circle cx="10" cy="6.3" r="0.9" fill="{c}" stroke="none"/>',
        "leaf": '<path d="M4 16C4 8 9 3 17 3c0 8-5 13-13 13Z"/><path d="M4.5 15.5 15 5"/>',
        "grid": '<path d="M3 4h14v12H3zM3 10h14M9 4v12"/>',
        "shield": '<path d="M10 2.5 16.5 5v5.2c0 4-2.7 6.6-6.5 7.8-3.8-1.2-6.5-3.8-6.5-7.8V5Z"/>',
    }
    body = paths.get(name, paths["info"]).format(c=color)
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 20 20" fill="none" '
        f'stroke="{color}" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" '
        f'style="vertical-align:-3px; margin-right:0.4rem;">{body}</svg>'
    )


def panel_title(label, icon_name=None, right_text=None, dark=False):
    ic = icon(icon_name, color=(TEXT_ON_DARK if dark else GREEN_MID)) if icon_name else ""
    right = f'<span>{right_text}</span>' if right_text else ""
    cls = "panel-title dark" if dark else "panel-title"
    return f'<div class="{cls}"><span>{ic}{label}</span>{right}</div>'


def render_fallback_logo(size=40, on_dark=True):
    """Small leaf-in-circle mark used when no logo file is found on disk."""
    ring = TEXT_ON_DARK if on_dark else GREEN_DEEP
    fill = "none"
    svg = (
        f'<svg width="{size}" height="{size}" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">'
        f'<circle cx="20" cy="20" r="18.5" fill="{fill}" stroke="{ring}" stroke-width="1.5"/>'
        f'<path d="M12 26 C12 16 20 11 29 11 C29 20 24 28 14 28 C13 28 12.3 27.4 12 26 Z" fill="{GREEN_LIGHT if on_dark else GREEN_MID}"/>'
        f'<path d="M13 27 L26 12" stroke="{ring}" stroke-width="1.1" fill="none"/>'
        f'</svg>'
    )
    return svg


# ----------------------------------------------------------------------------
# Global CSS
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
            color: {TEXT_PRIMARY};
        }}
        .stApp {{ background: {BG_MAIN}; }}

        /* Tighten default Streamlit chrome so content starts close to the top
           instead of leaving a large dead gap under the toolbar. */
        .block-container {{
            padding-top: 1.6rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }}
        header[data-testid="stHeader"] {{
            background: rgba(255,255,255,0.72);
            backdrop-filter: blur(6px);
            border-bottom: 1px solid {BORDER};
        }}
        [data-testid="stSidebarContent"] {{ padding-top: 1rem; }}
        section[data-testid="stSidebar"] {{
            background: {BG_PANEL};
            border-right: 1px solid {BORDER};
        }}
        section[data-testid="stSidebar"] .block-container {{ padding-top: 0.5rem; }}

        h1, h2, h3, h4 {{
            font-family: 'Space Grotesk', sans-serif !important;
            letter-spacing: -0.01em;
            color: {GREEN_DEEP};
        }}
        .eyebrow {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: {GREEN_MID};
        }}
        .eyebrow.on-dark {{ color: {TEXT_ON_DARK_MUTED}; }}

        /* ---------------- Hero ---------------- */
        .hero {{
            position: relative;
            overflow: hidden;
            background: linear-gradient(135deg, {BG_HERO_1} 0%, {BG_HERO_2} 100%);
            border-radius: 20px;
            padding: 1.9rem 2.2rem;
            margin-bottom: 1.3rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1.5rem;
            flex-wrap: wrap;
        }}
        .hero::before {{
            content: "";
            position: absolute;
            inset: 0;
            background-image: repeating-linear-gradient(
                115deg,
                rgba(255,255,255,0.05) 0px,
                rgba(255,255,255,0.05) 1.5px,
                transparent 1.5px,
                transparent 34px
            );
            pointer-events: none;
        }}
        .hero-brand {{
            display: flex;
            align-items: flex-start;
            gap: 0.9rem;
            position: relative;
            z-index: 1;
            min-width: 320px;
        }}
        .hero-logo-box {{
            flex-shrink: 0;
            width: 48px; height: 48px;
            display:flex; align-items:center; justify-content:center;
        }}
        .hero-logo-box img {{
            width: 46px; height: 46px; object-fit: contain;
            border-radius: 10px;
            background: {TEXT_ON_DARK};
            padding: 4px;
        }}
        .hero-title {{
            font-size: 1.85rem;
            font-weight: 700;
            margin: 0.25rem 0 0 0;
            line-height: 1.15;
            color: {TEXT_ON_DARK};
        }}
        .hero-sub {{
            color: {TEXT_ON_DARK_MUTED};
            font-size: 0.94rem;
            margin-top: 0.35rem;
            max-width: 480px;
            line-height: 1.5;
        }}
        .hero-chips {{
            display: flex;
            gap: 0.7rem;
            position: relative;
            z-index: 1;
            flex-wrap: wrap;
        }}
        .hero-chip {{
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.16);
            border-radius: 12px;
            padding: 0.6rem 1rem;
            min-width: 110px;
        }}
        .hero-chip .metric-label {{ color: {TEXT_ON_DARK_MUTED}; }}
        .hero-chip .metric-value {{ color: {TEXT_ON_DARK}; font-size: 1.05rem; }}

        .hairline {{
            border: none;
            border-top: 1px solid {BORDER};
            margin: 1.1rem 0 1.4rem 0;
        }}
        .panel {{
            background: {BG_PANEL};
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 1.15rem 1.3rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 2px rgba(15,61,38,0.04);
        }}
        .panel-title {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.74rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: {TEXT_MUTED};
            margin-bottom: 0.8rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .panel-title.dark {{ color: {TEXT_ON_DARK_MUTED}; }}
        .metric-row {{ display: flex; gap: 0.7rem; flex-wrap: wrap; }}
        .metric-card {{
            background: {BG_PANEL_ALT};
            border: 1px solid {BORDER};
            border-radius: 10px;
            padding: 0.7rem 0.9rem;
            flex: 1;
            min-width: 120px;
        }}
        .metric-card.highlight {{ background: {GREEN_DEEP}; border-color: {GREEN_DEEP}; }}
        .metric-card.highlight .metric-label {{ color: {GREEN_PALE}; }}
        .metric-card.highlight .metric-value {{ color: #FFFFFF; }}
        .metric-label {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.65rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: {TEXT_MUTED};
        }}
        .metric-value {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.35rem;
            font-weight: 600;
            margin-top: 0.15rem;
            color: {GREEN_DEEP};
        }}
        .badge-yes, .badge-no {{
            display: inline-block;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            padding: 0.35rem 0.85rem;
            border-radius: 999px;
            font-weight: 600;
        }}
        .badge-yes {{
            background: {ALERT_RUST_PALE};
            color: {ALERT_RUST};
            border: 1px solid rgba(179, 57, 42, 0.35);
        }}
        .badge-no {{
            background: {GREEN_PALE};
            color: {GREEN_MID};
            border: 1px solid rgba(31, 122, 76, 0.4);
        }}
        .station-id {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.74rem;
            color: {TEXT_MUTED};
            letter-spacing: 0.03em;
        }}
        .stButton > button {{
            background: {GREEN_DEEP};
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.1rem;
            font-weight: 600;
            font-family: 'Space Grotesk', sans-serif;
            width: 100%;
            transition: filter 0.12s ease, transform 0.12s ease;
        }}
        .stButton > button:hover {{ filter: brightness(1.18); transform: translateY(-1px); }}
        .stDownloadButton > button {{
            background: {BG_PANEL_ALT};
            color: {GREEN_DEEP};
            border: 1px solid {BORDER};
            border-radius: 10px;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.8rem;
        }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {BORDER}; }}
        .stTabs [data-baseweb="tab"] {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.8rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: {TEXT_MUTED};
            padding: 0.6rem 1rem;
        }}
        .stTabs [aria-selected="true"] {{
            color: {GREEN_DEEP} !important;
            border-bottom: 2px solid {GREEN_DEEP} !important;
        }}
        .footnote {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: {TEXT_MUTED}; }}
        div[data-testid="stMetricValue"] {{ font-family: 'Space Grotesk', sans-serif; color: {GREEN_DEEP}; }}

        .rec-box {{
            border-left: 3px solid {GREEN_MID};
            background: {BG_PANEL_ALT};
            padding: 0.85rem 1rem;
            border-radius: 0 8px 8px 0;
            font-size: 0.92rem;
            line-height: 1.5;
            color: {TEXT_PRIMARY};
        }}
        .rec-box.alert {{ border-left-color: {ALERT_RUST}; background: {ALERT_RUST_PALE}; }}

        /* Re-skin Streamlit's built-in info/warning/error/success boxes so
           they match the green/white/rust system instead of the default
           yellow/blue palette. */
        div[data-testid="stAlert"] {{
            background: {BG_PANEL_ALT} !important;
            border: 1px solid {BORDER} !important;
            border-left: 4px solid {GREEN_MID} !important;
            border-radius: 0 10px 10px 0 !important;
        }}
        div[data-testid="stAlert"] p, div[data-testid="stAlert"] li {{ color: {TEXT_PRIMARY} !important; }}
        div[data-testid="stAlert"] svg {{ fill: {GREEN_MID} !important; }}

        [data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 10px; }}
        [data-testid="stChatMessage"] {{
            background: {BG_PANEL_ALT}; border: 1px solid {BORDER}; border-radius: 12px;
        }}
        .stSlider [data-baseweb="slider"] div[role="slider"] {{ background-color: {GREEN_DEEP} !important; }}
        .stSlider [data-baseweb="slider"] > div > div {{ background: {GREEN_MID} !important; }}
        div[data-baseweb="select"] > div {{ border-color: {BORDER} !important; }}
        input, .stNumberInput input {{ caret-color: {GREEN_DEEP}; }}

        ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
        ::-webkit-scrollbar-track {{ background: {BG_PANEL}; }}
        ::-webkit-scrollbar-thumb {{ background: {BORDER_MPL}; border-radius: 10px; }}

        .gauge-idle {{
            width: 220px; height: 136px; margin: 0 auto;
            border: 1.5px dashed {BORDER_MPL};
            border-radius: 999px 999px 0 0;
            display: flex; align-items: flex-end; justify-content: center;
            color: {TEXT_MUTED};
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

if MODEL_DIR is None:
    st.error(
        "File model tidak ditemukan. Ekstrak `rain_prediction_outputs.zip` dari notebook, "
        "lalu letakkan folder `outputs/models` satu direktori dengan streamlit_app.py."
    )
    st.stop()


@st.cache_resource
def load_artifacts(model_dir):
    model = joblib.load(os.path.join(model_dir, "best_model.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    location_freq_map = joblib.load(os.path.join(model_dir, "location_freq_map.joblib"))
    feature_columns = joblib.load(os.path.join(model_dir, "feature_columns.joblib"))
    compass_to_deg = joblib.load(os.path.join(model_dir, "compass_to_deg.joblib"))
    with open(os.path.join(model_dir, "model_info.json")) as f:
        model_info = json.load(f)
    return model, scaler, location_freq_map, feature_columns, compass_to_deg, model_info


model, scaler, location_freq_map, feature_columns, compass_to_deg, model_info = load_artifacts(MODEL_DIR)
DECISION_THRESHOLD = float(model_info.get("threshold", 0.5))

locations = sorted(location_freq_map.index.tolist())
directions = list(compass_to_deg.keys())


@st.cache_data
def load_full_test_evaluation(model_dir, data_dir):
    """Recompute a full classification report (both classes + macro/weighted
    avg) from the saved held-out test set, so the app can show overall model
    quality, not just the minority-class F1 stored in model_info.json."""
    if data_dir is None:
        return None
    test_path = os.path.join(data_dir, "test_set_scaled.csv")
    if not os.path.exists(test_path):
        return None

    test_df = pd.read_csv(test_path)
    if "RainTomorrow_actual" not in test_df.columns:
        return None

    y_true = test_df["RainTomorrow_actual"].astype(int)
    X = test_df.drop(columns=["RainTomorrow_actual"])
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= DECISION_THRESHOLD).astype(int)

    report = classification_report(y_true, pred, target_names=["No", "Yes"], output_dict=True)
    cm = confusion_matrix(y_true, pred)
    return {"report": report, "confusion_matrix": cm, "n": len(y_true)}


FULL_EVAL = load_full_test_evaluation(MODEL_DIR, DATA_DIR)


# ----------------------------------------------------------------------------
# Shared feature engineering (mirrors the training notebook exactly)
# ----------------------------------------------------------------------------
def month_to_season(m):
    if m in [12, 1, 2]:
        return "Summer"
    if m in [3, 4, 5]:
        return "Autumn"
    if m in [6, 7, 8]:
        return "Winter"
    return "Spring"


def _rain_today_to_binary(v):
    if isinstance(v, str):
        return 1 if v.strip().lower() == "yes" else 0
    if pd.isna(v):
        return 0
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def engineer_features(raw_df):
    df = raw_df.copy()

    if "Season" not in df.columns:
        if "Date" in df.columns:
            df["Season"] = pd.to_datetime(df["Date"]).dt.month.apply(month_to_season)
        else:
            df["Season"] = "Summer"

    df["RainToday"] = df["RainToday"].apply(_rain_today_to_binary).astype(int)

    df["TempRange"] = df["MaxTemp"] - df["MinTemp"]
    df["PressureDiff"] = df["Pressure9am"] - df["Pressure3pm"]
    df["HumidityDiff"] = df["Humidity9am"] - df["Humidity3pm"]
    df["WindSpeedDiff"] = df["WindSpeed3pm"] - df["WindSpeed9am"]
    df["Humidity_Pressure_Interaction"] = df["Humidity3pm"] * (1013 - df["Pressure3pm"])
    df["High_Humidity_Flag"] = (df["Humidity3pm"] >= 70).astype(int)

    for prefix, col in [
        ("WindGustDir", "WindGustDir"),
        ("WindDir9am", "WindDir9am"),
        ("WindDir3pm", "WindDir3pm"),
    ]:
        deg = df[col].astype(str).str.strip().map(compass_to_deg)
        rad = np.deg2rad(deg)
        df[f"{prefix}_sin"] = np.sin(rad)
        df[f"{prefix}_cos"] = np.cos(rad)

    for s in ["Spring", "Summer", "Winter"]:
        df[f"Season_{s}"] = (df["Season"] == s).astype(int)

    mean_freq = float(np.mean(location_freq_map.values))
    df["Location_freq"] = df["Location"].map(location_freq_map).fillna(mean_freq)

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    return df[feature_columns], df


def predict_proba(X_df):
    X_scaled = pd.DataFrame(scaler.transform(X_df), columns=X_df.columns)
    return model.predict_proba(X_scaled)[:, 1]


def render_gauge(proba, verdict_color, size=220):
    """Returns a self-contained SVG string for the rain-probability gauge.

    Kept flush-left with no leading blank line: Streamlit runs Markdown
    before HTML, and Markdown treats a blank line followed by 4+ spaces of
    indentation as a fenced code block - that was silently turning this
    gauge into a wall of raw <svg> text instead of a rendered graphic.
    """
    angle = 180 * proba
    needle_len = size * 0.36
    cx, cy = size / 2, size * 0.55

    rad = math.radians(180 - angle)
    nx = cx + needle_len * math.cos(rad)
    ny = cy - needle_len * math.sin(rad)

    ticks = ""
    for pct in range(0, 101, 10):
        a = math.radians(180 - 180 * pct / 100)
        x1 = cx + (size * 0.42) * math.cos(a)
        y1 = cy - (size * 0.42) * math.sin(a)
        x2 = cx + (size * 0.47) * math.cos(a)
        y2 = cy - (size * 0.47) * math.sin(a)
        ticks += f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{TEXT_MUTED}" stroke-width="1.4"/>'

    thr_angle = math.radians(180 - 180 * DECISION_THRESHOLD)
    tx1 = cx + (size * 0.40) * math.cos(thr_angle)
    ty1 = cy - (size * 0.40) * math.sin(thr_angle)
    tx2 = cx + (size * 0.49) * math.cos(thr_angle)
    ty2 = cy - (size * 0.49) * math.sin(thr_angle)

    svg_lines = [
        f'<svg width="{size}" height="{size*0.62:.0f}" viewBox="0 0 {size} {size*0.62:.0f}">',
        f'<path d="M {cx - size*0.42} {cy} A {size*0.42} {size*0.42} 0 0 1 {cx + size*0.42} {cy}" '
        f'fill="none" stroke="{BG_PANEL_ALT}" stroke-width="14" stroke-linecap="round"/>',
        f'<path d="M {cx - size*0.42} {cy} A {size*0.42} {size*0.42} 0 0 1 {cx + size*0.42} {cy}" '
        f'fill="none" stroke="url(#gaugeGrad)" stroke-width="14" stroke-linecap="round" '
        f'stroke-dasharray="{math.pi*size*0.42:.1f}" stroke-dashoffset="{math.pi*size*0.42*(1-proba):.1f}"/>',
        '<defs><linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">'
        f'<stop offset="0%" stop-color="{GREEN_LIGHT}"/>'
        f'<stop offset="55%" stop-color="{GREEN_MID}"/>'
        f'<stop offset="100%" stop-color="{ALERT_RUST}"/>'
        '</linearGradient></defs>',
        ticks,
        f'<line x1="{tx1:.1f}" y1="{ty1:.1f}" x2="{tx2:.1f}" y2="{ty2:.1f}" stroke="{TEXT_PRIMARY}" stroke-width="2" stroke-dasharray="2,2"/>',
        f'<line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="{verdict_color}" stroke-width="3.2" stroke-linecap="round"/>',
        f'<circle cx="{cx}" cy="{cy}" r="6.5" fill="{verdict_color}"/>',
        f'<text x="{cx}" y="{cy - size*0.18:.0f}" text-anchor="middle" font-family="Space Grotesk, sans-serif" '
        f'font-size="{size*0.16:.0f}" font-weight="700" fill="{TEXT_PRIMARY}">{proba*100:.1f}%</text>',
        f'<text x="{cx}" y="{cy - size*0.02:.0f}" text-anchor="middle" font-family="IBM Plex Mono, monospace" '
        f'font-size="{size*0.052:.0f}" letter-spacing="1.5" fill="{TEXT_MUTED}">PELUANG HUJAN</text>',
        "</svg>",
    ]
    return "".join(svg_lines)


# ----------------------------------------------------------------------------
# Gemini assistant helpers
# ----------------------------------------------------------------------------
def get_gemini_key():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        try:
            key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            key = None
    if not key:
        key = st.session_state.get("manual_gemini_key")
    return key


def call_gemini(prompt, api_key):
    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
    )
    return response.text


def build_grounding_context():
    parts = [
        f"Model produksi: {model_info['best_model_name']}",
        f"Metrik model (kelas Yes/minoritas) - Accuracy: {model_info['metrics']['accuracy']:.3f}, "
        f"F1: {model_info['metrics']['f1']:.3f}, ROC-AUC: {model_info['metrics']['roc_auc']:.3f}",
        f"Threshold keputusan: {DECISION_THRESHOLD:.3f}",
    ]
    if FULL_EVAL:
        r = FULL_EVAL["report"]
        parts.append(
            "Evaluasi keseluruhan test set - "
            f"macro avg F1: {r['macro avg']['f1-score']:.3f}, "
            f"weighted avg F1: {r['weighted avg']['f1-score']:.3f}, "
            f"F1 kelas No: {r['No']['f1-score']:.3f}, F1 kelas Yes: {r['Yes']['f1-score']:.3f}."
        )
    if "realistic_benchmark_note" in model_info:
        parts.append(f"Catatan benchmark: {model_info['realistic_benchmark_note']}")

    conclusion_path = None
    for d in DATA_CANDIDATE_DIRS:
        p = os.path.join(d, "conclusion.txt")
        if os.path.exists(p):
            conclusion_path = p
            break
    if conclusion_path:
        with open(conclusion_path) as f:
            parts.append("Ringkasan kesimpulan notebook:\n" + f.read())

    last = st.session_state.get("last_prediction")
    if last:
        parts.append(
            f"Prediksi terakhir yang dijalankan pengguna: lokasi {last['lokasi']}, "
            f"tanggal {last['tanggal']}, probabilitas hujan besok {last['probabilitas_%']}%, "
            f"verdict {last['prediksi']}."
        )
    return "\n".join(parts)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    if LOGO_PATH:
        st.image(LOGO_PATH, width=150)
    else:
        st.markdown(
            f'<div style="display:flex; align-items:center; gap:0.6rem;">'
            f'{render_fallback_logo(38, on_dark=False)}'
            f'<span style="font-family:\'Space Grotesk\',sans-serif; font-weight:700; '
            f'font-size:1.05rem; color:{GREEN_DEEP};">AIGRA EON</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<span class="footnote">Taruh `aigra_logo.png` satu folder dengan '
            "streamlit_app.py untuk memakai logo resmi.</span>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='hairline'>", unsafe_allow_html=True)
    st.markdown(f'<div class="eyebrow">{icon("leaf")}Konfigurasi Stasiun</div>', unsafe_allow_html=True)
    st.markdown("#### Lokasi & Waktu")
    location = st.selectbox("Lokasi pengamatan", locations, index=locations.index("Sydney") if "Sydney" in locations else 0)
    obs_date = st.date_input("Tanggal pengamatan", datetime.date.today())
    season = month_to_season(obs_date.month)
    st.markdown(f'<span class="station-id">MUSIM TERDETEKSI: {season.upper()}</span>', unsafe_allow_html=True)

    st.markdown("<hr class='hairline'>", unsafe_allow_html=True)
    st.markdown(f'<div class="eyebrow">{icon("cpu")}Model Aktif</div>', unsafe_allow_html=True)
    st.markdown(f"**{model_info['best_model_name']}**")
    if FULL_EVAL:
        macro_f1 = FULL_EVAL["report"]["macro avg"]["f1-score"]
        st.markdown(
            f"<span class='footnote'>Macro F1 {macro_f1:.3f} &middot; "
            f"F1 (Yes) {model_info['metrics']['f1']:.3f} &middot; "
            f"ROC-AUC {model_info['metrics']['roc_auc']:.3f} &middot; "
            f"Acc {model_info['metrics']['accuracy']:.3f} &middot; "
            f"Threshold {DECISION_THRESHOLD:.2f}</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<span class='footnote'>F1 (Yes) {model_info['metrics']['f1']:.3f} &middot; "
            f"ROC-AUC {model_info['metrics']['roc_auc']:.3f} &middot; "
            f"Acc {model_info['metrics']['accuracy']:.3f} &middot; "
            f"Threshold {DECISION_THRESHOLD:.2f}</span>",
            unsafe_allow_html=True,
        )
    if "tuning" in model_info:
        st.markdown(
            f"<span class='footnote'>Dibandingkan dari {model_info.get('n_models_compared', '—')} model, "
            f"tuned via RandomizedSearchCV ({model_info['tuning']['tuned_base_model']})</span>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='hairline'>", unsafe_allow_html=True)
    st.markdown(f'<div class="eyebrow">{icon("layers")}Riwayat Sesi</div>', unsafe_allow_html=True)
    n_log = len(st.session_state.get("log", []))
    st.markdown(f"<span class='footnote'>{n_log} prediksi tercatat sesi ini</span>", unsafe_allow_html=True)
    if n_log and st.button("Bersihkan riwayat", width="stretch"):
        st.session_state["log"] = []
        st.rerun()

    st.markdown("<hr class='hairline'>", unsafe_allow_html=True)
    st.markdown(f'<div class="eyebrow">{icon("bot")}Asisten AI (Gemini)</div>', unsafe_allow_html=True)
    if not get_gemini_key():
        st.markdown(
            "<span class='footnote'>API key belum diset lewat secrets/env. "
            "Masukkan sementara di bawah (tidak disimpan permanen):</span>",
            unsafe_allow_html=True,
        )
        manual_key = st.text_input("GEMINI_API_KEY", type="password", key="manual_gemini_key_input")
        if manual_key:
            st.session_state["manual_gemini_key"] = manual_key
    else:
        st.markdown("<span class='footnote'>API key aktif.</span>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
if LOGO_PATH:
    logo_html = f'<div class="hero-logo-box"><img src="app/static/{os.path.basename(LOGO_PATH)}"/></div>'
    # Streamlit can't reliably embed a local file via a raw <img src> tag from
    # disk, so show the real logo with st.image right above the hero instead,
    # and keep the mark inside the hero as the compact fallback glyph.
    logo_html = f'<div class="hero-logo-box">{render_fallback_logo(44, on_dark=True)}</div>'
else:
    logo_html = f'<div class="hero-logo-box">{render_fallback_logo(44, on_dark=True)}</div>'

st.markdown(
    f"""<div class="hero">
<div class="hero-brand">
{logo_html}
<div>
<div class="eyebrow on-dark">Aigra Eon &middot; Smart Farming Weather Intelligence</div>
<p class="hero-title">Stasiun Prediksi Cuaca</p>
<p class="hero-sub">Estimasi peluang hujan esok hari dari kondisi atmosfer hari ini, membantu petani
merencanakan penyiraman, pemupukan, dan panen dengan lebih presisi (by Nazril Ravi Pratama).</p>
</div>
</div>
<div class="hero-chips">
<div class="hero-chip"><div class="metric-label">Lokasi</div><div class="metric-value">{location}</div></div>
<div class="hero-chip"><div class="metric-label">Tanggal</div><div class="metric-value">{obs_date.strftime('%d %b %Y')}</div></div>
<div class="hero-chip"><div class="metric-label">Musim</div><div class="metric-value">{season}</div></div>
</div>
</div>""",
    unsafe_allow_html=True,
)

if "log" not in st.session_state:
    st.session_state["log"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

tab_predict, tab_batch, tab_insight, tab_assistant, tab_about = st.tabs(
    ["Prediksi", "Prediksi Batch", "Wawasan Model", "Asisten AI", "Tentang"]
)

# ----------------------------------------------------------------------------
# TAB 1 - Single prediction
# ----------------------------------------------------------------------------
with tab_predict:
    left, right = st.columns([1.15, 1])

    with left:
        st.markdown(f'<div class="panel">{panel_title("Suhu (&deg;C)", "thermometer")}', unsafe_allow_html=True)
        colA, colB = st.columns(2)
        with colA:
            min_temp = st.slider("MinTemp", -10.0, 45.0, 15.0)
            temp_9am = st.slider("Temp9am", -10.0, 45.0, 18.0)
        with colB:
            max_temp = st.slider("MaxTemp", -5.0, 50.0, 25.0)
            temp_3pm = st.slider("Temp3pm", -5.0, 48.0, 23.0)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f'<div class="panel">{panel_title("Kelembapan &amp; Tekanan Udara", "droplet")}', unsafe_allow_html=True)
        colC, colD = st.columns(2)
        with colC:
            humidity_9am = st.slider("Humidity9am (%)", 0, 100, 60)
            pressure_9am = st.slider("Pressure9am (hPa)", 970.0, 1040.0, 1015.0)
        with colD:
            humidity_3pm = st.slider("Humidity3pm (%)", 0, 100, 45)
            pressure_3pm = st.slider("Pressure3pm (hPa)", 970.0, 1040.0, 1012.0)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f'<div class="panel">{panel_title("Angin", "wind")}', unsafe_allow_html=True)
        colE, colF, colG = st.columns(3)
        with colE:
            wind_gust_speed = st.slider("WindGustSpeed", 0.0, 140.0, 40.0)
            wind_gust_dir = st.selectbox("WindGustDir", directions, index=directions.index("W"))
        with colF:
            wind_speed_9am = st.slider("WindSpeed9am", 0.0, 100.0, 15.0)
            wind_dir_9am = st.selectbox("WindDir9am", directions, index=directions.index("W"))
        with colG:
            wind_speed_3pm = st.slider("WindSpeed3pm", 0.0, 100.0, 18.0)
            wind_dir_3pm = st.selectbox("WindDir3pm", directions, index=directions.index("WNW"))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f'<div class="panel">{panel_title("Curah Hujan Hari Ini", "rain")}', unsafe_allow_html=True)
        colH, colI = st.columns(2)
        with colH:
            rainfall = st.number_input("Rainfall (mm)", 0.0, 400.0, 0.0, step=0.5)
        with colI:
            rain_today = st.selectbox("RainToday", ["No", "Yes"])
        st.markdown("</div>", unsafe_allow_html=True)

        predict_clicked = st.button("Jalankan Prediksi", width="stretch")

    with right:
        if predict_clicked:
            raw_row = pd.DataFrame([{
                "Location": location, "Season": season,
                "MinTemp": min_temp, "MaxTemp": max_temp, "Rainfall": rainfall,
                "WindGustSpeed": wind_gust_speed, "WindGustDir": wind_gust_dir,
                "WindSpeed9am": wind_speed_9am, "WindSpeed3pm": wind_speed_3pm,
                "WindDir9am": wind_dir_9am, "WindDir3pm": wind_dir_3pm,
                "Humidity9am": humidity_9am, "Humidity3pm": humidity_3pm,
                "Pressure9am": pressure_9am, "Pressure3pm": pressure_3pm,
                "Temp9am": temp_9am, "Temp3pm": temp_3pm, "RainToday": rain_today,
            }])
            X_input, _ = engineer_features(raw_row)
            proba = float(predict_proba(X_input)[0])
            verdict = "Yes" if proba >= DECISION_THRESHOLD else "No"
            verdict_color = ALERT_RUST if verdict == "Yes" else GREEN_MID
            badge_class = "badge-yes" if verdict == "Yes" else "badge-no"
            badge_text = "Kemungkinan hujan" if verdict == "Yes" else "Kemungkinan cerah"

            st.markdown(
                f"""<div class="panel" style="text-align:center;">
{panel_title("Pembacaan Instrumen", "gauge")}
{render_gauge(proba, verdict_color)}
<div style="margin-top:0.6rem;"><span class="{badge_class}">{badge_text}</span></div>
<div class="footnote" style="margin-top:0.4rem;">Garis putus-putus = threshold keputusan model ({DECISION_THRESHOLD:.2f})</div>
</div>""",
                unsafe_allow_html=True,
            )

            reco_text = (
                "Tunda penyiraman terjadwal dan pastikan drainase lahan siap. "
                "Aktivitas panen sensitif air sebaiknya dipercepat sebelum hujan turun."
                if verdict == "Yes" else
                "Kondisi relatif kondusif untuk penyiraman terjadwal atau panen. "
                "Tetap pantau prakiraan resmi BMKG sebagai referensi tambahan."
            )
            reco_class = "rec-box alert" if verdict == "Yes" else "rec-box"
            st.markdown(f'<div class="{reco_class}"><b>Rekomendasi:</b> {reco_text}</div>', unsafe_allow_html=True)

            st.markdown(f'<div class="panel" style="margin-top:0.9rem;">{panel_title("Turunan Fitur Cuaca", "chart")}', unsafe_allow_html=True)
            derived = {
                "TempRange": max_temp - min_temp,
                "PressureDiff": pressure_9am - pressure_3pm,
                "HumidityDiff": humidity_9am - humidity_3pm,
                "WindSpeedDiff": wind_speed_3pm - wind_speed_9am,
            }
            st.markdown('<div class="metric-row">', unsafe_allow_html=True)
            for k, v in derived.items():
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">{k}</div>'
                    f'<div class="metric-value" style="font-size:1.05rem;">{v:.1f}</div></div>',
                    unsafe_allow_html=True,
                )
            st.markdown("</div></div>", unsafe_allow_html=True)

            entry = {
                "waktu": datetime.datetime.now().strftime("%H:%M:%S"),
                "lokasi": location,
                "tanggal": obs_date.strftime("%Y-%m-%d"),
                "probabilitas_%": round(proba * 100, 1),
                "prediksi": verdict,
            }
            st.session_state["log"].append(entry)
            st.session_state["last_prediction"] = entry
        else:
            st.markdown(
                f"""<div class="panel" style="text-align:center; padding:2rem 1rem 1.6rem;">
{panel_title("Menunggu Input", "gauge")}
<div class="gauge-idle">
<span class="footnote" style="margin-bottom:0.6rem;">0%</span>
</div>
<p class="footnote" style="margin-top:1rem;">
Atur parameter cuaca di sebelah kiri, lalu klik &ldquo;Jalankan Prediksi&rdquo;.</p>
</div>""",
                unsafe_allow_html=True,
            )

    if st.session_state["log"]:
        st.markdown("<hr class='hairline'>", unsafe_allow_html=True)
        st.markdown(f'<div class="eyebrow">{icon("layers")}Log Prediksi Sesi Ini</div>', unsafe_allow_html=True)
        log_df = pd.DataFrame(st.session_state["log"]).iloc[::-1].reset_index(drop=True)
        st.dataframe(log_df, width="stretch", hide_index=True)
        st.download_button(
            "Unduh log (.csv)",
            log_df.to_csv(index=False).encode("utf-8"),
            file_name="log_prediksi.csv",
            mime="text/csv",
        )

# ----------------------------------------------------------------------------
# TAB 2 - Batch prediction
# ----------------------------------------------------------------------------
with tab_batch:
    st.markdown(f'<div class="eyebrow">{icon("layers")}Prediksi Massal dari File CSV</div>', unsafe_allow_html=True)
    st.markdown(
        "Unggah beberapa baris data cuaca sekaligus (format menyerupai dataset weatherAUS) "
        "untuk mendapatkan prediksi RainTomorrow secara batch."
    )

    template_cols = [
        "Location", "Date", "MinTemp", "MaxTemp", "Rainfall", "WindGustSpeed", "WindGustDir",
        "WindSpeed9am", "WindSpeed3pm", "WindDir9am", "WindDir3pm", "Humidity9am", "Humidity3pm",
        "Pressure9am", "Pressure3pm", "Temp9am", "Temp3pm", "RainToday",
    ]
    template_row = [
        "Sydney", "2026-07-17", 15.0, 25.0, 0.0, 40.0, "W",
        15.0, 18.0, "W", "WNW", 60, 45, 1015.0, 1012.0, 18.0, 23.0, "No",
    ]
    template_df = pd.DataFrame([template_row], columns=template_cols)

    colT1, colT2 = st.columns([1, 1])
    with colT1:
        st.download_button(
            "Unduh template CSV",
            template_df.to_csv(index=False).encode("utf-8"),
            file_name="template_prediksi_batch.csv",
            mime="text/csv",
            width="stretch",
        )
    with colT2:
        uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_raw = pd.read_csv(uploaded_file)
            missing_cols = [c for c in template_cols if c not in batch_raw.columns]
            if missing_cols:
                st.error(f"Kolom berikut tidak ditemukan pada file: {missing_cols}")
            else:
                X_batch, _ = engineer_features(batch_raw)
                probs = predict_proba(X_batch)
                result_df = batch_raw.copy()
                result_df["Prob_RainTomorrow_%"] = (probs * 100).round(2)
                result_df["Prediksi"] = np.where(probs >= DECISION_THRESHOLD, "Yes", "No")

                st.markdown(f'<div class="panel">{panel_title("Ringkasan Batch", "chart")}', unsafe_allow_html=True)
                yes_pct = (result_df["Prediksi"] == "Yes").mean() * 100
                st.markdown('<div class="metric-row">', unsafe_allow_html=True)
                for label, value in [
                    ("Total baris", f"{len(result_df)}"),
                    ("Prediksi hujan", f"{yes_pct:.1f}%"),
                    ("Rata-rata prob.", f"{result_df['Prob_RainTomorrow_%'].mean():.1f}%"),
                ]:
                    st.markdown(
                        f'<div class="metric-card"><div class="metric-label">{label}</div>'
                        f'<div class="metric-value" style="font-size:1.1rem;">{value}</div></div>',
                        unsafe_allow_html=True,
                    )
                st.markdown("</div></div>", unsafe_allow_html=True)

                st.dataframe(result_df, width="stretch", hide_index=True)
                st.download_button(
                    "Unduh hasil prediksi (.csv)",
                    result_df.to_csv(index=False).encode("utf-8"),
                    file_name="hasil_prediksi_batch.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            st.error(f"Gagal memproses file: {exc}")

# ----------------------------------------------------------------------------
# TAB 3 - Model insight (full evaluation, not just minority-class F1)
# ----------------------------------------------------------------------------
with tab_insight:
    st.markdown(f'<div class="eyebrow">{icon("chart")}Performa &amp; Interpretasi Model</div>', unsafe_allow_html=True)

    if FULL_EVAL:
        rep = FULL_EVAL["report"]
        cm = FULL_EVAL["confusion_matrix"]

        st.markdown(f'<div class="panel">{panel_title("Evaluasi Keseluruhan (Test Set)", "shield")}', unsafe_allow_html=True)
        st.markdown('<div class="metric-row">', unsafe_allow_html=True)
        overall_cards = [
            ("Accuracy", f"{rep['accuracy']*100:.1f}%", False),
            ("Macro Avg F1", f"{rep['macro avg']['f1-score']*100:.1f}%", True),
            ("Weighted Avg F1", f"{rep['weighted avg']['f1-score']*100:.1f}%", True),
            ("ROC-AUC", f"{model_info['metrics']['roc_auc']*100:.1f}%", False),
        ]
        for label, value, is_highlight in overall_cards:
            css_class = "metric-card highlight" if is_highlight else "metric-card"
            st.markdown(
                f'<div class="{css_class}"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div></div>", unsafe_allow_html=True)

        st.markdown(f'<div class="panel">{panel_title("Rincian per Kelas", "grid")}', unsafe_allow_html=True)
        breakdown_rows = []
        for label in ["No", "Yes", "macro avg", "weighted avg"]:
            row = rep[label]
            breakdown_rows.append({
                "Kelas": label,
                "Precision": round(row["precision"], 4),
                "Recall": round(row["recall"], 4),
                "F1-score": round(row["f1-score"], 4),
                "Support": int(row["support"]),
            })
        breakdown_df = pd.DataFrame(breakdown_rows).set_index("Kelas")
        st.dataframe(breakdown_df, width="stretch")
        st.markdown(
            f'<p class="footnote">Dihitung ulang langsung dari test set tersimpan '
            f'({FULL_EVAL["n"]:,} baris) setiap kali aplikasi dijalankan, jadi angka ini selalu '
            f'konsisten dengan model &amp; threshold yang sedang aktif.</p>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            '<div class="rec-box" style="margin-bottom:1rem;">'
            "<b>Cara baca:</b> Macro/weighted avg F1 merangkum performa model di kedua kelas "
            "sekaligus (No &amp; Yes). F1 kelas &ldquo;Yes&rdquo; sendiri biasanya lebih rendah "
            "karena hujan adalah kelas minoritas itu wajar untuk dataset dengan class "
            "imbalance seperti ini, bukan tanda model bermasalah.</div>",
            unsafe_allow_html=True,
        )

        cm_fig, cm_ax = plt.subplots(figsize=(4.6, 4.0))
        cm_fig.patch.set_facecolor(BG_MAIN)
        cm_ax.set_facecolor(BG_MAIN)
        cm_ax.imshow(cm, cmap=plt.cm.Greens)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text_color = "white" if cm[i, j] > cm.max() / 2 else GREEN_DEEP
                cm_ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                           color=text_color, fontsize=12, fontweight="bold")
        cm_ax.set_xticks([0, 1]); cm_ax.set_yticks([0, 1])
        cm_ax.set_xticklabels(["No", "Yes"], color=TEXT_PRIMARY)
        cm_ax.set_yticklabels(["No", "Yes"], color=TEXT_PRIMARY)
        cm_ax.set_xlabel("Prediksi", color=TEXT_MUTED)
        cm_ax.set_ylabel("Aktual", color=TEXT_MUTED)
        cm_ax.set_title(f"Confusion Matrix - {model_info['best_model_name']}", color=GREEN_DEEP, fontsize=11)
        for spine in cm_ax.spines.values():
            spine.set_color(BORDER_MPL)
        st.markdown(f'<div class="panel">{panel_title("Confusion Matrix", "grid")}', unsafe_allow_html=True)
        st.pyplot(cm_fig)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning(
            "File `test_set_scaled.csv` tidak ditemukan di folder data, jadi evaluasi keseluruhan "
            "(macro/weighted avg) tidak bisa dihitung ulang. Menampilkan metrik kelas minoritas "
            "yang tersimpan di model_info.json saja."
        )
        m = model_info["metrics"]
        st.markdown('<div class="metric-row">', unsafe_allow_html=True)
        for label, key in [
            ("Accuracy", "accuracy"), ("Precision (Yes)", "precision"), ("Recall (Yes)", "recall"),
            ("F1 (Yes)", "f1"), ("ROC-AUC", "roc_auc"), ("Threshold", "threshold"),
        ]:
            val = m.get(key, DECISION_THRESHOLD)
            display_val = f"{val*100:.1f}%" if key != "threshold" else f"{val:.3f}"
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{display_val}</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    if "realistic_benchmark_note" in model_info:
        st.markdown(
            f'<div class="rec-box" style="margin-top:0.9rem;"><b>Catatan realistis:</b> '
            f'{model_info["realistic_benchmark_note"]}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if "tuning" in model_info:
        t = model_info["tuning"]
        st.markdown(f'<div class="panel">{panel_title("Hyperparameter Tuning", "cpu")}', unsafe_allow_html=True)
        st.markdown(
            f"Model dasar **{t['tuned_base_model']}** di-tuning dengan RandomizedSearchCV "
            f"(CV F1: **{t['cv_f1_score']:.4f}**)."
        )
        st.json(t["best_params"])
        st.markdown("</div>", unsafe_allow_html=True)

    if DATA_DIR:
        st.markdown(f'<div class="panel">{panel_title("Perbandingan Seluruh Model", "layers")}', unsafe_allow_html=True)
        comp_df = pd.read_csv(os.path.join(DATA_DIR, "model_comparison.csv"), index_col=0)
        st.dataframe(comp_df.style.format("{:.4f}"), width="stretch")

        fig, ax = plt.subplots(figsize=(9, 4.5))
        fig.patch.set_facecolor(BG_PANEL)
        ax.set_facecolor(BG_PANEL)
        comp_df[["f1", "roc_auc"]].sort_values("f1").plot(
            kind="barh", ax=ax, color=[GREEN_MID, GREEN_LIGHT], width=0.7
        )
        ax.set_xlabel("Skor", color=TEXT_MUTED)
        ax.tick_params(colors=TEXT_MUTED)
        ax.legend(facecolor=BG_PANEL, labelcolor=TEXT_PRIMARY, edgecolor=BORDER_MPL)
        for spine in ax.spines.values():
            spine.set_color(BORDER_MPL)
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    if FIG_DIR:
        st.markdown(f'<div class="panel">{panel_title("Feature Importance (dari Notebook)", "leaf")}', unsafe_allow_html=True)
        st.image(os.path.join(FIG_DIR, "13_feature_importance.png"), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info(
            "Gambar feature importance belum ditemukan. Salin folder `outputs/figures` dari hasil "
            "zip notebook ke direktori yang sama dengan streamlit_app.py untuk menampilkannya."
        )

# ----------------------------------------------------------------------------
# TAB 4 - Gemini AI Assistant
# ----------------------------------------------------------------------------
with tab_assistant:
    st.markdown(f'<div class="eyebrow">{icon("bot")}Asisten AI &middot; Gemini 2.5 Flash-Lite</div>', unsafe_allow_html=True)
    st.markdown(
        "Tanyakan apa pun seputar hasil prediksi, kualitas model, atau rekomendasi pertanian. "
        "Asisten ini dibekali konteks (RAG ringan) dari metrik model dan prediksi terakhir yang "
        "kamu jalankan di tab Prediksi."
    )

    api_key = get_gemini_key()
    if not api_key:
        st.warning(
            "API key Gemini belum tersedia. Set environment variable `GEMINI_API_KEY`, "
            "isi `st.secrets['GEMINI_API_KEY']` di `.streamlit/secrets.toml`, atau masukkan "
            "sementara lewat sidebar. **Jangan hardcode API key langsung di kode**, terutama "
            "jika repo ini akan diunggah ke GitHub publik."
        )
    else:
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_question = st.chat_input("Tulis pertanyaanmu di sini...")
        if user_question:
            st.session_state["chat_history"].append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            context = build_grounding_context()
            system_prompt = (
                "Kamu adalah asisten AI untuk aplikasi prediksi cuaca smart farming bernama "
                "'Stasiun Prediksi Cuaca'. Jawab dalam Bahasa Indonesia, singkat, jelas, dan "
                "praktis untuk petani. Gunakan konteks berikut jika relevan, dan jangan mengarang "
                "angka yang tidak ada di konteks:\n\n"
                f"{context}\n\nPertanyaan pengguna: {user_question}"
            )

            with st.chat_message("assistant"):
                with st.spinner("Berpikir..."):
                    try:
                        answer = call_gemini(system_prompt, api_key)
                    except Exception as exc:
                        answer = f"Maaf, terjadi kendala saat menghubungi Gemini API: {exc}"
                st.markdown(answer)
            st.session_state["chat_history"].append({"role": "assistant", "content": answer})

        if st.session_state["chat_history"] and st.button("Bersihkan percakapan"):
            st.session_state["chat_history"] = []
            st.rerun()

# ----------------------------------------------------------------------------
# TAB 5 - About
# ----------------------------------------------------------------------------
with tab_about:
    st.markdown(f'<div class="eyebrow">{icon("info")}Tentang Proyek</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="panel">
        <p>Aplikasi ini adalah antarmuka prediksi <b>RainTomorrow</b> yang dilatih pada dataset
        <i>Rain in Australia</i> (Kaggle), dibangun sebagai bagian dari coding test AI Engineer
        Intern PT AIGRA EON INDONESIA - Smart Farming.</p>
        <p>Sebanyak {model_info.get('n_models_compared', 'beberapa')} varian model dibandingkan
        (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, SVM,
        Neural Network/Deep Learning, model hasil tuning, Stacking Ensemble, dan Blended
        Ensemble), dengan penanganan class imbalance lewat class weighting serta optimisasi
        threshold keputusan pada validation set. Model produksi yang dipakai aplikasi ini:
        <b>{model_info['best_model_name']}</b> (threshold={DECISION_THRESHOLD:.3f}).</p>
        <p class="footnote">Catatan: prediksi ini bersifat estimatif berdasarkan data historis dan
        sebaiknya dikombinasikan dengan prakiraan resmi BMKG untuk keputusan pertanian penting.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="panel">{panel_title("Fitur yang Digunakan Model", "grid")}', unsafe_allow_html=True)
    st.code(", ".join(feature_columns), language=None)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        f'<div class="panel">{panel_title("Keamanan API Key", "shield")}'
        '<p style="font-size:0.88rem; line-height:1.5;">Aplikasi ini membaca API key Gemini dari '
        '<code>GEMINI_API_KEY</code> (environment variable) atau <code>st.secrets</code>, bukan dari '
        'kode sumber. Saat mengunggah repo ini ke GitHub publik, pastikan file '
        '<code>.streamlit/secrets.toml</code> masuk <code>.gitignore</code> agar API key tidak ikut '
        'ter-commit.</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="panel">{panel_title("Catatan Kompatibilitas Versi", "info")}'
        '<p style="font-size:0.88rem; line-height:1.5;">Model disimpan dengan scikit-learn/XGBoost '
        'versi tertentu saat training di notebook. Jika environment tempat Streamlit ini dijalankan '
        'memakai versi library yang berbeda, kamu akan melihat warning '
        '<code>InconsistentVersionWarning</code> di terminal (sudah diredam dari tampilan aplikasi, '
        'tapi tetap muncul di log). Untuk menghilangkannya secara permanen, samakan versi '
        '<code>scikit-learn</code> dan <code>xgboost</code> di <code>requirements.txt</code> deployment '
        'dengan versi yang dipakai saat notebook menyimpan <code>best_model.joblib</code>.</p></div>',
        unsafe_allow_html=True,
    )

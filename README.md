# Rain Prediction Intelligence System

**Live Demo / Web Dashboard:** https://nazril-aigra-technicaltest-webdaashboard.streamlit.app/

An end-to-end machine learning application for next-day rainfall prediction using the *Rain in Australia* (Weather Australia) dataset. This project was developed as an AI Engineer technical test and demonstrates the complete machine learning lifecycle, from exploratory data analysis and statistical testing, to leakage-free preprocessing, feature engineering, multi-model benchmarking, hyperparameter tuning, decision-threshold optimization, and deployment through an interactive Streamlit dashboard with an integrated Gemini AI assistant for prediction interpretation and smart farming recommendations.

The target variable is highly imbalanced (`RainTomorrow = Yes` is the minority class), so this project deliberately tracks **two F1 scores** throughout evaluation: the **minority-class F1** (`Yes`, the practically important one for a rain-alert system) and the **overall/weighted F1**, rather than optimizing for accuracy alone.

Main notebook: **`rain_prediction_colab_nazril.ipynb`**

---

## Problem Statement

PT AIGRA EON INDONESIA operates in smart farming, where knowing whether it will rain tomorrow helps farmers plan irrigation, fertilization, and harvest timing. This project builds a binary classifier for `RainTomorrow` (Yes/No) from same-day weather station readings, and packages it as a usable tool rather than a notebook-only exercise.

## Dataset

- **Source:** [Rain in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) (Kaggle)
- ~145,000 daily observations across 49 Australian weather stations, 23 raw columns
- Target: `RainTomorrow` (`Yes` / `No`), materially imbalanced (~78% No / ~22% Yes)

---

## Project Workflow

The project follows a complete, leakage-aware machine learning pipeline:

1. **Data Loading** reads `weatherAUS.csv`; auto-prompts upload when run on Colab.
2. **Exploratory Data Analysis (EDA)** dataset info, missing-value map, target distribution, numeric feature distributions, correlation heatmaps, categorical breakdowns (rain rate by location, `RainToday` vs `RainTomorrow`).
3. **Statistical Analysis**
   - Skewness analysis per numeric feature
   - Outlier detection via the IQR method (bounds and outlier % per column)
   - Hypothesis testing: Welch's t-test (numeric features vs. target) and Chi-square test (`RainToday` vs. `RainTomorrow`) to confirm associations are statistically significant, not incidental
4. **Feature Selection** evidence-based shortlist (humidity, pressure, `RainToday`, wind gust speed, temperature range, location, wind direction, season), with columns >35% missing (`Evaporation`, `Sunshine`, `Cloud9am`, `Cloud3pm`) dropped before imputation to avoid over-relying on heavily-imputed signals.
5. **Train / Validation / Test Split** stratified 64% / 16% / 20% split performed **before** any imputation or encoding statistic is computed, to prevent data leakage.
6. **Missing Value Imputation** numeric columns imputed by per-`Location` median (fallback: global median), categorical wind-direction columns by per-`Location` mode (fallback: global mode); all statistics fit on the **training split only** and re-applied to validation/test. `RainToday` (which had its own residual missing values) is re-derived deterministically from the imputed `Rainfall` column (`RainToday = 1` if `Rainfall > 1.0mm`) instead of being mode-imputed.
7. **Outlier Handling** IQR-based winsorization/clipping, with bounds computed on the training split only.
8. **Feature Engineering**
   - `TempRange` = `MaxTemp - MinTemp`
   - `PressureDiff` = `Pressure9am - Pressure3pm`
   - `HumidityDiff` = `Humidity9am - Humidity3pm`
   - `WindSpeedDiff` = `WindSpeed3pm - WindSpeed9am`
   - `Humidity_Pressure_Interaction` = `Humidity3pm * (1013 - Pressure3pm)`
   - `High_Humidity_Flag` (binary, `Humidity3pm >= 70`)
   - Cyclical sine/cosine encoding of `WindGustDir`, `WindDir9am`, `WindDir3pm` (compass degrees) instead of naive one-hot encoding
   - `Season` derived from `Date` (Australian seasons), one-hot encoded
   - `Location_freq`: frequency encoding of `Location`, fit on the training split only
9. **EDA After Preprocessing** re-checked distributions, correlation heatmap, and boxplots by target on the cleaned/engineered training data to confirm preprocessing didn't distort signal.
10. **Feature Scaling** `StandardScaler` fit on the training split only, applied to validation/test.
11. **Decision Threshold Optimization** for every model, the classification threshold is tuned on the **validation set** (maximizing minority-class F1 via the precision-recall curve) instead of the default 0.5, then applied to the untouched test set. This is the single largest lever for improving minority-class F1 on an imbalanced target, and is reported for every model alongside the default-threshold result for transparency.
12. **Machine Learning Model Development** see [Models](#machine-learning-models) below, all trained with `class_weight="balanced"` / `scale_pos_weight` / Keras `class_weight` to counter class imbalance.
13. **Hyperparameter Optimization** `RandomizedSearchCV` (3-fold stratified CV, F1-scored) on the strongest tree-based baseline.
14. **Ensemble Learning** a `StackingClassifier` (Random Forest + XGBoost + Logistic Regression base learners, logistic-regression meta-learner) and a manual probability-blended ensemble (Stacking + Neural Network).
15. **Model Comparison** all models ranked side-by-side on accuracy, precision, recall, minority-class F1 (default and tuned threshold), and ROC-AUC.
16. **Final Model Selection** the best-performing model that is still cleanly serializable with `joblib` (i.e., excluding the standalone Keras network and the manual blend, which would require bundling multiple artifacts) is promoted to production.
17. **Comprehensive Model Evaluation** classification report, confusion matrices, and ROC curves for the top models.
18. **Feature Importance Analysis** top contributing features for the selected production model.
19. **Model Serialization** production model, scaler, location frequency map, compass-degree map, feature column order, and the neural network all saved to `models/`, plus `model_info.json` documenting the selected model, its threshold, and its metrics.
20. **Deployment with Streamlit** the exact same feature-engineering logic from the notebook is re-implemented in `streamlit_app.py` so training and inference stay consistent.

---

## Machine Learning Models

The following models were developed and compared:

| Model | Notes |
|---|---|
| Logistic Regression | `class_weight="balanced"` baseline |
| Decision Tree | `class_weight="balanced"` |
| Random Forest | `class_weight="balanced"`, 300 trees |
| Gradient Boosting | `sample_weight`-balanced |
| XGBoost | `scale_pos_weight` for imbalance, early stopping on validation AUC |
| Support Vector Machine (RBF) | trained on a stratified subsample (≤20k rows) for tractable runtime; evaluated on full validation/test |
| Multi-Layer Perceptron (Neural Network) | Keras/TensorFlow, 3 hidden layers with batch norm + dropout, class-weighted, early-stopped on validation AUC |
| Tuned model (RandomizedSearchCV) | tuning applied to the strongest baseline of the run |
| Stacking Ensemble | RF + XGBoost + Logistic Regression → Logistic Regression meta-learner |
| Blended Ensemble | 50/50 probability blend of the Stacking Ensemble and the Neural Network |

The production model is selected automatically at the end of each notebook run as the **highest minority-class F1 (after threshold optimization) among the `joblib`-serializable candidates**. In this run, that was an **optimized XGBoost-based classifier** with a tuned decision threshold. Exact metrics for your run are in `data/model_comparison.csv` and `models/model_info.json` these are not hardcoded here since results are recomputed every time the notebook runs from scratch.

**On realistic expectations:** this is a widely-studied public dataset. Credible, leakage-free results on the minority class typically land around **F1 ≈ 0.62–0.70**, accuracy **≈ 84–87%**, and ROC-AUC **≈ 0.88–0.91**. This project targets the top of that legitimate range rather than an inflated score, and documents the full leakage-prevention methodology above so the numbers are reproducible and trustworthy.

---

## Dashboard Features

The Streamlit application provides:

- Rainfall prediction interface (single-input form with sliders for all weather readings)
- Batch prediction from an uploaded CSV, with a downloadable results file
- Interactive data visualizations and a rain-probability gauge with the model's decision threshold marked
- Model performance overview (accuracy, precision, recall, F1, ROC-AUC, threshold, tuning details)
- Feature importance visualization
- Production model inference (same feature pipeline as training, no train/serve skew)
- Google Gemini AI Assistant (Gemini 2.5 Flash-Lite) grounded with the model's metrics, the notebook's conclusion, and the user's latest prediction a lightweight RAG-style Q&A about the results
- Session prediction log with CSV export
- Smart farming recommendations tailored to the predicted verdict

---

## Project Structure

```text
.
├── data/
│   ├── test_set_scaled.csv
│   ├── model_comparison.csv
│   └── conclusion.txt
│
├── figures/
│
├── models/
│   ├── best_model.joblib
│   ├── scaler.joblib
│   ├── feature_columns.joblib
│   ├── compass_to_deg.joblib
│   ├── location_freq_map.joblib
│   ├── mlp_model.keras
│   └── model_info.json
│
├── .streamlit/
├── streamlit_app.py
├── rain_prediction_colab_nazril.ipynb   # main notebook
├── requirements.txt
├── secrets.toml.example
└── README.md
```

---

## Installation

Clone the repository
```bash
git clone https://github.com/zrll-shesh/Nazril-AIGRA.git
cd Nazril-AIGRA
```

Install dependencies
```bash
pip install -r requirements.txt
```

Create a local Streamlit secrets file (never commit the real key see [Security Notes](#security-notes))
```text
.streamlit/secrets.toml
```

Example
```toml
GEMINI_API_KEY = "YOUR_API_KEY"
```

Run the application
```bash
streamlit run streamlit_app.py
```

---

## Security Notes

- `GEMINI_API_KEY` is read from `st.secrets` / the `GEMINI_API_KEY` environment variable never hardcoded in source.
- `.streamlit/secrets.toml` is excluded via `.gitignore` and must **not** be committed; only `secrets.toml.example` (with a placeholder) is tracked.
- When deployed on Streamlit Community Cloud, the key is configured through **App → Settings → Secrets** on the dashboard, not through the repository.
- `xgboost` is version-pinned in `requirements.txt` to match the version used when the model was trained/serialized, to avoid deserialization warnings/errors across environments.

---

## Technologies

**Programming Language**
- Python

**Data Processing**
- Pandas, NumPy, SciPy

**Machine Learning**
- Scikit-learn, XGBoost, TensorFlow / Keras

**Visualization**
- Matplotlib, Seaborn

**Deployment**
- Streamlit (Community Cloud)

**AI Integration**
- Google Gemini API (`gemini-2.5-flash-lite`)

---

## Limitations & Future Work

- Tabular daily-weather features cap achievable minority-class F1 well below accuracy; incorporating satellite imagery, radar, or multi-day sequences would be needed to push further.
- `SVM` is trained on a subsample for runtime reasons; a full-data kernel SVM was not evaluated.
- Threshold optimization is fit on a single validation split; a k-fold threshold search would give a more stable estimate.
- Potential next steps: SMOTE/ADASYN comparison against class weighting, recursive feature elimination, probability calibration (`CalibratedClassifierCV`), and monitoring for model drift in production.

---

## Author

**Nazril Ravi Pratama**
AI Engineer Technical Test PT AIGRA EON INDONESIA

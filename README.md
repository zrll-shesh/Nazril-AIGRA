# Rain Prediction Intelligence System

**Live Demo/Web Dashboard:** https://nazril-aigra-technicaltest-webdaashboard.streamlit.app/

An end-to-end machine learning application for next-day rainfall prediction using the Weather Australia dataset. This project was developed as an AI Engineer technical test and demonstrates the complete machine learning lifecycle, from exploratory data analysis and feature engineering to model optimization and deployment through an interactive Streamlit dashboard for prediction interpretation and smart farming recommendations.

---

## Project Workflow

The project follows a complete end-to-end machine learning pipeline consisting of:

1. Data Loading
2. Exploratory Data Analysis (EDA)
3. Statistical Analysis
   - Skewness Analysis
   - Outlier Detection
   - Statistical Hypothesis Testing
4. Feature Selection
5. Train, Validation, and Test Split
6. Missing Value Imputation
7. Outlier Handling
8. Feature Engineering
9. Exploratory Data Analysis After Preprocessing
10. Feature Scaling
11. Decision Threshold Optimization
12. Machine Learning Model Development
13. Hyperparameter Optimization
14. Ensemble Learning
15. Model Comparison
16. Final Model Selection
17. Comprehensive Model Evaluation
18. Feature Importance Analysis
19. Model Serialization
20. Deployment with Streamlit

---

## Machine Learning Models

The following models were developed and compared throughout the experiment:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- Support Vector Machine (RBF)
- Multi-Layer Perceptron (Neural Network)
- Tuned XGBoost
- Stacking Ensemble
- Blended Ensemble (Stacking + Neural Network)

The production model is an optimized XGBoost classifier with decision threshold optimization for handling class imbalance.

---

## Dashboard Features

The Streamlit application provides:

- Rainfall prediction interface
- Interactive data visualizations
- Model performance overview
- Feature importance visualization
- Production model inference
- Google Gemini AI Assistant
- Prediction interpretation
- Smart farming recommendations

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
├── rain_prediction_colab_nazril.ipynb (file utama)
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

Create a local Streamlit secrets file

```text
.streamlit/secrets.toml
```

Example

```toml
GEMINI_API_KEY="YOUR_API_KEY"
```

Run the application

```bash
streamlit run streamlit_app.py
```

---

## Technologies

### Programming Language

- Python

### Data Processing

- Pandas
- NumPy
- SciPy

### Machine Learning

- Scikit-learn
- XGBoost
- TensorFlow / Keras

### Visualization

- Matplotlib
- Seaborn

### Deployment

- Streamlit

### AI Integration

- Google Gemini API

---

## Author

**Nazril Ravi Pratama**

AI Engineer Technical Test

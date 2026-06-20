# 🌍 Earthquake Magnitude Prediction — Regression Models

Predicting earthquake magnitude using machine learning regression models on a dataset of **782 earthquakes** recorded from 2001 to 2023 worldwide.

---

## 🎯 Problem Statement

Given earthquake sensor data (depth, alert level, significance score, continent, etc.), predict the **magnitude** of the earthquake (regression task).

- **Target variable**: `magnitude` (range: 6.5 – 9.1)
- **Dataset size**: 782 records, 19 features
- **Train/Test split**: 80/20

---

## 📊 Dataset Features

| Feature | Description |
|---------|-------------|
| `magnitude` | Earthquake magnitude (target) |
| `cdi` | Maximum reported intensity |
| `mmi` | Maximum estimated instrumental intensity |
| `alert` | Alert level: green / yellow / orange / red |
| `tsunami` | 1 = oceanic region, 0 = otherwise |
| `sig` | Significance score of the event |
| `nst` | Number of seismic stations used |
| `depth` | Depth of earthquake (km) |
| `latitude` / `longitude` | Geographic coordinates |
| `continent` | Continent of occurrence |

---

## 🔄 ML Pipeline

### 1. Exploratory Data Analysis (EDA)
- Missing value analysis (367 missing in `alert`, 576 in `continent`)
- Outlier detection using IQR
- Correlation heatmap — `sig` has strongest correlation with magnitude (0.52)
- Feature distribution plots & skewness/kurtosis analysis

### 2. Data Preprocessing
- **Numerical**: Median imputation → IQR outlier capping → RobustScaler
- **Categorical**: Mode imputation → OneHotEncoding (`alert`, `magType`, `continent`)
- Dropped low-correlation features: `tsunami`, `dmin`, `gap`, `latitude`, `longitude`

### 3. Model Comparison (LazyPredict — 42 models)

| Model | R² Score | RMSE |
|-------|----------|------|
| **RandomForestRegressor** | **0.62** | **0.23** |
| GradientBoostingRegressor | 0.59 | 0.24 |
| HistGradientBoostingRegressor | 0.54 | 0.25 |
| LGBMRegressor | 0.54 | 0.25 |
| XGBRegressor | 0.53 | 0.25 |
| LinearRegression | 0.41 | 0.29 |

### 4. Hyperparameter Tuning (RandomizedSearchCV)
Best model: **GradientBoostingRegressor**
```
Best parameters:
  n_estimators     : 50
  min_samples_split: 5
  max_depth        : 2
  criterion        : squared_error

Test R² Score: 0.60
```

---

## 🏆 Results

| Model | R² | MAE | MSE |
|-------|----|-----|-----|
| RandomForestRegressor | 0.64 | 0.13 | 0.05 |
| GradientBoostingRegressor (tuned) | 0.60 | 0.14 | 0.06 |
| LinearRegression | 0.42 | 0.21 | 0.08 |

**Best model**: `RandomForestRegressor` with R² = 0.64

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Missingno |
| EDA Report | ydata-profiling |
| ML Models | Scikit-learn, XGBoost, LightGBM |
| Model Comparison | LazyPredict |
| Tuning | RandomizedSearchCV |

---

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm lazypredict ydata-profiling missingno
```

Open `V2_Test_earthquake_regressor_week4_sub.ipynb` in Jupyter Notebook and run all cells.

---

## 📁 Project Structure

```
earthquake-magnitude-regression/
│
├── V2_Test_earthquake_regressor_week4_sub.ipynb   # Main notebook
├── earthquake_data.csv                             # Dataset (782 earthquakes)
└── README.md
```

---

## 👩‍💻 Author

**Angela** — Data Scientist | AI • ML • GenAI • RAG  
📍 Kuala Lumpur, Malaysia  
🔗 [GitHub](https://github.com/angelaadida)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

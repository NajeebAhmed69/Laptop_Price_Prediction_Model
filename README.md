```markdown
# 💻 Laptop Price Predictor

An end-to-end Machine Learning web application that predicts retail laptop prices based on technical hardware specifications. The project features automated text extraction, robust data cleaning, multi-model regression benchmarking with Scikit-Learn Pipelines, and an interactive UI built using Streamlit.

---

## 📌 Project Overview

Laptop pricing depends heavily on combinations of hardware configurations (Processor architecture, Memory size, Storage type, Display metrics, and Brand equity). 

This project implements:
- **Feature Extraction & Cleaning**: Parsing unstructured technical specs into discrete numeric and categorical signals.
- **Engineered Metrics**: Dynamic calculation of display pixel density (**PPI**) and separation of hybrid multi-drive storage configurations (SSD vs. HDD).
- **Log-Transformed Regression Modeling**: Mitigates target price skewness to stabilize variance and optimize regression metrics ($R^2$ and MAE).
- **Interactive UI**: A Streamlit dashboard allowing users to customize components and receive immediate price valuations.

---

## 📊 Dataset & Features

The model trains on a laptop specification dataset (`laptop_data.csv`) processed into the following predictive features:

| Feature | Type | Description |
| :--- | :--- | :--- |
| `Company` | Categorical | Manufacturer brand (e.g., Apple, Dell, Lenovo, HP, Asus) |
| `TypeName` | Categorical | Chassis form factor (Ultrabook, Notebook, Gaming, 2 in 1 Convertible, Workstation, Netbook) |
| `Ram` | Numeric | System RAM capacity in GB (e.g., 4, 8, 16, 32, 64) |
| `Weight` | Numeric | Laptop weight in kilograms (kg) |
| `TouchScreen` | Binary | Display touchscreen capability (`1` = Yes, `0` = No) |
| `Ips` | Binary | In-Plane Switching panel display (`1` = Yes, `0` = No) |
| `ppi` | Numeric | Pixels Per Inch: $\text{PPI} = \frac{\sqrt{X_{\text{res}}^2 + Y_{\text{res}}^2}}{\text{Inches}}$ |
| `Cpu Brand` | Categorical | Processor tier (Intel Core i3, i5, i7, Other Intel, AMD) |
| `HDD` | Numeric | Total Hard Disk Drive capacity in GB |
| `SSD` | Numeric | Total Solid-State Drive capacity in GB |
| `Gpu Brand` | Categorical | Graphics card manufacturer (Intel, Nvidia, AMD) |
| `os` | Categorical | Simplified OS category (`Windows`, `Mac`, `Others/NO OS/Linux`) |
| **`Price`** | **Target** | **Retail price (modeled internally using $\log(\text{Price})$)** |

---

## ⚙️ Data Preprocessing & Pipeline Architecture

All data transformations are packaged using Scikit-Learn's `ColumnTransformer` and `Pipeline` to ensure complete isolation and prevent data leakage:

1. **Storage Separation**:
   - Parses multi-drive configurations (e.g., `128GB SSD + 1TB HDD`) using regex extraction and vectorized operations to separate storage across dedicated `SSD` and `HDD` columns without chained assignment errors.
2. **Display Feature Engineering**:
   - Resolves screen resolution dimensions ($X \times Y$) and combines them with diagonal display size to derive screen sharpness (**PPI**).
3. **Categorical Encoding**:
   - Applies `OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')` to `['Company', 'TypeName', 'Cpu Brand', 'Gpu Brand', 'os']`.
4. **Target Variable Transformation**:
   - $y = \log(\text{Price})$ is used during model training to correct right-skewness. Predictions are transformed back to actual currency using $\exp(\hat{y})$.

---

## 🏆 Model Benchmarking

The training pipeline benchmarks 9 regression algorithms on an 85/15 train-test split:

- Linear Regression
- Ridge Regression ($\alpha = 10.0$)
- Lasso Regression ($\alpha = 0.001$)
- K-Nearest Neighbors Regressor ($k = 3$)
- Support Vector Regressor (RBF Kernel, $C = 10000$)
- Random Forest Regressor ($n = 100, \text{max\_depth} = 15$)
- Extra Trees Regressor ($n = 100, \text{max\_depth} = 15$)
- AdaBoost Regressor ($n = 500$)
- Gradient Boosting Regressor ($n = 500$)

The best-performing model based on $R^2$ score is selected and serialized as `pipeline.pkl` alongside the cleaned dataset `df.pkl`.

---

## 🗂️ Project Directory Structure

```text
├── laptop_data.csv              # Raw laptop dataset
├── laptop_price_prediction.py   # Data cleaning, pipeline setup, training & serialization
├── app.py                       # Streamlit web dashboard
├── df.pkl                       # Serialized cleaned DataFrame (reference categories)
├── pipeline.pkl                 # Serialized Scikit-Learn inference pipeline
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation

```

---

## 🚀 Quickstart Guide

### 1. Clone Repository & Set Up Virtual Environment

```bash
git clone [https://github.com/NajeebAhmed69/Laptop_Price_Prediction_Model.git](https://github.com/NajeebAhmed69/Laptop_Price_Prediction_Model.git)
cd laptop-price-predictor

python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt

```

### 2. Train and Export Artifacts

Run the training pipeline to clean the data, run model benchmarks, and export `df.pkl` and `pipeline.pkl`:

```bash
python laptop_price_prediction.py

```

### 3. Launch the Streamlit App

Run the interactive dashboard:

```bash
streamlit run app.py

```

Open your browser and navigate to `http://localhost:8501`.

---

## 📦 Requirements (`requirements.txt`)

```text
numpy
pandas
scikit-learn
streamlit

```

---

## 🛠️ Tech Stack

* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (`Pipeline`, `ColumnTransformer`, `OneHotEncoder`, Ensemble Regressors)
* **Serialization:** Pickle
* **Deployment:** Streamlit

```

```
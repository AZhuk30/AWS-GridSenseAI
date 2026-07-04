# GridSense AI: ERCOT Grid Stress Early Warning System

An end-to-end machine learning pipeline that predicts electrical grid stress events 24 hours in advance across the ERCOT (Texas) grid, using weather, energy demand, and renewable generation data.

---

## Overview

Grid operators need advance warning of stress events to schedule maintenance, arrange reserve capacity, and avoid outages. This project builds a binary classification model that ingests historical energy demand data (EIA), weather data (Open-Meteo), and renewable generation data (NREL) for the Austin, TX region, engineers 26 predictive features, and trains an XGBoost classifier to flag high-risk stress periods a full day ahead of time.

---

## Dataset

Combined multi-source dataset covering Austin, TX (30.3°N, -97.7°W), 2022-2024

- EIA - historical grid demand and load data
- Open-Meteo - historical weather data (temperature, wind, humidity, etc.)
- NREL - renewable generation and solar/wind resource data
- 26 engineered features combining weather, demand, and generation signals
- Class imbalance addressed with SMOTE, resampled to a 30/70 ratio

---

## Pipeline Architecture

The project is organized as five sequential notebooks:

| Notebook | Description |
|---|---|
| 01_eia_exploration.ipynb | EIA grid demand data ingestion and exploration |
| 02_openmeteo_exploration.ipynb | Open-Meteo weather data ingestion and exploration (Austin, TX) |
| 03_nrel_exploration.ipynb | NREL renewable generation data ingestion and exploration |
| 04_data_preparation.ipynb | Feature engineering, merging sources, scaling, SMOTE resampling |
| 05_model_training.ipynb | XGBoost model training, threshold tuning, and evaluation |

---

## Repository Structure

```
ADS-508-Final-Project/
│
├── notebooks/
│   ├── 01_eia_exploration.ipynb
│   ├── 02_openmeteo_exploration.ipynb
│   ├── 03_nrel_exploration.ipynb
│   ├── 04_data_preparation.ipynb
│   └── 05_model_training.ipynb
│
├── data/
│   └── (raw and processed EIA, Open-Meteo, and NREL data)
│
└── README.md
```
---

## Model

XGBoost binary classifier predicting grid stress events 24 hours ahead.

- 26 engineered features
- SMOTE resampling at a 30/70 class ratio
- Classification threshold tuned to 0.35
- Trained and evaluated using data stored in AWS S3 (gridsense-ai-data-team1)

### Results

| Metric | Score |
|---|---|
| AUC-ROC | 0.986 |
| Precision | 77.6% |
| Recall | 77.9% |

---

## How to Run

### Prerequisites

- Python 3.10+ (or Google Colab)
- AWS credentials with access to the gridsense-ai-data-team1 S3 bucket
- Standard data science stack: pandas, numpy, scikit-learn, xgboost, imbalanced-learn, boto3

### Steps

1. Run 01_eia_exploration.ipynb, 02_openmeteo_exploration.ipynb, and 03_nrel_exploration.ipynb to pull and explore each raw data source.
2. Run 04_data_preparation.ipynb to merge sources, engineer features, scale, and apply SMOTE resampling.
3. Run 05_model_training.ipynb to train the XGBoost model, tune the classification threshold, and generate evaluation metrics.

---

## Team

| Team Member | Role |
|---|---|
| Mark Villanueva | Feature engineering and data preparation |
| Alexander Zhuk | Model training and evaluation |
| Michael Ha | Data exploration and AWS/S3 pipeline |

---

## GitHub

https://github.com/mvillanueva00/ADS-508-Final-Project

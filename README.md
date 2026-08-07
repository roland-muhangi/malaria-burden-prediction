# Malaria Burden Prediction
### Sub-Saharan Africa — 47 Countries, 2000–2023

> **Certificate in Data Analysis and Engineering — Refactory Uganda**
> Personal ML Project | Roland Muhangi

---

## Project Overview

This project builds a machine learning model to predict whether a Sub-Saharan African country will face **high malaria burden** in a given year, using routine health indicators from the World Bank Open Data portal.

The goal is to support early resource allocation — positioning medicines, bed nets, and health workers **before** a high-burden year hits, rather than reacting after the fact.

---

## ML Task

| | |
|---|---|
| **Task type** | Supervised Binary Classification |
| **Target variable** | `high_burden` — 1 (high risk) or 0 (low risk) |
| **Target definition** | Country's malaria incidence exceeds the SSA regional median that year |
| **Models** | Random Forest + Neural Network (Keras), compared on same test set |

---

## Dataset

### Source
World Bank Open Data — WHO malaria indicators:
- **SH.MLR.INCD.P3** — Malaria incidence per 1,000 population at risk
- **SH.MLR.NETS.ZS** — Bed net coverage (% of under-5 sleeping under ITN)
- **SH.MLR.TRET.ZS** — Antimalarial treatment (% of under-5 with fever treated)

Publicly available at data.worldbank.org — no login required.

### Scale

| | |
|---|---|
| **Rows** | ~980 (one row per country per year) |
| **Features** | 11 engineered features |
| **Years covered** | 2000 – 2023 |
| **Countries** | 47 Sub-Saharan African countries including Uganda |

---

## Features

| Feature | Description |
|---|---|
| `incidence_per_1000` | Malaria cases per 1,000 population at risk ⭐ key feature |
| `incidence_lag1` | Last year's incidence |
| `incidence_lag2` | Incidence from two years ago |
| `incidence_3yr_avg` | 3-year rolling average incidence |
| `incidence_change` | Year-over-year change (positive = getting worse) |
| `itn_coverage_pct` | Bed net coverage percentage |
| `itn_lag1` | Last year's bed net coverage |
| `treatment_pct` | Antimalarial treatment coverage |
| `intervention_gap` | Bed net coverage minus treatment coverage |
| `east_africa` | 1 if Uganda or direct neighbour, 0 otherwise |
| `decade` | Decade indicator (2000, 2010, 2020) |

---

## Target Variable

```
high_burden = 1  if incidence_per_1000 > SSA regional median that year
high_burden = 0  if incidence_per_1000 ≤ SSA regional median that year
```

**Why SSA annual median?**
Malaria burden has improved dramatically since 2000. Using the annual regional median makes the threshold context-sensitive — it always asks: is this country doing worse than its regional peers this year?

---

## Repository Structure

```
malaria-burden-prediction/
│
├── ml_ready_malaria.csv          # Clean ML-ready dataset
├── clean_malaria.py              # Data cleaning & feature engineering script
├── Malaria_Submission_Slide.pptx # Project proposal slide
├── .gitignore                    # Excludes raw ZIP files
└── README.md                     # This file
```

---

## How to Reproduce

**Step 1 — Download the three World Bank ZIP files:**
```
https://api.worldbank.org/v2/en/indicator/SH.MLR.INCD.P3?downloadformat=csv  → save as incidence.zip
https://api.worldbank.org/v2/en/indicator/SH.MLR.NETS.ZS?downloadformat=csv   → save as bednets.zip
https://api.worldbank.org/v2/en/indicator/SH.MLR.TRET.ZS?downloadformat=csv   → save as treatment.zip
```

**Step 2 — Install dependency and run:**
```bash
pip3 install pandas
python3 clean_malaria.py
```

**Output:** `ml_ready_malaria.csv`

---

## Pipeline (Planned)

```
World Bank ZIP files
    ↓
clean_malaria.py        →  ml_ready_malaria.csv
    ↓
EDA & Visualisation     →  Incidence trends, Uganda vs peers, correlation heatmap
    ↓
Data Preparation        →  80/20 train/test split, StandardScaler
    ↓
Random Forest           →  Baseline classical ML model + feature importances
    ↓
Neural Network (Keras)  →  3-layer dense network, Dropout regularisation
    ↓
Evaluation              →  Accuracy, F1, ROC-AUC, Confusion Matrix
    ↓
Streamlit Demo          →  Input country indicators → get burden prediction
```

---

## Data Ethics

- All data is country-level aggregate — no individual patient data
- Source is fully public (World Bank Open Data)
- Raw ZIP files are excluded from this repository via .gitignore

---

*Roland Muhangi — Refactory Uganda · August 2026*

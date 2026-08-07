"""
Malaria Burden — Data Download & Cleaning Script
Project : Predicting High Malaria Burden in Sub-Saharan Africa
Source  : World Bank Open Data (WHO/GHO data via World Bank)
Output  : ml_ready_malaria.csv

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO DOWNLOAD YOUR 3 FILES (takes about 5 minutes)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OPTION A — Click the direct download links below in your browser.
Each one downloads a ZIP file automatically.

FILE 1 — Malaria incidence (cases per 1,000 population at risk)
  https://api.worldbank.org/v2/en/indicator/SH.MLR.INCD.P3?downloadformat=csv
  Save ZIP as: incidence.zip

FILE 2 — Bed net coverage (% of under-5 sleeping under ITN)
  https://api.worldbank.org/v2/en/indicator/SH.MLR.NETS.ZS?downloadformat=csv
  Save ZIP as: bednets.zip

FILE 3 — Antimalarial treatment (% of under-5 with fever treated)
  https://api.worldbank.org/v2/en/indicator/SH.MLR.TRET.ZS?downloadformat=csv
  Save ZIP as: treatment.zip

OPTION B — Visit the page and click the CSV button manually:
  https://data.worldbank.org/indicator/SH.MLR.INCD.P3
  https://data.worldbank.org/indicator/SH.MLR.NETS.ZS
  https://data.worldbank.org/indicator/SH.MLR.TRET.ZS

Put all 3 ZIP files in the SAME folder as this script then run:
    pip install pandas
    python clean_malaria.py
"""

import pandas as pd
import numpy as np
import zipfile
import os

print("=" * 60)
print("MALARIA BURDEN PREDICTION — DATA PIPELINE")
print("=" * 60)

SSA_COUNTRIES = [
    "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi",
    "Cameroon", "Central African Republic", "Chad", "Comoros",
    "Congo, Dem. Rep.", "Congo, Rep.", "Cote d'Ivoire", "Djibouti",
    "Equatorial Guinea", "Eritrea", "Eswatini", "Ethiopia", "Gabon",
    "Gambia, The", "Ghana", "Guinea", "Guinea-Bissau", "Kenya",
    "Lesotho", "Liberia", "Madagascar", "Malawi", "Mali",
    "Mauritania", "Mozambique", "Namibia", "Niger", "Nigeria",
    "Rwanda", "Senegal", "Sierra Leone", "Somalia", "South Africa",
    "South Sudan", "Sudan", "Tanzania", "Togo", "Uganda",
    "Zambia", "Zimbabwe"
]

def load_wb_zip(zip_filename, value_name):
    """
    Load a World Bank indicator ZIP file and return long-format dataframe.
    World Bank CSVs are wide format (one column per year).
    We melt them into long format (one row per country-year).
    """
    if not os.path.exists(zip_filename):
        raise FileNotFoundError(
            f"\n  File not found: {zip_filename}\n"
            f"  Download it from World Bank — see instructions above.\n"
        )
    print(f"  Reading {zip_filename}...")
    with zipfile.ZipFile(zip_filename, "r") as z:
        data_files = [f for f in z.namelist() if f.startswith("API_")]
        if not data_files:
            raise ValueError(f"No data file found in {zip_filename}")
        with z.open(data_files[0]) as f:
            df = pd.read_csv(f, skiprows=4)
    id_cols   = ["Country Name", "Country Code"]
    year_cols = [c for c in df.columns if str(c).isdigit()]
    df = df[id_cols + year_cols].copy()
    df_long = df.melt(
        id_vars=id_cols, value_vars=year_cols,
        var_name="year", value_name=value_name
    )
    df_long = df_long.rename(columns={"Country Name": "country", "Country Code": "iso_code"})
    df_long["year"]     = df_long["year"].astype(int)
    df_long[value_name] = pd.to_numeric(df_long[value_name], errors="coerce")
    df_long = df_long[df_long["country"].isin(SSA_COUNTRIES)]
    df_long = df_long[(df_long["year"] >= 2000) & (df_long["year"] <= 2023)]
    df_long = df_long.dropna(subset=[value_name])
    print(f"    {len(df_long):,} rows | {df_long['country'].nunique()} countries | {df_long['year'].min()}-{df_long['year'].max()}")
    return df_long

print("\n[1/6] Loading World Bank ZIP files...")
try:
    df_inc = load_wb_zip("incidence.zip",  "incidence_per_1000")
    df_itn = load_wb_zip("bednets.zip",    "itn_coverage_pct")
    df_trt = load_wb_zip("treatment.zip",  "treatment_pct")
except FileNotFoundError as e:
    print(e); exit(1)

print("\n[2/6] Merging on country + year...")
df = df_inc.merge(
    df_itn[["country","year","itn_coverage_pct"]], on=["country","year"], how="left"
).merge(
    df_trt[["country","year","treatment_pct"]], on=["country","year"], how="left"
)
print(f"  Merged: {df.shape[0]:,} rows x {df.shape[1]} columns")

print("\n[3/6] Engineering features...")
df = df.sort_values(["country","year"]).reset_index(drop=True)
df["incidence_lag1"]    = df.groupby("country")["incidence_per_1000"].shift(1)
df["incidence_lag2"]    = df.groupby("country")["incidence_per_1000"].shift(2)
df["itn_lag1"]          = df.groupby("country")["itn_coverage_pct"].shift(1)
df["incidence_change"]  = df["incidence_per_1000"] - df["incidence_lag1"]
df["incidence_3yr_avg"] = df.groupby("country")["incidence_per_1000"].transform(
    lambda x: x.shift(1).rolling(3, min_periods=2).mean()
)
df["intervention_gap"]  = df["itn_coverage_pct"] - df["treatment_pct"]
EAST_AFRICA = ["Uganda","Kenya","Tanzania","Rwanda","Burundi","South Sudan","Ethiopia","Congo, Dem. Rep."]
df["east_africa"]       = df["country"].isin(EAST_AFRICA).astype(int)
df["decade"]            = (df["year"] // 10) * 10

print("\n[4/6] Building target variable...")
df["ssa_median"]  = df.groupby("year")["incidence_per_1000"].transform("median")
df["high_burden"] = (df["incidence_per_1000"] > df["ssa_median"]).astype(int)
counts = df["high_burden"].value_counts()
pct    = df["high_burden"].value_counts(normalize=True) * 100
print(f"  High burden (1): {counts.get(1,0):,}  ({pct.get(1,0):.1f}%)")
print(f"  Low burden  (0): {counts.get(0,0):,}  ({pct.get(0,0):.1f}%)")

print("\n[5/6] Cleaning missing values...")
feature_cols = [
    "incidence_per_1000","incidence_lag1","incidence_lag2",
    "incidence_3yr_avg","incidence_change","itn_coverage_pct",
    "itn_lag1","treatment_pct","intervention_gap","east_africa","decade"
]
id_cols  = ["country","iso_code","year"]
ml_df    = df[id_cols + feature_cols + ["high_burden"]].copy()
before   = len(ml_df)
ml_df    = ml_df.dropna(subset=["high_burden","incidence_per_1000"])
for col in feature_cols:
    if ml_df[col].isnull().any():
        ml_df[col] = ml_df[col].fillna(ml_df[col].median())
print(f"  Rows before: {before:,}  after: {len(ml_df):,}  (dropped {before-len(ml_df):,})")

print("\n[6/6] Saving...")
ml_df.to_csv("ml_ready_malaria.csv", index=False)

print(f"\n{'='*60}")
print(f"DONE — ml_ready_malaria.csv")
print(f"{'='*60}")
print(f"Shape     : {ml_df.shape[0]:,} rows x {ml_df.shape[1]} columns")
print(f"Countries : {ml_df['country'].nunique()}")
print(f"Features  : {len(feature_cols)}")
print(f"\nUganda:")
uga = ml_df[ml_df["country"]=="Uganda"][["year","incidence_per_1000","itn_coverage_pct","high_burden"]]
print(uga.to_string(index=False))
print("\nDataset ready for EDA and modelling.")

"""Notebook section: library imports and shared plotting defaults."""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Notebook-style global settings shared by every downstream step.
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
pd.set_option("display.float_format", lambda value: f"{value:.4f}")

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("notebook", font_scale=1.1)
sns.set_palette("Set2")

# Analysis calendar constants used across feature and causal steps.
MERGE_DATE = pd.Timestamp("2022-09-15")
ANALYSIS_START = pd.Timestamp("2021-06-01")
ANALYSIS_END = pd.Timestamp("2025-08-31 23:59:59")
DATA_DIR = PROCESSED_DATA_DIR

# Step scripts expect string paths for os.path.join compatibility.
FIG_DIR = str(FIGURES_DIR)
TAB_DIR = str(TABLES_DIR)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 12,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
    }
)
print(f"Output directories ready:\nFigures: {FIG_DIR}\nTables: {TAB_DIR}")

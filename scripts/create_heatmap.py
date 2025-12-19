import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

# ==============================
# Configuration
# ==============================

RESULTS_DIR = Path(__file__).parent.parent / "results/category_results"
FIGURES_DIR = Path(__file__).parent.parent / "results/figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

ATTACK_CATEGORIES = {
    "flooding": "Flooding",
    "session_deletion": "Deletion",
    "session_modification": "Modification",
    "upf_pdn0_fault": "PDN0 Fault",
    "restoration_teid": "Restoration TEID",
}

NORMAL_COLUMN_NAME = "Normal"

# ==============================
# Model name → Acronym mapping
# ==============================

MODEL_NAME_MAP = {
    # Single detectors
    "ABOD": "ABOD",
    "COPOD": "COPOD",
    "ECOD": "ECOD",
    "FeatureBagging": "FeatureBagging",
    "GMM": "GMM",
    "HBOS": "HBOS",
    "IForest": "IForest",
    "INNE": "INNE",
    "KNN": "KNN",
    "LODA": "LODA",
    "LOF": "LOF",
    "PCA": "PCA",

    # Ensembles
    "Ensemble_SVC_C10_G10_HBOS_KNN_ABOD_INNE_PCA": "Ens-HKAIP",
    "Ensemble_SVC_C10_G10_HBOS_KNN_GMM_INNE_PCA": "Ens-HKGIP",
    "Ensemble_SVC_C10_G10_HBOS_KNN_LOF_INNE_PCA": "Ens-HKLIP",
    "Ensemble_SVC_C100_G100_HBOS_KNN_LOF_INNE_FeatureBagging": "Ens-HKLIF",
}

# Desired order in heatmap
MODEL_ORDER = [
    "ABOD",
    "COPOD",
    "ECOD",
    "FeatureBagging",
    "GMM",
    "HBOS",
    "IForest",
    "INNE",
    "KNN",
    "LODA",
    "LOF",
    "PCA",
    "Ens-HKAIP",
    "Ens-HKGIP",
    "Ens-HKLIP",
    "Ens-HKLIF",
]

# ==============================
# JSON Parsing
# ==============================

def load_model_results(json_path: Path):
    with open(json_path, "r") as f:
        data = json.load(f)

    raw_name = data.get(
        "model_name",
        json_path.stem,
    )

    model_name = MODEL_NAME_MAP.get(raw_name, raw_name)

    row = {}

    # ---- NORMAL ----
    normal = data.get("normal_condition", {})
    row[NORMAL_COLUMN_NAME] = normal.get("accuracy", float("nan"))

    # ---- ATTACKS ----
    attacks = data.get("attacks_by_category", {})
    for attack_key, display_name in ATTACK_CATEGORIES.items():
        row[display_name] = attacks.get(
            attack_key, {}
        ).get("detection_rate", float("nan"))

    return model_name, row


def build_heatmap_dataframe(results_dir: Path) -> pd.DataFrame:
    rows = {}

    for file in sorted(results_dir.iterdir()):
        if file.suffix != ".json":
            continue

        model_name, row = load_model_results(file)
        rows[model_name] = row

    df = pd.DataFrame.from_dict(rows, orient="index")

    # Column order
    ordered_cols = [NORMAL_COLUMN_NAME] + list(ATTACK_CATEGORIES.values())
    df = df[ordered_cols]

    # Row order (models)
    df = df.reindex(MODEL_ORDER)

    return df


# ==============================
# Heatmap Plot
# ==============================

def plot_heatmap(df: pd.DataFrame):
    plt.figure(figsize=(13, 6))

    ax = sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        linewidths=0.4,
        alpha=0.85,
        cbar_kws={"label": "Class-wise Accuracy / Detection Rate", "pad": 0.02},
    )
    ax.set_aspect("equal")

    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.labelpad = 10

    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="center")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.title("Class-wise Performance Heatmap (Normal + Attacks)", fontsize=18, pad=20)
    plt.xlabel("Class", fontsize=13)
    plt.ylabel("Model", fontsize=13)

    plt.tight_layout()

    out_path = FIGURES_DIR / "heatmap_detection_rate.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Heatmap saved to: {out_path}")


if __name__ == "__main__":
    df = build_heatmap_dataframe(RESULTS_DIR)

    print("\n=== Class-wise Performance Matrix (Acronyms) ===")
    print(df)

    plot_heatmap(df)

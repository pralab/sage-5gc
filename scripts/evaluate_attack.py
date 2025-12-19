import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

ROOT = Path(__file__).parent.parent
IN_RANDOM = ROOT / "results/random_attack"
IN_BLACKBOX = ROOT / "results/blackbox_attack"
OUT_DIR = ROOT / "results/evasion_score"
OUT_RANDOM = OUT_DIR / "random"
OUT_BLACKBOX = OUT_DIR / "blackbox"
OUT_DIFF = OUT_BLACKBOX / "differential"
OUT_EVOL = OUT_BLACKBOX / "evolution"

# Ensure output directories exist
OUT_RANDOM.mkdir(parents=True, exist_ok=True)
OUT_DIFF.mkdir(parents=True, exist_ok=True)
OUT_EVOL.mkdir(parents=True, exist_ok=True)

def process_json(input_file, model_key="evaded", alt_key="success"):
    with open(input_file, "r") as f:
        results = json.load(f)
    n_samples = len(results)
    # Handles both blackbox (evaded) and random (evaded or success)
    n_evaded = sum(
        1 for x in results.values()
        if x.get(model_key, x.get(alt_key, False))
    )
    evasion_rate = n_evaded / n_samples if n_samples else 0.0
    return n_samples, n_evaded, round(evasion_rate, 3)

# --- Process RANDOM ---
for file in IN_RANDOM.glob("*.json"):
    model = file.stem
    n_samples, n_evaded, evasion_rate = process_json(file, alt_key="success")
    out = {
        "model": model,
        "n_samples": n_samples,
        "n_evaded": n_evaded,
        "evasion_rate": evasion_rate,
    }
    with open(OUT_RANDOM / f"{model}.json", "w") as fout:
        json.dump(out, fout, indent=2)

# --- Process BLACKBOX (differential + evolution) ---
BLACKBOX_MAP = {
    "differentialevolution": OUT_DIFF,
    "evolutionstrategy": OUT_EVOL
}
for subdir, out_subdir in BLACKBOX_MAP.items():
    bb_dir = IN_BLACKBOX / subdir
    if not bb_dir.exists():
        continue
    for file in bb_dir.glob("*.json"):
        model = file.stem
        n_samples, n_evaded, evasion_rate = process_json(file)
        out = {
            "model": model,
            "n_samples": n_samples,
            "n_evaded": n_evaded,
            "evasion_rate": evasion_rate,
        }
        with open(out_subdir / f"{model}.json", "w") as fout:
            json.dump(out, fout, indent=2)

print(f"Evasion results saved in '{OUT_DIR}'")

LATEX_OUT = OUT_DIR / "evasion_rates_table.tex"

MODEL_GROUPS = {
    "Statistical": [
        "HBOS", "COPOD", "ECOD"
    ],
    "Density-based": [
        "KNN", "LOF", "IForest", "LODA", "INNE"
    ],
    "Geometric-based": [
        "PCA", "ABOD", "GMM"
    ],
    "Ensemble": [
        "FeatureBagging"
    ],
    "Combo": [
        "Ensemble_SVC_C10_G10_HBOS_KNN_ABOD_INNE_PCA",
        "Ensemble_SVC_C10_G10_HBOS_KNN_GMM_INNE_PCA",
        "Ensemble_SVC_C10_G10_HBOS_KNN_LOF_INNE_PCA",
        "Ensemble_SVC_C100_G100_HBOS_KNN_LOF_INNE_FeatureBagging",
    ]
}


def load_rate(json_path):
    if not json_path.exists():
        return None
    with open(json_path, "r") as f:
        return json.load(f).get("evasion_rate", None)


def fmt(x):
    return "--" if x is None else f"{x:.3f}"


latex_rows = []

for group, models in MODEL_GROUPS.items():
    first = True
    for model in models:
        r = load_rate(OUT_RANDOM / f"{model}.json")
        d = load_rate(OUT_DIFF / f"{model}.json")
        e = load_rate(OUT_EVOL / f"{model}.json")

        if first:
            latex_rows.append(
                rf"\multirow{{{len(models)}}}{{*}}{{\rotatebox{{90}}{{\textbf{{{group}}}}}}}"
                f" & \\texttt{{{model}}} & {fmt(r)} & {fmt(d)} & {fmt(e)} \\\\"
            )
            first = False
        else:
            latex_rows.append(
                f" & \\texttt{{{model}}} & {fmt(r)} & {fmt(d)} & {fmt(e)} \\\\"
            )
    latex_rows.append(r"\hline")

latex_body = "\n".join(latex_rows[:-1])  # remove last \hline

latex_table = rf"""
            \begin{{table}}[t]
            \centering
            \renewcommand{{\arraystretch}}{{1.25}}
            \setlength{{\tabcolsep}}{{6pt}}
            \begin{{tabular}}{{clccc}}
            \hline
            & \multirow{{2}}{{*}}{{\textbf{{Model}}}} &
            \textbf{{Random}} &
            \multicolumn{{2}}{{c}}{{\textbf{{Black-box}}}} \\
            \cline{{4-5}}
            & & &
            \textbf{{Diff. Evolution}} &
            \textbf{{Evol. Strategy}} \\
            \hline
            {latex_body}
            \end{{tabular}}
            \caption{{Evasion rates under random and adaptive black-box attack strategies.}}
            \label{{tab:evasion_rates}}
            \end{{table}}
            """.strip()

with open(LATEX_OUT, "w") as f:
    f.write(latex_table)

print(f"LaTeX table saved to '{LATEX_OUT}'")
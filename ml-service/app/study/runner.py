"""Run the full flight-fare study (report Units 1–5 + extended analysis) and save every artefact.

    python -m app.study.runner [--data PATH] [--skip-extended]      (from ml-service/)

Outputs
    <project>/report_assets/figures/*.png   report figures (same numbering as the report)
    <project>/report_assets/code/*.png      code screenshots for Appendix A
    <project>/report_assets/results.json    every number behind the tables and figures
    <project>/report_assets/summary.md      plain-language summary of the findings
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from ..config import PROJECT_ROOT
from ..utils import to_jsonable
from . import codeshots, data, figures, unit3, unit4, unit5, unit12

# STUDY_ASSETS_DIR overrides the output folder (used by tests so stand-in data never lands in report_assets).
ASSETS = Path(os.environ.get("STUDY_ASSETS_DIR") or PROJECT_ROOT / "report_assets")
RESULTS_FILE = ASSETS / "results.json"


def run_study(data_path: str | None = None, extended: bool = True, progress=print, assets: Path = ASSETS) -> dict:
    t0 = time.perf_counter()
    path = data.find_dataset(data_path)
    progress(f"Loading {path.name}")
    df = data.load(path)
    r: dict = {"dataset": {"file": str(path), "rows": int(len(df))}}

    progress("Unit 1: summary statistics and EDA")
    r["unit1"] = {"describe": unit12.describe_dataset(df), "eda": unit12.eda_data(df)}

    progress("Unit 2: regression with label-encoded features")
    r["unit2"] = {"label_encoded": unit12.label_encoded_regression(df)}
    gc.collect()

    dedup, X_train, X_test, y_train, y_test = unit12.common_split(df)
    progress(f"Unit 2: common pipeline ({len(dedup):,} de-duplicated rows)")
    r["unit2"]["one_hot"] = unit12.one_hot_regression(X_train, X_test, y_train, y_test)

    progress("Unit 3: exploratory K-means")
    r["unit3"] = {"exploratory": unit3.exploratory_kmeans(df), "cluster_regression": unit3.cluster_label_regression(df)}
    progress("Unit 3: K-means pipeline (K sweep — the slowest step)")
    r["unit3"]["pipeline"] = unit3.kmeans_pipeline(dedup, progress)
    gc.collect()

    progress("Unit 4: sequences, Markov chain and HMM")
    seq = unit4.build_sequences(df)
    r["unit4"] = {"sequences": unit4.sequence_summary(seq), "states": unit4.fare_states(seq),
                  "transition": unit4.transition_matrix(seq), "hmm": unit4.hmm_demonstration()}
    if extended:
        progress("Unit 4 (extended): Baum–Welch on fare sequences")
        r["unit4"]["learned_hmm"] = unit4.learned_hmm(seq)
    del seq
    gc.collect()

    progress("Unit 5: decision tree, random forest, XGBoost")
    r["unit5"] = {**unit5.train_tree_models(X_train, X_test, y_train, y_test), "n_rows": int(len(dedup))}
    r["comparison"] = sorted(r["unit5"]["results"] + r["unit2"]["one_hot"]["results"], key=lambda m: -m["r2"])

    if extended:
        progress("Extended: cross-validation, log target, clustering comparison")
        r["extended"] = {
            "cross_validation": unit5.cross_validation(dedup),
            "log_target": unit5.log_target(X_train, X_test, y_train, y_test),
            "clustering": unit5.clustering_comparison(dedup),
        }

    r["summary"] = summarize(r)
    r["generated_at"] = datetime.now(timezone.utc).isoformat()
    r["duration_seconds"] = time.perf_counter() - t0
    r = to_jsonable(r)

    progress("Rendering figures")
    r["figures"] = figures.render_all(r, assets / "figures")
    progress("Rendering code screenshots")
    r["code_images"] = codeshots.render_all(assets / "code")
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "results.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
    (assets / "summary.md").write_text(summary_markdown(r), encoding="utf-8")
    progress(f"Done in {r['duration_seconds']:.0f}s — saved to {assets}")
    return r


# ----------------------------------------------------------------- narrative summary
def _pct(x):
    return f"{x * 100:.2f}%"


def summarize(r: dict) -> list[dict]:
    """Plain-language findings. Every interpretation is chosen from the computed numbers."""
    d = r["unit1"]["describe"]
    fare = d["numeric_stats"]["Fare"]
    le = {m["model"]: m for m in r["unit2"]["label_encoded"]["results"]}
    oh = {m["model"]: m for m in r["unit2"]["one_hot"]["results"]}
    u3 = r["unit3"]["pipeline"]
    tm = r["unit4"]["transition"]["matrix"]
    hmm = r["unit4"]["hmm"]
    best, worst = r["comparison"][0], r["comparison"][-1]
    rep = u3["classification_report"]
    alone = list(d["variance_explained_alone"].items())

    skew = d["fare_skewness"]
    shape = ("strongly right-skewed" if skew > 1 else "moderately right-skewed" if skew > 0.5
             else "roughly symmetric" if skew > -0.5 else "left-skewed")
    lin_le, lin_oh = le["Linear regression"]["r2"], oh["Linear regression"]["r2"]
    ridge_gap = abs(oh["Ridge regression (alpha = 1)"]["r2"] - lin_oh)
    sil = u3["test_silhouette"]
    sil_text = ("low, so the clusters overlap" if sil < 0.25 else
                "moderate, so the clusters are partly separated" if sil < 0.5 else "high, so the clusters are well separated")
    diag = [tm[i][i] for i in range(3)]
    persistence = ("persistent" if min(diag) > 0.5 else "mixed: some states persist and others change often"
                   if max(diag) > 0.5 else "volatile")
    tree_names = {"Decision tree (depth 15)", "Random forest", "XGBoost"}
    best_is_tree = best["model"] in tree_names
    linear_best = max((m for m in r["comparison"] if m["model"] not in tree_names), key=lambda m: m["r2"])

    sections = [
        {"title": "What the data looks like", "points": [
            f"{d['rows']:,} flight listings with {d['columns']} attributes; {d['missing_values']} missing values and "
            f"{d['duplicate_rows']:,} exact duplicate rows.",
            f"Fares range from ₹{fare['min']:,.0f} to ₹{fare['max']:,.0f} with mean ₹{fare['mean']:,.0f} and median "
            f"₹{fare['median']:,.0f}; skewness {skew:.2f}, so the distribution is {shape} (Figure 2.1).",
            f"On its own, {alone[0][0]} explains {_pct(alone[0][1])} of fare variance and {alone[1][0]} "
            f"{_pct(alone[1][1])}, while Duration_in_hours explains {_pct(d['variance_explained_alone']['Duration_in_hours'])} "
            f"and Days_left {_pct(d['variance_explained_alone']['Days_left'])} (Figures 2.2 – 2.4).",
        ]},
        {"title": "Units 1–2: regression baselines", "points": [
            f"With label-encoded categories, linear regression explains {_pct(lin_le)} of fare variance and the "
            f"degree-2 polynomial {_pct(le['Polynomial regression (degree 2)']['r2'])}.",
            f"After removing duplicates and one-hot encoding, linear regression reaches {_pct(lin_oh)} "
            f"(MAE ₹{oh['Linear regression']['mae']:,.0f}), a change of {(lin_oh - lin_le) * 100:+.2f} points. "
            + ("Ridge (α = 1) is practically identical, so regularisation has little effect at this sample size."
               if ridge_gap < 0.005 else f"Ridge (α = 1) differs by {ridge_gap * 100:.2f} points.") + " (Figure 2.6)",
        ]},
        {"title": "Unit 3: K-means clustering", "points": [
            f"Clustering used {u3['n_features']:,} one-hot and scaled features; validation accuracy peaked at K = "
            f"{u3['best_k']} (Figure 3.4).",
            f"On the {u3['n_test']:,} test flights, mapping clusters to the majority fare class gives "
            f"{_pct(u3['test_accuracy'])} accuracy (about 50% by chance), with recall "
            f"{rep['Higher Fare']['recall']:.2f} for higher fares and {rep['Lower Fare']['recall']:.2f} for lower fares.",
            f"The silhouette score is {sil:.4f} — {sil_text} (Figure 3.8).",
        ]},
        {"title": "Unit 4: fare-state sequences", "points": [
            f"{r['unit4']['sequences']['total_sequences']:,} flight sequences were built; "
            f"{r['unit4']['sequences']['multiple_observations']:,} have at least two observations and contribute transitions.",
            f"Fare states are {persistence}: P(Low→Low) = {diag[0]:.4f}, P(Medium→Medium) = {diag[1]:.4f}, "
            f"P(High→High) = {diag[2]:.4f} (Figure 3.12).",
            f"On the report's hand-set HMM the forward and backward likelihoods are {hmm['forward_probability']:.7f} and "
            f"{hmm['backward_probability']:.7f}"
            + (" (identical, as they must be)" if abs(hmm['forward_probability'] - hmm['backward_probability']) < 1e-12 else "")
            + f"; scaled log-likelihood {hmm['scaled_log_likelihood']:.4f}; Viterbi path {hmm['viterbi_path']}.",
        ]},
        {"title": "Unit 5: trees and ensembles", "points": [
            f"Best model: {best['model']} with R² {best['r2']:.4f} and MAE ₹{best['mae']:,.0f}; the weakest is "
            f"{worst['model']} at R² {worst['r2']:.4f} (Figure 4.3).",
            (f"The best tree-based model beats the best linear model ({linear_best['model']}, R² "
             f"{linear_best['r2']:.4f}) by {(best['r2'] - linear_best['r2']) * 100:.2f} points."
             if best_is_tree else
             f"A linear model ({best['model']}) is best here, so the extra flexibility of trees did not help on this data."),
        ]},
    ]
    if "learned_hmm" in r["unit4"]:
        h = r["unit4"]["learned_hmm"]
        imp = r["unit5"]["feature_importance"]["XGBoost"]
        gaps = sorted(((m["train_r2"] - m["r2"], m["model"]) for m in r["comparison"] if "train_r2" in m), reverse=True)
        better = h["held_out_log_likelihood_per_obs"] > h["report_hmm_log_likelihood_per_obs"]
        points = [
            f"Most important attributes for XGBoost: {', '.join(x['attribute'] for x in imp[:3])} (Figure E.1).",
            f"Largest train–test R² gap: {gaps[0][1]} ({gaps[0][0] * 100:.2f} points), the strongest sign of "
            "over-fitting (Figure E.2).",
            f"Baum–Welch learned the HMM from {h['train_sequences']:,} sequences; on held-out sequences it scores "
            f"{h['held_out_log_likelihood_per_obs']:.4f} log-likelihood per observation vs "
            f"{h['report_hmm_log_likelihood_per_obs']:.4f} for the hand-set parameters, i.e. it fits "
            + ("better" if better else "worse") + " (Figure E.5).",
        ]
        if "extended" in r:
            cv = max(r["extended"]["cross_validation"], key=lambda m: m["mean_r2"])
            cl = max(r["extended"]["clustering"], key=lambda m: m["accuracy"])
            points += [f"5-fold cross-validation: best mean R² {cv['mean_r2']:.4f} ± {cv['std_r2']:.4f} for "
                       f"{cv['model']} (Figure E.3).",
                       f"Clustering comparison on a sample: {cl['method']} gives the highest accuracy "
                       f"({_pct(cl['accuracy'])}) (Figure E.4)."]
        sections.append({"title": "Extended analysis (report future scope)", "points": points})
    return sections


def summary_markdown(r: dict) -> str:
    lines = [f"# Flight-fare study — results summary", "",
             f"Dataset: `{Path(r['dataset']['file']).name}` · generated {r['generated_at']}", ""]
    for s in r["summary"]:
        lines += [f"## {s['title']}", ""] + [f"- {p}" for p in s["points"]] + [""]
    lines += ["## Model comparison (common test set)", "", "| Rank | Model | R² | MAE (Rs.) | RMSE (Rs.) | Train time (s) |",
              "|---|---|---|---|---|---|"]
    for i, m in enumerate(r["comparison"], 1):
        lines.append(f"| {i} | {m['model']} | {m['r2']:.4f} | {m['mae']:,.2f} | {m['rmse']:,.2f} | {m['train_seconds']:.2f} |")
    lines += ["", "## Figures", ""] + [f"- Figure {f['number']}: {f['title']} — `figures/{f['file']}`" for f in r["figures"]]
    lines += ["", "## Code screenshots", ""] + [f"- {c['chapter']}: {c['caption']} — `code/{c['file']}`" for c in r["code_images"]]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------- background job (used by the API)
class StudyJob:
    """Runs one study at a time in a background thread and records its progress."""

    def __init__(self, run_fn, find_data):
        self._lock = threading.Lock()
        self._run_fn = run_fn
        self._find_data = find_data
        self.state = {"status": "idle", "log": [], "started_at": None, "finished_at": None, "error": None}

    def start(self, extended: bool = True) -> dict:
        with self._lock:
            if self.state["status"] == "running":
                return self.state
            self._find_data()  # fail fast with a clear message
            self.state = {"status": "running", "log": [], "started_at": datetime.now(timezone.utc).isoformat(),
                          "finished_at": None, "error": None}
        threading.Thread(target=self._run, args=(extended,), daemon=True).start()
        return self.state

    def _log(self, msg: str) -> None:
        self.state["log"].append({"t": datetime.now(timezone.utc).isoformat(), "message": msg})

    def _run(self, extended):
        try:
            self._run_fn(extended=extended, progress=self._log)
            self.state["status"] = "done"
        except Exception as exc:  # report to the UI instead of dying silently
            self.state["status"] = "error"
            self.state["error"] = f"{exc.__class__.__name__}: {exc}"
            self._log(traceback.format_exc(limit=3))
        finally:
            self.state["finished_at"] = datetime.now(timezone.utc).isoformat()


JOB = StudyJob(lambda extended, progress: run_study(None, extended, progress), data.find_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the flight-fare study and save report assets.")
    parser.add_argument("--data", help="Path to Cleaned_dataset.csv or .zip (default: ml-service/data/)")
    parser.add_argument("--out", help="Output folder (default: <project>/report_assets)")
    parser.add_argument("--skip-extended", action="store_true", help="Only reproduce the report (faster)")
    args = parser.parse_args()
    run_study(args.data, extended=not args.skip_extended, assets=Path(args.out) if args.out else ASSETS,
              progress=lambda m: print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True))

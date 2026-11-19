"""Run the network-attack study and render its figures, code screenshots and summary.

    python -m app.study.attack_report [--skip-extended] [--out FOLDER]      (from ml-service/)

Saves to <project>/report_assets/attacks/{figures,code}/*.png, results.json and summary.md.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from pygments import highlight  # noqa: E402
from pygments.formatters import ImageFormatter  # noqa: E402
from pygments.lexers import PythonLexer  # noqa: E402
from sklearn.tree import plot_tree  # noqa: E402

from ..config import PROJECT_ROOT  # noqa: E402
from ..ml import hmm  # noqa: E402
from ..utils import to_jsonable  # noqa: E402
from . import attacks as A  # noqa: E402
from .codeshots import _source  # noqa: E402
from .figures import BLUE, _box, _save, _workflow  # noqa: E402
from .runner import StudyJob  # noqa: E402

ASSETS = Path(os.environ.get("ATTACK_ASSETS_DIR") or PROJECT_ROOT / "report_assets" / "attacks")
COLORS = {"normal": "#2ca02c", "DoS": "#d62728", "Probe": "#ff7f0e", "R2L": "#9467bd", "U2R": "#8c564b"}
FIGURES: list[dict] = []


def fig(number, title, chapter):
    def wrap(fn):
        FIGURES.append({"file": fn.__name__ + ".png", "number": number, "title": title, "chapter": chapter, "fn": fn})
        return fn
    return wrap


# ----------------------------------------------------------------- chapter 2: Units 1 and 2
@fig("2.1", "Connections per category (log scale)", "Units 1 and 2")
def fig_2_1_category_distribution(r, out):
    c = r["unit1"]["describe"]["category_counts"]
    f, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(list(c), list(c.values()), color=[COLORS[k] for k in c])
    ax.bar_label(bars, labels=[f"{v:,}" for v in c.values()], fontsize=9)
    ax.set(yscale="log", title="Distribution of Connection Categories", xlabel="Category", ylabel="Connections (log)")
    _save(f, out, "fig_2_1_category_distribution")


@fig("2.2", "The 20 most frequent attack types", "Units 1 and 2")
def fig_2_2_attack_types(r, out):
    c = {k: v for k, v in r["unit1"]["describe"]["attack_type_counts"].items() if k != "normal"}
    items = sorted(c.items(), key=lambda kv: kv[1])[-20:]
    f, ax = plt.subplots(figsize=(9, 6))
    ax.barh([k for k, _ in items], [v for _, v in items], color=BLUE)
    ax.set(xscale="log", title="Attack Types (de-duplicated connections)", xlabel="Connections (log)")
    _save(f, out, "fig_2_2_attack_types")


@fig("2.3", "Protocol type by connection category", "Units 1 and 2")
def fig_2_3_protocol_by_category(r, out):
    t = r["unit1"]["describe"]["protocol_by_category"]
    protos = list(t)
    f, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(protos))
    for c in A.CATEGORIES:
        vals = np.array([t[p][c] for p in protos])
        ax.bar(protos, vals, bottom=bottom, label=c, color=COLORS[c])
        bottom += vals
    ax.set(title="Protocol Type by Category", xlabel="Protocol", ylabel="Connections")
    ax.legend()
    _save(f, out, "fig_2_3_protocol_by_category")


@fig("2.4", "Distribution of bytes sent: normal vs attack", "Units 1 and 2")
def fig_2_4_src_bytes_hist(r, out):
    h = r["unit1"]["distributions"]["log_src_bytes_hist"]
    e = np.array(h["edges"])
    f, ax = plt.subplots(figsize=(8, 5))
    for name, color in (("normal", COLORS["normal"]), ("attack", COLORS["DoS"])):
        ax.bar(e[:-1], h[name], width=np.diff(e), align="edge", alpha=0.55, label=name, color=color)
    ax.set(title="log10(src_bytes + 1) by Class", xlabel="log10(bytes sent + 1)", ylabel="Connections")
    ax.legend()
    _save(f, out, "fig_2_4_src_bytes_hist")


@fig("2.5", "Connections to the same host in the last 2 s (count) by category", "Units 1 and 2")
def fig_2_5_count_box(r, out):
    b = r["unit1"]["distributions"]["count_box"]
    f, ax = plt.subplots(figsize=(9, 5))
    _box(ax, [b[c] for c in A.CATEGORIES], A.CATEGORIES)
    ax.set(title="Connection Count by Category", xlabel="Category", ylabel="count")
    ax.grid(True, alpha=0.4)
    _save(f, out, "fig_2_5_count_box")


@fig("2.6", "Correlation of key traffic features", "Units 1 and 2")
def fig_2_6_correlation(r, out):
    c = r["unit1"]["describe"]["correlation"]
    m = np.array(c["matrix"])
    f, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(c["features"])), c["features"], rotation=60, ha="right")
    ax.set_yticks(range(len(c["features"])), c["features"])
    for i in range(len(m)):
        for j in range(len(m)):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if abs(m[i, j]) > 0.6 else "black")
    f.colorbar(im, ax=ax)
    ax.set_title("Pearson Correlation of Key Features")
    _save(f, out, "fig_2_6_correlation")


@fig("2.7", "Bayes' rule: P(attack) before and after observing evidence", "Units 1 and 2")
def fig_2_7_bayes(r, out):
    rows = r["unit1"]["bayes"]
    f, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(rows))
    ax.barh(y + 0.2, [x["prior"] for x in rows], 0.4, label="Prior P(attack)", color="#9aa8b4")
    ax.barh(y - 0.2, [x["posterior"] for x in rows], 0.4, label="Posterior P(attack | evidence)", color=COLORS["DoS"])
    ax.set_yticks(y, [x["evidence"] for x in rows])
    ax.set(xlim=(0, 1.05), xlabel="Probability", title="Bayes' Rule on Connection Evidence")
    for i, x in enumerate(rows):
        ax.text(x["posterior"] + 0.01, i - 0.2, f"{x['posterior']:.3f}", va="center", fontsize=8)
    ax.legend(loc="lower right")
    _save(f, out, "fig_2_7_bayes")


@fig("2.8", "Polynomial curve fitting: attack rate vs log(connection count)", "Units 1 and 2")
def fig_2_8_polynomial(r, out):
    p = r["unit1"]["polynomial"]
    f, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(p["train_points"]["x"], p["train_points"]["y"], s=18, label="train bins", color=BLUE)
    axes[0].scatter(p["test_points"]["x"], p["test_points"]["y"], s=18, marker="s", label="test bins", color="#ff7f0e")
    for d, style in (("1", "--"), ("3", "-"), ("9", ":")):
        axes[0].plot(p["grid"], p["curves"][d], style, label=f"degree {d}", color="black")
    axes[0].set(title="Fitted Curves", xlabel="log(1 + count)", ylabel="Attack rate", ylim=(-0.1, 1.1))
    axes[0].legend(fontsize=8)
    s = p["scan"]
    axes[1].plot([x["degree"] for x in s], [x["train_rmse"] for x in s], marker="o", label="train")
    axes[1].plot([x["degree"] for x in s], [x["test_rmse"] for x in s], marker="o", label="test")
    axes[1].set(title="RMSE vs Polynomial Degree", xlabel="Degree", ylabel="RMSE")
    axes[1].legend()
    axes[1].grid(True)
    _save(f, out, "fig_2_8_polynomial")


@fig("2.9", "Workflow of the Unit 1 and Unit 2 experiments", "Units 1 and 2")
def fig_2_9_workflow(r, out):
    d = r["unit1"]["describe"]
    _workflow(out, "fig_2_9_workflow", [
        f"KDD Cup 1999 (10%)\n{d['raw_rows']:,} connections", f"Remove {d['duplicates_removed']:,}\nexact duplicates\n→ {d['rows']:,}",
        "Map 22 attacks to\nDoS / Probe / R2L /\nU2R", "Stratified 80:20 split\n(random_state = 42)",
        "One-hot (3) + log1p +\nstandard scaling", "Logistic, LDA, naive\nBayes, RBF SVM",
        "Accuracy, macro-F1,\nper-class recall, ROC", "Bayes' rule and\npolynomial fitting", "Compare with Unit 5\nmodels"], highlight=5)


@fig("2.10", "ROC curves: attack vs normal", "Units 1 and 2")
def fig_2_10_roc(r, out):
    f, ax = plt.subplots(figsize=(7, 6))
    for m in r["unit2"] + r["unit5"]["results"]:
        if "roc" in m:
            ax.plot(m["roc"]["fpr"], m["roc"]["tpr"], label=f"{m['model']} (AUC {m['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="#999")
    ax.set(title="ROC Curve (attack vs normal)", xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(True, alpha=0.4)
    _save(f, out, "fig_2_10_roc")


def _confusion(ax, cm, labels, title):
    cm = np.array(cm)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]:,}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=7.5,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set(title=title, xlabel="Predicted", ylabel="Actual")


@fig("2.11", "Confusion matrix of the best Unit 2 classifier", "Units 1 and 2")
def fig_2_11_confusion_unit2(r, out):
    best = max(r["unit2"], key=lambda m: m["macro_f1"])
    f, ax = plt.subplots(figsize=(7, 6))
    _confusion(ax, best["confusion_matrix"], A.CATEGORIES, f"{best['model']} — test set")
    _save(f, out, "fig_2_11_confusion_unit2")


# ----------------------------------------------------------------- chapter 3: Units 3 and 4
@fig("3.1", "K-means validation accuracy versus the number of clusters", "Unit 3")
def fig_3_1_kmeans_accuracy(r, out):
    s = r["unit3"]["sweep"]
    f, ax = plt.subplots(figsize=(8, 5))
    ax.plot([x["k"] for x in s], [x["validation_accuracy"] * 100 for x in s], marker="o")
    ax.set(title="K-Means Validation Accuracy vs K (attack vs normal)", xlabel="Number of Clusters (K)", ylabel="Validation accuracy (%)")
    ax.grid(True)
    _save(f, out, "fig_3_1_kmeans_accuracy")


@fig("3.2", "Elbow plot of inertia versus the number of clusters", "Unit 3")
def fig_3_2_elbow(r, out):
    s = r["unit3"]["sweep"]
    f, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot([x["k"] for x in s], [x["inertia"] for x in s], marker="o")
    axes[0].set(title="Elbow Method for Selecting K", xlabel="K", ylabel="Inertia")
    axes[1].plot([x["k"] for x in s], [x["silhouette"] for x in s], marker="o", color="#ff7f0e")
    axes[1].set(title="Silhouette Score vs K", xlabel="K", ylabel="Silhouette")
    for ax in axes:
        ax.grid(True)
    _save(f, out, "fig_3_2_elbow")


@fig("3.3", "Confusion matrix of the K-means classification on the test set", "Unit 3")
def fig_3_3_kmeans_confusion(r, out):
    f, ax = plt.subplots(figsize=(6, 5))
    _confusion(ax, r["unit3"]["confusion_matrix"], ["normal", "attack"], f"K-Means (K={r['unit3']['best_k']}) mapped to classes")
    _save(f, out, "fig_3_3_kmeans_confusion")


@fig("3.4", "Cluster composition: share of each category per K-means cluster", "Unit 3")
def fig_3_4_cluster_profile(r, out):
    p = r["unit3"]["profile"]
    f, ax = plt.subplots(figsize=(9, 5))
    ax.bar([str(c["cluster"]) for c in p], [c["connections"] for c in p],
           color=[COLORS["DoS"] if c["mapped"] == "attack" else COLORS["normal"] for c in p])
    for i, c in enumerate(p):
        ax.text(i, c["connections"], f"{c['attack_share']:.0%} attack\n{c['dominant_category']}", ha="center", va="bottom", fontsize=7.5)
    ax.set(title="Connections per Cluster (colour = mapped class)", xlabel="Cluster", ylabel="Connections")
    _save(f, out, "fig_3_4_cluster_profile")


@fig("3.5", "K-means clusters and true categories on the first two principal components", "Unit 3")
def fig_3_5_pca(r, out):
    p = r["unit3"]["pca"]
    f, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    sc = axes[0].scatter(p["x"], p["y"], c=p["cluster"], cmap="viridis", s=6, alpha=0.6)
    f.colorbar(sc, ax=axes[0], label="Cluster")
    axes[0].set(title=f"K-Means Clusters (K={r['unit3']['best_k']})", xlabel="PC 1", ylabel="PC 2")
    for c in A.CATEGORIES:
        idx = [i for i, v in enumerate(p["category"]) if v == c]
        axes[1].scatter([p["x"][i] for i in idx], [p["y"][i] for i in idx], s=6, alpha=0.6, label=c, color=COLORS[c])
    axes[1].set(title="True Categories", xlabel="PC 1", ylabel="PC 2")
    axes[1].legend(markerscale=3)
    # A handful of extreme connections would squash the plot; zoom to the 1st–99th percentile.
    xs, ys = np.array(p["x"]), np.array(p["y"])
    lo_x, hi_x = np.percentile(xs, [1, 99])
    lo_y, hi_y = np.percentile(ys, [1, 99])
    pad_x, pad_y = (hi_x - lo_x) * 0.08, (hi_y - lo_y) * 0.08
    hidden = int(((xs < lo_x - pad_x) | (xs > hi_x + pad_x) | (ys < lo_y - pad_y) | (ys > hi_y + pad_y)).sum())
    for ax in axes:
        ax.set_xlim(lo_x - pad_x, hi_x + pad_x)
        ax.set_ylim(lo_y - pad_y, hi_y + pad_y)
    f.text(0.5, 0.005, f"Axes zoomed to the 1st–99th percentile; {hidden} of {len(xs):,} sampled points lie outside the view.",
           ha="center", fontsize=8, color="#555")
    _save(f, out, "fig_3_5_pca")


@fig("3.6", "Workflow of the Unit 3 K-means experiment", "Unit 3")
def fig_3_6_workflow(r, out):
    u = r["unit3"]
    _workflow(out, "fig_3_6_workflow", [
        "De-duplicated\nconnections", "Stratified 80:20 split", "Binary label:\nattack vs normal",
        f"One-hot + log1p +\nscaling = {u['n_features']} features", "Validation split\n(20% of training)",
        "K sweep: " + ", ".join(str(s["k"]) for s in u["sweep"]), f"Select K = {u['best_k']}\n(best validation\naccuracy)",
        "Map clusters to\nmajority class", "Test evaluation +\nPCA (2-D)"], highlight=6)


@fig("3.7", "Markov transition matrix between categories of consecutive connections", "Unit 4")
def fig_3_7_markov(r, out):
    t = r["unit4"]["markov"]
    m = np.array(t["matrix"])
    f, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(5), t["states"])
    ax.set_yticks(range(5), t["states"])
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{m[i, j]:.3f}", ha="center", va="center", fontsize=8, color="white" if m[i, j] > 0.5 else "black")
    f.colorbar(im, ax=ax)
    ax.set(title="Category Transition Matrix (raw stream)", xlabel="Next connection", ylabel="Current connection")
    _save(f, out, "fig_3_7_markov")


@fig("3.8", "HMM: Baum–Welch training and learned emissions (hidden states → protocol/flag)", "Unit 4")
def fig_3_8_hmm_training(r, out):
    h = r["unit4"]["hmm"]
    f, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(range(1, len(h["trace"]) + 1), h["trace"], marker=".")
    axes[0].set(title="Log-likelihood per EM Iteration", xlabel="Iteration", ylabel="log P(O | λ)")
    axes[0].grid(True)
    B = np.array(h["emission"])
    names = [f"state {s} ({h['state_meaning'].get(str(s), '?')})" for s in range(len(B))]
    im = axes[1].imshow(B, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_xticks(range(len(h["symbols"])), h["symbols"])
    axes[1].set_yticks(range(len(B)), names)
    for i in range(len(B)):
        for j in range(B.shape[1]):
            axes[1].text(j, i, f"{B[i, j]:.3f}", ha="center", va="center", color="white" if B[i, j] > 0.5 else "black")
    axes[1].set(title="Learned Emission Matrix B", xlabel="Observed protocol / flag")
    f.colorbar(im, ax=axes[1])
    _save(f, out, "fig_3_8_hmm_training")


@fig("3.9", "HMM decoding: true attack phase, Viterbi path and posterior P(attack)", "Unit 4")
def fig_3_9_hmm_decoding(r, out):
    h = r["unit4"]["hmm"]
    d = h["demo"]
    t = np.arange(len(d["truth"]))
    f, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    axes[0].step(t, d["truth"], where="mid", color=COLORS["DoS"])
    axes[0].set(ylabel="true\nattack", yticks=[0, 1])
    axes[1].step(t, d["viterbi_attack"], where="mid", color=BLUE)
    axes[1].set(ylabel="Viterbi\nattack", yticks=[0, 1])
    axes[2].fill_between(t, d["posterior_attack"], step="mid", alpha=0.5, color="#ff7f0e")
    axes[2].set(ylabel="P(attack\n| O)", xlabel="Connection index in window", ylim=(0, 1))
    axes[0].set_title(f"Median-accuracy held-out window with both phases ({d['window_accuracy']:.0%} of {h['window']} correct; one of {d['mixed_windows']}) — "
                      f"overall held-out accuracy {h['viterbi_accuracy']:.1%} vs majority baseline {h['majority_baseline']:.1%}",
                      fontsize=10)
    _save(f, out, "fig_3_9_hmm_decoding")


# ----------------------------------------------------------------- chapter 4: Unit 5
@fig("4.1", "Decision tree (top 3 levels)", "Unit 5")
def fig_4_1_tree(r, out, tree=None):
    if tree is None:
        raise KeyError("tree")
    f, ax = plt.subplots(figsize=(16, 7))
    plot_tree(tree["model"], max_depth=3, feature_names=tree["feature_names"], class_names=tree["class_names"],
              filled=True, impurity=False, proportion=True, fontsize=7, ax=ax)
    ax.set_title("Decision Tree (CART) — first three levels")
    _save(f, out, "fig_4_1_tree")


@fig("4.2", "Accuracy and macro-F1 of all classifiers", "Unit 5")
def fig_4_2_model_comparison(r, out):
    rows = r["comparison"]
    y = np.arange(len(rows))
    f, ax = plt.subplots(figsize=(11, 5.5))
    ax.barh(y + 0.2, [m["accuracy"] for m in rows], 0.4, label="Accuracy", color=BLUE)
    ax.barh(y - 0.2, [m["macro_f1"] for m in rows], 0.4, label="Macro-F1", color="#ff7f0e")
    for i, m in enumerate(rows):
        ax.text(m["accuracy"] + 0.005, i + 0.2, f"{m['accuracy']:.4f}", va="center", fontsize=7.5)
        ax.text(m["macro_f1"] + 0.005, i - 0.2, f"{m['macro_f1']:.4f}", va="center", fontsize=7.5)
    ax.set_yticks(y, [m["model"] for m in rows])
    ax.invert_yaxis()
    ax.set(xlim=(0, 1.1), title="Model Comparison on the Test Set")
    ax.legend(loc="lower right")
    _save(f, out, "fig_4_2_model_comparison")


@fig("4.3", "Recall per attack category for every classifier", "Unit 5")
def fig_4_3_recall(r, out):
    rows = r["comparison"]
    m = np.array([[x["recall"][c] for c in A.CATEGORIES] for x in rows])
    f, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(m, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(5), A.CATEGORIES)
    ax.set_yticks(range(len(rows)), [x["model"] for x in rows])
    for i in range(len(rows)):
        for j in range(5):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Recall by Category (rare R2L / U2R are hardest)")
    _save(f, out, "fig_4_3_recall")


@fig("4.4", "Confusion matrix of the best classifier", "Unit 5")
def fig_4_4_confusion_best(r, out):
    best = r["comparison"][0]
    f, ax = plt.subplots(figsize=(7, 6))
    _confusion(ax, best["confusion_matrix"], A.CATEGORIES, f"{best['model']} — test set")
    _save(f, out, "fig_4_4_confusion_best")


@fig("4.5", "Feature importance (one-hot columns summed per attribute)", "Unit 5")
def fig_4_5_importance(r, out):
    imp = r["unit5"]["feature_importance"]
    f, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (name, rows) in zip(axes, imp.items()):
        ax.barh([x["attribute"] for x in rows], [x["importance"] for x in rows], color=BLUE)
        ax.invert_yaxis()
        ax.set(title=f"{name}", xlabel="Importance")
    _save(f, out, "fig_4_5_importance")


@fig("4.6", "Workflow of the Unit 5 tree-based and ensemble experiments", "Unit 5")
def fig_4_6_workflow(r, out):
    _workflow(out, "fig_4_6_workflow", [
        "De-duplicated\nconnections", "Same stratified\n80:20 split", "Same preprocessing\npipeline",
        "Decision tree (CART)\nmax_depth = 12", "Random forest\n100 trees", "AdaBoost\n100 stumps",
        "XGBoost\n200 rounds, depth 6", "Accuracy, macro-F1,\nrecall, ROC, time", "Feature importance +\ncomparison"], highlight=6)


# ----------------------------------------------------------------- extended
@fig("E.1", "Training versus test accuracy (over-fitting check)", "Extended analysis")
def fig_e1_train_test(r, out):
    rows = [m for m in r["comparison"] if "train_accuracy" in m]
    x = np.arange(len(rows))
    f, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - 0.2, [m["train_accuracy"] for m in rows], 0.4, label="Train", color="#7b9cc8")
    ax.bar(x + 0.2, [m["accuracy"] for m in rows], 0.4, label="Test", color="#2d4a7a")
    ax.set_xticks(x, [m["model"] for m in rows], rotation=20, ha="right")
    ax.set(ylim=(min(0.5, min(m["accuracy"] for m in rows) - 0.05), 1.01), ylabel="Accuracy")
    ax.legend()
    _save(f, out, "fig_e1_train_test")


@fig("E.2", "5-fold cross-validated macro-F1", "Extended analysis")
def fig_e2_cv(r, out):
    rows = r["extended"]["cross_validation"]
    f, ax = plt.subplots(figsize=(9, 5))
    ax.bar([x["model"] for x in rows], [x["mean_macro_f1"] for x in rows], yerr=[x["std"] for x in rows], capsize=6, color=BLUE)
    ax.set(ylim=(0, 1), ylabel="Macro-F1", title=f"5-fold CV on {rows[0]['sample_size']:,} connections")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    _save(f, out, "fig_e2_cv")


@fig("E.3", "Clustering comparison: K-means, Gaussian mixture and hierarchical", "Extended analysis")
def fig_e3_clustering(r, out):
    rows = r["extended"]["clustering"]
    x = np.arange(len(rows))
    f, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.2, [m["silhouette"] for m in rows], 0.4, label="Silhouette")
    ax.bar(x + 0.2, [m["adjusted_rand"] for m in rows], 0.4, label="Adjusted Rand vs category")
    ax.set_xticks(x, [m["method"] for m in rows])
    ax.legend()
    _save(f, out, "fig_e3_clustering")


# ----------------------------------------------------------------- code screenshots
CODE_BLOCKS = [
    ("code_ch2_1_loading_preprocessing", "Chapter 2 – Units 1 and 2", "Loading, preprocessing pipeline and split",
     [A.load, A.make_preprocessor, A.split]),
    ("code_ch2_2_bayes", "Chapter 2 – Unit 1", "Bayes' rule on connection evidence", [A.bayes_evidence]),
    ("code_ch2_3_polynomial", "Chapter 2 – Unit 1", "Polynomial curve fitting of the attack rate", [A.polynomial_attack_curve]),
    ("code_ch2_4_classifiers", "Chapter 2 – Unit 2", "Logistic regression, LDA, naive Bayes and SVM",
     [A.unit2_models, A.train_unit2]),
    ("code_ch2_5_evaluation", "Chapter 2 – Unit 2", "Evaluation: accuracy, macro-F1, recall, confusion, ROC", [A.evaluate]),
    ("code_ch3_1_kmeans", "Chapter 3 – Unit 3", "K-means: K selection, test evaluation and PCA", [A.kmeans_study]),
    ("code_ch3_2_markov", "Chapter 3 – Unit 4", "Markov chain over connection categories", [A.markov_categories]),
    ("code_ch3_3_hmm", "Chapter 3 – Unit 4", "HMM of attack phases: observations, Baum–Welch and Viterbi", [A.observation_symbols, A.hmm_attack_phases]),
    ("code_ch3_4_hmm_algorithms", "Chapter 3 – Unit 4", "Scaled forward, backward, sum-product and Viterbi",
     [hmm.HMM.forward, hmm.HMM.backward, hmm.HMM.sum_product, hmm.HMM.viterbi]),
    ("code_ch4_1_ensembles", "Chapter 4 – Unit 5", "Decision tree, random forest, AdaBoost and XGBoost",
     [A.unit5_models, A.train_unit5, A.attribute_importance]),
    ("code_ch4_2_extended", "Extended analysis", "Cross-validation and clustering comparison",
     [A.cross_validation, A.clustering_comparison]),
]


def render_code(out: Path) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for stem, chapter, caption, fns in CODE_BLOCKS:
        code = "\n\n\n".join(_source(fn) for fn in fns)
        png = highlight(code, PythonLexer(), ImageFormatter(style="github-dark", font_name="Consolas", font_size=15,
                                                            line_numbers=False, image_pad=18, line_pad=3))
        (out / f"{stem}.png").write_bytes(png)
        made.append({"file": f"{stem}.png", "chapter": chapter, "caption": caption,
                     "functions": [fn.__qualname__ for fn in fns], "lines": code.count("\n") + 1})
    return made


# ----------------------------------------------------------------- run + summary
def run(extended: bool = True, progress=print, assets: Path = ASSETS) -> dict:
    t0 = time.perf_counter()
    progress("Loading de-duplicated KDD connections and the raw stream")
    df = A.load()
    stream = A.load_raw_stream()
    r: dict = {"study": "attacks", "dataset": {"file": str(A.DATA_FILE), "rows": int(len(df)), "raw_rows": int(len(stream))}}
    progress("Unit 1: summary, distributions, Bayes' rule, polynomial fitting")
    r["unit1"] = {"describe": A.describe(df, len(stream)), "distributions": A.feature_distributions(df),
                  "bayes": A.bayes_evidence(df), "polynomial": A.polynomial_attack_curve(df)}
    X_train, X_test, y_train, y_test = A.split(df)
    progress("Unit 2: logistic regression, LDA, naive Bayes, RBF SVM")
    r["unit2"] = A.train_unit2(df, X_train, X_test, y_train, y_test)
    progress("Unit 3: K-means sweep (slowest step)")
    r["unit3"] = A.kmeans_study(df, X_train, X_test, y_train, y_test, progress)
    gc.collect()
    progress("Unit 4: Markov chain and HMM of attack phases")
    r["unit4"] = {"markov": A.markov_categories(stream), "hmm": A.hmm_attack_phases(stream)}
    del stream
    progress("Unit 5: decision tree, random forest, AdaBoost, XGBoost")
    u5 = A.train_unit5(df, X_train, X_test, y_train, y_test)
    tree = u5.pop("tree")
    r["unit5"] = u5
    r["comparison"] = sorted(r["unit2"] + u5["results"], key=lambda m: (-m["macro_f1"], -m["accuracy"]))
    r["split"] = {"train": int(len(y_train)), "test": int(len(y_test))}
    if extended:
        progress("Extended: cross-validation and clustering comparison")
        r["extended"] = {"cross_validation": A.cross_validation(df, X_train, y_train),
                         "clustering": A.clustering_comparison(df, X_train, y_train)}
    r["summary"] = summarize(r)
    r["generated_at"] = datetime.now(timezone.utc).isoformat()
    r["duration_seconds"] = time.perf_counter() - t0
    r = to_jsonable(r)

    progress("Rendering figures")
    out_fig = assets / "figures"
    out_fig.mkdir(parents=True, exist_ok=True)
    r["figures"] = []
    for spec in FIGURES:
        try:
            spec["fn"](r, out_fig, tree=tree) if spec["fn"] is fig_4_1_tree else spec["fn"](r, out_fig)
            r["figures"].append({k: v for k, v in spec.items() if k != "fn"})
        except KeyError:
            continue
        finally:
            plt.close("all")
    progress("Rendering code screenshots")
    r["code_images"] = render_code(assets / "code")
    (assets / "results.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
    (assets / "summary.md").write_text(markdown(r), encoding="utf-8")
    progress(f"Done in {r['duration_seconds']:.0f}s — saved to {assets}")
    return r


def _pct(x):
    return f"{x * 100:.2f}%"


def summarize(r: dict) -> list[dict]:
    """Plain-language findings; every interpretation is chosen from the computed numbers."""
    d = r["unit1"]["describe"]
    cats = d["category_counts"]
    best_b = max(r["unit1"]["bayes"], key=lambda b: b["posterior"])
    poly = r["unit1"]["polynomial"]["scan"]
    best_deg = min(poly, key=lambda s: s["test_rmse"])
    best2 = max(r["unit2"], key=lambda m: m["macro_f1"])
    worst2 = min(r["unit2"], key=lambda m: m["macro_f1"])
    u3 = r["unit3"]
    mk = r["unit4"]["markov"]
    hm = r["unit4"]["hmm"]
    best = r["comparison"][0]
    top2 = sorted(cats, key=lambda c: -cats[c])[:2]
    top2_share = (cats[top2[0]] + cats[top2[1]]) / d["rows"]
    rare = min(A.CATEGORIES, key=lambda c: best["recall"][c])
    sections = [
        {"title": "What the data looks like", "points": [
            f"{d['raw_rows']:,} raw connections; {d['duplicates_removed']:,} were exact duplicates "
            f"({d['duplicates_removed'] / d['raw_rows']:.0%}), leaving {d['rows']:,} unique connections with {d['features']} features.",
            "Class balance: " + ", ".join(f"{c} {v:,} ({v / d['rows']:.1%})" for c, v in cats.items()) + ".",
            f"Strongest single signal: with '{best_b['evidence']}', P(attack) rises from {best_b['prior']:.3f} to "
            f"{best_b['posterior']:.3f} by Bayes' rule (direct count {best_b['posterior_direct']:.3f}).",
            f"Polynomial fitting of the attack rate against log(count): test error is lowest at degree "
            f"{best_deg['degree']} (RMSE {best_deg['test_rmse']:.4f}); "
            + ("it keeps falling up to the highest degree tested, so this curve is smooth enough not to over-fit."
               if best_deg["degree"] == poly[-1]["degree"] else
               "higher degrees start to over-fit." if poly[-1]["test_rmse"] > best_deg["test_rmse"] * 1.05
               else "higher degrees barely change the error."),
        ]},
        {"title": "Unit 2: linear, generative and kernel classifiers", "points": [
            f"Best: {best2['model']} — accuracy {_pct(best2['accuracy'])}, macro-F1 {best2['macro_f1']:.4f}.",
            f"Weakest: {worst2['model']} — accuracy {_pct(worst2['accuracy'])}, macro-F1 {worst2['macro_f1']:.4f}"
            + (" (its feature-independence assumption does not hold for correlated traffic features)."
               if worst2["model"] == "Gaussian naive Bayes" else "."),
        ]},
        {"title": "Unit 3: K-means clustering", "points": [
            f"Validation accuracy (clusters mapped to attack/normal) peaked at K = {u3['best_k']}; test accuracy "
            f"{_pct(u3['test_accuracy'])} without ever seeing a label during clustering.",
            f"Silhouette {u3['test_silhouette']:.3f}; adjusted Rand {u3['adjusted_rand_category']:.3f} against the five "
            f"categories. The two largest categories ({top2[0]} and {top2[1]}) make up {top2_share:.0%} of connections, "
            + (f"so with K = {u3['best_k']} the clusters mainly split those two; the small families "
               f"({', '.join(c for c in A.CATEGORIES if c not in top2)}) are not separated."
               if u3["best_k"] <= 3 else "and the extra clusters also pick out smaller attack families."),
        ]},
        {"title": "Unit 4: sequences", "points": [
            f"In the raw stream the same category tends to repeat: mean run length {mk['mean_run_length']:.1f} "
            f"connections, longest {mk['longest_run']:,}; P(DoS→DoS) = {mk['matrix'][1][1]:.4f}, "
            f"P(normal→normal) = {mk['matrix'][0][0]:.4f}.",
            f"A {hm['n_states']}-state HMM learned by Baum–Welch from protocol/flag symbols alone decodes the hidden attack phase with "
            f"{_pct(hm['viterbi_accuracy'])} Viterbi accuracy on held-out windows, versus {_pct(hm['majority_baseline'])} for "
            "always guessing the commoner class — "
            + ("so the sequence model adds real information." if hm["viterbi_accuracy"] > hm["majority_baseline"] + 0.01
               else "so on this data it does not beat the trivial baseline."),
        ]},
        {"title": "Unit 5: trees and ensembles", "points": [
            f"Best overall: {best['model']} — accuracy {_pct(best['accuracy'])}, macro-F1 {best['macro_f1']:.4f}"
            + (f", ROC AUC {best['roc_auc']:.4f}" if "roc_auc" in best else "") + ".",
            f"Hardest category even for the best model: {rare} (recall {best['recall'][rare]:.2f}), because it has "
            f"only {cats[rare]:,} examples.",
            "Most important attributes (XGBoost): " + ", ".join(x["attribute"] for x in r["unit5"]["feature_importance"]["XGBoost"][:4]) + ".",
        ]},
    ]
    if "extended" in r:
        cv = max(r["extended"]["cross_validation"], key=lambda m: m["mean_macro_f1"])
        gap = max(r["comparison"], key=lambda m: m.get("train_accuracy", 0) - m["accuracy"])
        sections.append({"title": "Extended analysis", "points": [
            f"5-fold cross-validation: {cv['model']} has the best macro-F1 {cv['mean_macro_f1']:.4f} ± {cv['std']:.4f}.",
            f"Largest train–test accuracy gap: {gap['model']} "
            f"({(gap.get('train_accuracy', gap['accuracy']) - gap['accuracy']) * 100:.2f} points).",
        ]})
    return sections


def markdown(r: dict) -> str:
    lines = ["# Network-attack study (KDD Cup 1999) — results summary", "", f"Generated {r['generated_at']}", ""]
    for s in r["summary"]:
        lines += [f"## {s['title']}", ""] + [f"- {p}" for p in s["points"]] + [""]
    lines += ["## Model comparison (test set)", "", "| Rank | Model | Accuracy | Macro-F1 | ROC AUC | Train time (s) |", "|---|---|---|---|---|---|"]
    for i, m in enumerate(r["comparison"], 1):
        auc = f"{m['roc_auc']:.4f}" if m.get("roc_auc") is not None else "—"
        lines.append(f"| {i} | {m['model']} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {auc} | {m['train_seconds']:.2f} |")
    lines += ["", "## Figures", ""] + [f"- Figure {f['number']}: {f['title']} — `figures/{f['file']}`" for f in r["figures"]]
    lines += ["", "## Code screenshots", ""] + [f"- {c['chapter']}: {c['caption']} — `code/{c['file']}`" for c in r["code_images"]]
    return "\n".join(lines) + "\n"


JOB = StudyJob(lambda extended, progress: run(extended, progress), A.find_dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the network-attack study and save report assets.")
    parser.add_argument("--out", help="Output folder (default: <project>/report_assets/attacks)")
    parser.add_argument("--skip-extended", action="store_true")
    args = parser.parse_args()
    run(not args.skip_extended, progress=lambda m: print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True),
        assets=Path(args.out) if args.out else ASSETS)

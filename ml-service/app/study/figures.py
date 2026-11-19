"""Render the report's figures (and extended-analysis figures) as PNG files from study results."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

DPI = 160
BLUE = "#1f77b4"
CATALOG: list[dict] = []  # filled by @figure: [{"file", "number", "title", "chapter"}]


def figure(number: str, title: str, chapter: str):
    def wrap(fn):
        CATALOG.append({"file": fn.__name__ + ".png", "number": number, "title": title, "chapter": chapter, "fn": fn})
        return fn
    return wrap


def _save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=DPI)
    plt.close(fig)


def _workflow(out: Path, name: str, steps: list[str], highlight: int) -> None:
    """3 × 3 box workflow in the style of the report's architecture diagrams."""
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.axis("off")
    centres = [(c + 0.5, 2.5 - r) for r in range(3) for c in range(3)][: len(steps)]
    for i, ((x, y), text) in enumerate(zip(centres, steps)):
        face = "#fde9cc" if i == highlight else "#e8edf7"
        ax.add_patch(FancyBboxPatch((x - 0.4, y - 0.27), 0.8, 0.54, boxstyle="round,pad=0.02,rounding_size=0.04",
                                    fc=face, ec="#3b4f7a", lw=1.2))
        ax.text(x, y, text, ha="center", va="center", fontsize=8.5, family="serif", wrap=True)
        if i + 1 < len(steps):
            nx, ny = centres[i + 1]
            if ny == y:
                ax.annotate("", (nx - 0.41, ny), (x + 0.41, y), arrowprops=dict(arrowstyle="-|>", color="#333"))
            else:
                ax.annotate("", (nx, ny + 0.28), (x - 0.1, y - 0.28), arrowprops=dict(arrowstyle="-|>", color="#333"))
    _save(fig, out, name)


def _box(ax, stats: list[dict], labels: list[str]) -> None:
    ax.bxp([{"med": s["median"], "q1": s["q1"], "q3": s["q3"], "whislo": s["whislo"], "whishi": s["whishi"],
             "fliers": s["fliers"], "label": lab} for s, lab in zip(stats, labels)],
           showfliers=True, flierprops=dict(marker="o", markersize=4, markerfacecolor="none"))


# ----------------------------------------------------------------- chapter 2
@figure("2.1", "Distribution of flight fares", "Units 1 and 2")
def fig_2_1_fare_distribution(r, out):
    h = r["unit1"]["eda"]["fare_hist"]
    fig, ax = plt.subplots(figsize=(8, 5))
    edges = np.array(h["edges"])
    ax.bar(edges[:-1], h["counts"], width=np.diff(edges), align="edge", color=BLUE, edgecolor="white")
    ax.set(title="Distribution of Flight Fares", xlabel="Flight Fare", ylabel="Number of Flights")
    _save(fig, out, "fig_2_1_fare_distribution")


@figure("2.2", "Flight duration versus flight fare", "Units 1 and 2")
def fig_2_2_duration_vs_fare(r, out):
    s = r["unit1"]["eda"]["scatter"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(s["duration"], s["fare"], alpha=0.3, s=10, color=BLUE)
    ax.set(title="Flight Duration vs Flight Fare", xlabel="Flight Duration (hours)", ylabel="Flight Fare")
    _save(fig, out, "fig_2_2_duration_vs_fare")


@figure("2.3", "Days left before the journey versus flight fare", "Units 1 and 2")
def fig_2_3_days_left_vs_fare(r, out):
    s = r["unit1"]["eda"]["scatter"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(s["days_left"], s["fare"], alpha=0.3, s=10, color=BLUE)
    ax.set(title="Days Left vs Flight Fare", xlabel="Days Left Before Journey", ylabel="Flight Fare")
    _save(fig, out, "fig_2_3_days_left_vs_fare")


@figure("2.4", "Fare distribution by airline", "Units 1 and 2")
def fig_2_4_fare_by_airline(r, out):
    by = r["unit1"]["eda"]["fare_by_airline"]
    names = sorted(by)
    fig, ax = plt.subplots(figsize=(12, 6))
    _box(ax, [by[n] for n in names], names)
    ax.set(title="Flight Fare Distribution by Airline", xlabel="Airline", ylabel="Fare")
    ax.grid(True, alpha=0.4)
    plt.setp(ax.get_xticklabels(), rotation=45)
    _save(fig, out, "fig_2_4_fare_by_airline")


@figure("2.5", "Workflow of the Unit 1 and Unit 2 experiments", "Units 1 and 2")
def fig_2_5_workflow_unit1_2(r, out):
    d = r["unit1"]["describe"]
    _workflow(out, "fig_2_5_workflow_unit1_2", [
        f"Raw dataset\n(Cleaned_dataset.csv)\n{d['rows']:,} x {d['columns']}",
        "Data inspection:\ndtypes, missing\nvalues, describe()",
        "Label encoding of 10\ncategorical columns",
        "Train-test split\n80:20 (random_state\n= 42)",
        "Models: Polynomial\n(degree 2), Linear\n(OLS), Ridge (alpha = 1)",
        "Evaluation: R2, MAE,\nRMSE",
    ], highlight=4)


@figure("2.6", "R² score of the three regression models under the two encoding strategies", "Units 1 and 2")
def fig_2_6_r2_encoding_comparison(r, out):
    label = r["unit2"]["label_encoded"]["results"]
    onehot = r["unit2"]["one_hot"]["results"]
    names = ["Linear", "Polynomial (deg 2)", "Ridge"]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    b1 = ax.bar(x - 0.2, [m["r2"] for m in label], 0.4, label="Label-encoded features", color="#7b9cc8")
    b2 = ax.bar(x + 0.2, [m["r2"] for m in onehot], 0.4, label="One-hot encoded, de-duplicated", color="#2d4a7a")
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.4f", fontsize=8)
    ax.set_xticks(x, names)
    ax.set(ylabel="R² score", ylim=(0, 1.2))
    ax.legend(loc="upper center", ncol=2, fontsize=8)
    _save(fig, out, "fig_2_6_r2_encoding_comparison")


# ----------------------------------------------------------------- chapter 3, Unit 3
@figure("3.1", "Exploratory K-means clusters (K = 6) plotted as flight duration versus fare", "Unit 3")
def fig_3_1_exploratory_kmeans(r, out):
    p = r["unit3"]["exploratory"]["points"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(p["duration"], p["fare"], c=p["cluster"], cmap="viridis", alpha=0.5, s=10)
    ax.set(title="K-Means Clustering of Flight Data", xlabel="Flight Duration (hours)", ylabel="Flight Fare")
    _save(fig, out, "fig_3_1_exploratory_kmeans")


@figure("3.2", "Actual versus predicted fare for the regression using the cluster label", "Unit 3")
def fig_3_2_cluster_regression(r, out):
    p = r["unit3"]["cluster_regression"]["points"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(p["actual"], p["predicted"], alpha=0.5, s=10, color=BLUE)
    ax.set(title="Actual vs Predicted Flight Fare", xlabel="Actual Fare", ylabel="Predicted Fare")
    _save(fig, out, "fig_3_2_cluster_regression")


@figure("3.3", "Workflow of the Unit 3 K-means experiment", "Unit 3")
def fig_3_3_workflow_unit3(r, out):
    u = r["unit3"]["pipeline"]
    ks = ", ".join(str(s["k"]) for s in u["sweep"])
    _workflow(out, "fig_3_3_workflow_unit3", [
        f"Dataset ({r['unit1']['describe']['rows']:,}\nrows) - remove {r['unit1']['describe']['duplicate_rows']:,}\nduplicates",
        "80:20 train-test\nsplit (random_state\n= 42)",
        f"Binary fare label:\nmedian of training\nfares ({u['median_fare']:,.0f})",
        f"One-hot encoding (10\ncolumns) + standard\nscaling (2 columns) =\n{u['n_features']:,} features",
        "Validation split\n(20% of training)\nfor choosing K",
        f"K sweep: {ks}",
        f"Select K = {u['best_k']} (best\nvalidation accuracy)",
        "Final K-means on full\ntraining set; map\nclusters to majority\nclass",
        "Test evaluation +\nPCA (2-D)\nvisualisation",
    ], highlight=6)


@figure("3.4", "K-means validation accuracy versus the number of clusters", "Unit 3")
def fig_3_4_kmeans_validation_accuracy(r, out):
    s = r["unit3"]["pipeline"]["sweep"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([x["k"] for x in s], [x["validation_accuracy"] * 100 for x in s], marker="o")
    ax.set(title="K-Means Validation Accuracy vs K", xlabel="Number of Clusters (K)", ylabel="Validation Accuracy (%)")
    ax.grid(True)
    _save(fig, out, "fig_3_4_kmeans_validation_accuracy")


@figure("3.5", "Elbow plot of inertia versus the number of clusters", "Unit 3")
def fig_3_5_elbow(r, out):
    s = r["unit3"]["pipeline"]["sweep"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([x["k"] for x in s], [x["inertia"] for x in s], marker="o")
    ax.set(title="Elbow Method for Selecting K", xlabel="Number of Clusters (K)", ylabel="Inertia")
    ax.grid(True)
    _save(fig, out, "fig_3_5_elbow")


@figure("3.6", "Confusion matrix of the K-means classification on the test set", "Unit 3")
def fig_3_6_confusion_matrix(r, out):
    cm = np.array(r["unit3"]["pipeline"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="viridis")
    labels = ["Lower Fare", "Higher Fare"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] < cm.max() / 2 else "black")
    ax.set(title="K-Means Confusion Matrix", xlabel="Predicted label", ylabel="True label")
    fig.colorbar(im, ax=ax)
    _save(fig, out, "fig_3_6_confusion_matrix")


@figure("3.7", "Number of flights in each K-means cluster", "Unit 3")
def fig_3_7_cluster_sizes(r, out):
    c = r["unit3"]["pipeline"]["clusters"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([str(x["cluster"]) for x in c], [x["flights"] for x in c], color=BLUE)
    ax.set(title="Number of Flights in Each K-Means Cluster", xlabel="Cluster", ylabel="Number of Flights")
    _save(fig, out, "fig_3_7_cluster_sizes")


@figure("3.8", "K-means clusters projected onto the first two principal components", "Unit 3")
def fig_3_8_pca_clusters(r, out):
    p = r["unit3"]["pipeline"]["pca"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(p["x"], p["y"], c=p["cluster"], cmap="viridis", alpha=0.5, s=8)
    fig.colorbar(sc, ax=ax, label="Cluster")
    ax.set(title=f"K-Means Clustering Diagram (K={r['unit3']['pipeline']['best_k']})",
           xlabel="Principal Component 1", ylabel="Principal Component 2")
    _save(fig, out, "fig_3_8_pca_clusters")


# ----------------------------------------------------------------- chapter 3, Unit 4
@figure("3.9", "Number of flights in each fare category", "Unit 4")
def fig_3_9_fare_category_counts(r, out):
    s = r["unit4"]["states"]["states"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([x["state"] for x in s], [x["records"] for x in s], color=["blue", "orange", "green"])
    ax.set(title="Flight Fare Categories", xlabel="Fare Category", ylabel="Number of Flights")
    _save(fig, out, "fig_3_9_fare_category_counts")


@figure("3.10", "Fare distribution within each fare category", "Unit 4")
def fig_3_10_fare_by_category(r, out):
    s = r["unit4"]["states"]["states"]
    fig, ax = plt.subplots(figsize=(9, 5))
    _box(ax, s, [x["state"] for x in s])
    ax.set(title="Fare Distribution by Category", xlabel="Fare Category", ylabel="Flight Fare (₹)")
    ax.grid(True, alpha=0.4)
    _save(fig, out, "fig_3_10_fare_by_category")


@figure("3.11", "Workflow of the Unit 4 sequential modelling experiment", "Unit 4")
def fig_3_11_workflow_unit4(r, out):
    _workflow(out, "fig_3_11_workflow_unit4", [
        f"Dataset: {r['unit1']['describe']['rows']:,}\nobservations",
        "Sequence_ID =\nFlight_code + Date +\nSource + Destination",
        "Order each sequence\nby Days_left\n(descending)",
        "Discretise Fare into 3\nstates (tertiles):\nLow, Medium, High",
        "Count consecutive\nstate pairs; row-\nnormalise",
        "First-order Markov\ntransition matrix",
        "HMM with pi, A, B\n(report parameters +\nBaum-Welch estimate)",
        "Forward / Backward /\nScaled forward / Sum-\nproduct",
        "Viterbi decoding of\nhidden state\nsequence",
    ], highlight=6)


@figure("3.12", "Estimated fare-state transition matrix", "Unit 4")
def fig_3_12_transition_matrix(r, out):
    t = r["unit4"]["transition"]
    m = np.array(t["matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3), t["states"])
    ax.set_yticks(range(3), t["states"])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{m[i, j]:.4f}", ha="center", va="center", color="white" if m[i, j] > 0.5 else "black")
    ax.set(title="Fare-State Transition Matrix", xlabel="To", ylabel="From")
    fig.colorbar(im, ax=ax)
    _save(fig, out, "fig_3_12_transition_matrix")


# ----------------------------------------------------------------- chapter 4, Unit 5
@figure("4.1", "Workflow of the Unit 5 tree-based and ensemble experiments", "Unit 5")
def fig_4_1_workflow_unit5(r, out):
    u = r["unit5"]
    _workflow(out, "fig_4_1_workflow_unit5", [
        f"Dataset - drop\nduplicates ({r['unit5']['n_rows']:,}\nrows)",
        "Select 10 input\nfeatures; Fare as\ntarget",
        f"One-hot (8\ncategorical) + pass-\nthrough numeric (2)\n= {u['n_encoded_columns']} columns",
        "Train-test split\n80:20 (random_state\n= 42)",
        "Decision Tree\n(CART): max_depth =\n15",
        "Random Forest: 20\ntrees, depth 8",
        "XGBoost: 50 rounds,\ndepth 4, learning rate\n0.1",
        "Evaluation: R2, MAE,\nRMSE, training time",
        "Comparison with Unit\n1 and Unit 2 models",
    ], highlight=8)


@figure("4.2", "XGBoost: actual versus predicted fare for the first 1,000 test records", "Unit 5")
def fig_4_2_xgb_actual_vs_predicted(r, out):
    p = r["unit5"]["xgb_first_1000"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(p["actual"], p["predicted"], alpha=0.5, s=12, color=BLUE)
    ax.set(title="XGBoost: Actual vs Predicted Flight Fare", xlabel="Actual Fare (₹)", ylabel="Predicted Fare (₹)")
    _save(fig, out, "fig_4_2_xgb_actual_vs_predicted")


@figure("4.3", "Comparison of all regression models on the common test set", "Unit 5")
def fig_4_3_model_comparison(r, out):
    rows = r["comparison"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names = [x["model"] for x in rows]
    axes[0].barh(names, [x["r2"] for x in rows], color=BLUE)
    axes[0].set(title="R² score (higher is better)", xlim=(0, 1))
    axes[0].invert_yaxis()
    axes[1].barh(names, [x["mae"] for x in rows], color="#ff7f0e")
    axes[1].set(title="MAE in rupees (lower is better)")
    axes[1].invert_yaxis()
    for ax, key, fmt in ((axes[0], "r2", "{:.4f}"), (axes[1], "mae", "{:,.0f}")):
        for i, x in enumerate(rows):
            ax.text(x[key], i, " " + fmt.format(x[key]), va="center", fontsize=8)
    _save(fig, out, "fig_4_3_model_comparison")


# ----------------------------------------------------------------- extended analysis
@figure("E.1", "Feature importance aggregated to the original attributes", "Extended analysis")
def fig_e1_feature_importance(r, out):
    imp = r["unit5"]["feature_importance"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, rows) in zip(axes, imp.items()):
        ax.barh([x["attribute"] for x in rows], [x["importance"] for x in rows], color=BLUE)
        ax.invert_yaxis()
        ax.set(title=f"{name}: feature importance", xlabel="Importance")
    _save(fig, out, "fig_e1_feature_importance")


@figure("E.2", "Training versus test R² (over-fitting check)", "Extended analysis")
def fig_e2_train_vs_test(r, out):
    rows = r["comparison"]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - 0.2, [m.get("train_r2", np.nan) for m in rows], 0.4, label="Training R²", color="#7b9cc8")
    ax.bar(x + 0.2, [m["r2"] for m in rows], 0.4, label="Test R²", color="#2d4a7a")
    ax.set_xticks(x, [m["model"] for m in rows], rotation=20, ha="right")
    ax.set(ylabel="R²", ylim=(0, 1.05))
    ax.legend()
    _save(fig, out, "fig_e2_train_vs_test")


@figure("E.3", "5-fold cross-validated R² (mean ± standard deviation)", "Extended analysis")
def fig_e3_cross_validation(r, out):
    rows = r["extended"]["cross_validation"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([x["model"] for x in rows], [x["mean_r2"] for x in rows], yerr=[x["std_r2"] for x in rows],
           capsize=6, color=BLUE)
    ax.set(ylabel="R²", ylim=(0, 1), title=f"5-fold CV on {rows[0]['sample_size']:,} sampled records")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    _save(fig, out, "fig_e3_cross_validation")


@figure("E.4", "Clustering comparison: K-means, Gaussian mixture and hierarchical", "Extended analysis")
def fig_e4_clustering_comparison(r, out):
    rows = r["extended"]["clustering"]
    metrics = [("accuracy", "Accuracy vs fare label"), ("silhouette", "Silhouette"), ("adjusted_rand", "Adjusted Rand")]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (key, label) in enumerate(metrics):
        ax.bar(x + (i - 1) * 0.27, [m[key] for m in rows], 0.27, label=label)
    ax.set_xticks(x, [m["method"] for m in rows])
    ax.legend()
    ax.axhline(0, color="#333", lw=0.8)
    _save(fig, out, "fig_e4_clustering_comparison")


@figure("E.5", "Baum–Welch training of the fare-state HMM", "Extended analysis")
def fig_e5_baum_welch(r, out):
    h = r["unit4"]["learned_hmm"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(range(1, len(h["trace"]) + 1), h["trace"], marker=".")
    axes[0].set(title="Log-likelihood per EM iteration", xlabel="Iteration", ylabel="log P(O | λ)")
    axes[0].grid(True)
    B = np.array(h["emission"])
    im = axes[1].imshow(B, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_xticks(range(3), ["Low", "Medium", "High"])
    axes[1].set_yticks(range(3), ["State 0", "State 1", "State 2"])
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f"{B[i, j]:.3f}", ha="center", va="center", color="white" if B[i, j] > 0.5 else "black")
    axes[1].set(title="Learned emission matrix B", xlabel="Observed fare level")
    fig.colorbar(im, ax=axes[1])
    _save(fig, out, "fig_e5_baum_welch")


def render_all(results: dict, out: Path) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for spec in CATALOG:
        try:
            spec["fn"](results, out)
            made.append({k: v for k, v in spec.items() if k != "fn"})
        except KeyError:
            continue  # section not computed (e.g. extended analysis skipped)
    return made

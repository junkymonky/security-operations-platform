"""Render the study's source code as dark-theme PNG images (like the report's Appendix A)."""
from __future__ import annotations

import inspect
import textwrap
from pathlib import Path

from pygments import highlight
from pygments.formatters import ImageFormatter
from pygments.lexers import PythonLexer

from ..ml import hmm
from . import data, unit3, unit4, unit5, unit12

# (file stem, chapter heading, caption, [callables])
BLOCKS = [
    ("code_ch2_1_data_loading", "Chapter 2 – Units 1 and 2", "Dataset loading and validation",
     [data.find_dataset, data.load]),
    ("code_ch2_2_summary_eda", "Chapter 2 – Units 1 and 2", "Summary statistics and EDA data",
     [unit12.describe_dataset, unit12.eda_data]),
    ("code_ch2_3_label_encoded_regression", "Chapter 2 – Units 1 and 2",
     "Linear, polynomial and ridge regression (label-encoded features)", [unit12.label_encoded_regression]),
    ("code_ch2_4_common_pipeline", "Chapter 2 – Units 1 and 2",
     "Common preprocessing pipeline and one-hot regression", [unit12.common_split, unit12.one_hot_regression]),
    ("code_ch3_1_exploratory_kmeans", "Chapter 3 – Unit 3", "Exploratory K-means and regression with the cluster label",
     [unit3.exploratory_kmeans, unit3.cluster_label_regression]),
    ("code_ch3_2_kmeans_pipeline", "Chapter 3 – Unit 3", "K-means pipeline: K selection, evaluation and PCA",
     [unit3.map_clusters_to_classes, unit3.kmeans_pipeline]),
    ("code_ch3_3_sequences_markov", "Chapter 3 – Unit 4", "Sequence construction, fare states and Markov transitions",
     [unit4.build_sequences, unit4.sequence_summary, unit4.transition_matrix]),
    ("code_ch3_4_forward_backward", "Chapter 3 – Unit 4", "Forward and backward algorithms (unscaled)",
     [unit4.forward_algorithm, unit4.backward_algorithm]),
    ("code_ch3_5_scaled_forward_sum_product", "Chapter 3 – Unit 4", "Scaled forward pass, backward pass and sum-product",
     [hmm.HMM.forward, hmm.HMM.backward, hmm.HMM.sum_product]),
    ("code_ch3_6_viterbi", "Chapter 3 – Unit 4", "Viterbi algorithm and the HMM demonstration",
     [hmm.HMM.viterbi, unit4.hmm_demonstration]),
    ("code_ch3_7_baum_welch", "Chapter 3 – Unit 4 (extended)", "Baum–Welch estimation of the HMM from data",
     [hmm.baum_welch, unit4.learned_hmm]),
    ("code_ch4_1_tree_models", "Chapter 4 – Unit 5", "Decision tree, random forest and XGBoost pipelines",
     [unit5.make_preprocessor, unit5.unit5_models, unit5.train_tree_models]),
    ("code_ch4_2_extended", "Chapter 4 – Unit 5 (extended)", "Feature importance, cross-validation and log target",
     [unit5.attribute_importance, unit5.cross_validation, unit5.log_target]),
    ("code_ch4_3_clustering_comparison", "Chapter 3 – Unit 3 (extended)", "K-means vs Gaussian mixture vs hierarchical",
     [unit5.clustering_comparison]),
]


def _source(fn) -> str:
    return textwrap.dedent(inspect.getsource(fn)).rstrip()


def render_all(out: Path) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    formatter_kw = dict(style="github-dark", font_name="Consolas", font_size=15, line_numbers=False,
                        image_pad=18, line_pad=3)
    made = []
    for stem, chapter, caption, fns in BLOCKS:
        code = "\n\n\n".join(_source(fn) for fn in fns)
        png = highlight(code, PythonLexer(), ImageFormatter(**formatter_kw))
        (out / f"{stem}.png").write_bytes(png)
        made.append({"file": f"{stem}.png", "chapter": chapter, "caption": caption,
                     "functions": [fn.__qualname__ for fn in fns], "lines": code.count("\n") + 1})
    return made

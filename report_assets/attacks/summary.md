# Network-attack study (KDD Cup 1999) — results summary

Generated 2026-11-17T21:46:27.400841+00:00

## What the data looks like

- 494,021 raw connections; 348,435 were exact duplicates (71%), leaving 145,586 unique connections with 40 features.
- Class balance: normal 87,832 (60.3%), DoS 54,572 (37.5%), Probe 2,131 (1.5%), R2L 999 (0.7%), U2R 52 (0.0%).
- Strongest single signal: with 'same service rate < 0.1', P(attack) rises from 0.397 to 0.999 by Bayes' rule (direct count 0.999).
- Polynomial fitting of the attack rate against log(count): test error is lowest at degree 10 (RMSE 0.0113); it keeps falling up to the highest degree tested, so this curve is smooth enough not to over-fit.

## Unit 2: linear, generative and kernel classifiers

- Best: Logistic regression — accuracy 99.75%, macro-F1 0.8890.
- Weakest: Gaussian naive Bayes — accuracy 59.99%, macro-F1 0.3086 (its feature-independence assumption does not hold for correlated traffic features).

## Unit 3: K-means clustering

- Validation accuracy (clusters mapped to attack/normal) peaked at K = 2; test accuracy 96.88% without ever seeing a label during clustering.
- Silhouette 0.430; adjusted Rand 0.885 against the five categories. The two largest categories (normal and DoS) make up 98% of connections, so with K = 2 the clusters mainly split those two; the small families (Probe, R2L, U2R) are not separated.

## Unit 4: sequences

- In the raw stream the same category tends to repeat: mean run length 942.8 connections, longest 193,127; P(DoS→DoS) = 0.9999, P(normal→normal) = 0.9975.
- A 4-state HMM learned by Baum–Welch from protocol/flag symbols alone decodes the hidden attack phase with 95.23% Viterbi accuracy on held-out windows, versus 77.98% for always guessing the commoner class — so the sequence model adds real information.

## Unit 5: trees and ensembles

- Best overall: XGBoost — accuracy 99.93%, macro-F1 0.9408, ROC AUC 1.0000.
- Hardest category even for the best model: U2R (recall 0.70), because it has only 52 examples.
- Most important attributes (XGBoost): same_srv_rate, service, count, diff_srv_rate.

## Extended analysis

- 5-fold cross-validation: XGBoost has the best macro-F1 0.9240 ± 0.0751.
- Largest train–test accuracy gap: SVM (RBF kernel) (0.17 points).

## Model comparison (test set)

| Rank | Model | Accuracy | Macro-F1 | ROC AUC | Train time (s) |
|---|---|---|---|---|---|
| 1 | XGBoost | 0.9993 | 0.9408 | 1.0000 | 7.57 |
| 2 | Random forest | 0.9992 | 0.9406 | 1.0000 | 2.55 |
| 3 | Logistic regression | 0.9975 | 0.8890 | 0.9999 | 8.25 |
| 4 | Decision tree (CART) | 0.9974 | 0.8458 | 0.9980 | 2.47 |
| 5 | Linear discriminant (LDA) | 0.9904 | 0.7902 | 0.9957 | 2.29 |
| 6 | SVM (RBF kernel) | 0.9980 | 0.7814 | — | 2.20 |
| 7 | AdaBoost (stumps) | 0.9843 | 0.6442 | 0.9976 | 34.14 |
| 8 | Gaussian naive Bayes | 0.5999 | 0.3086 | 0.9387 | 0.60 |

## Figures

- Figure 2.1: Connections per category (log scale) — `figures/fig_2_1_category_distribution.png`
- Figure 2.2: The 20 most frequent attack types — `figures/fig_2_2_attack_types.png`
- Figure 2.3: Protocol type by connection category — `figures/fig_2_3_protocol_by_category.png`
- Figure 2.4: Distribution of bytes sent: normal vs attack — `figures/fig_2_4_src_bytes_hist.png`
- Figure 2.5: Connections to the same host in the last 2 s (count) by category — `figures/fig_2_5_count_box.png`
- Figure 2.6: Correlation of key traffic features — `figures/fig_2_6_correlation.png`
- Figure 2.7: Bayes' rule: P(attack) before and after observing evidence — `figures/fig_2_7_bayes.png`
- Figure 2.8: Polynomial curve fitting: attack rate vs log(connection count) — `figures/fig_2_8_polynomial.png`
- Figure 2.9: Workflow of the Unit 1 and Unit 2 experiments — `figures/fig_2_9_workflow.png`
- Figure 2.10: ROC curves: attack vs normal — `figures/fig_2_10_roc.png`
- Figure 2.11: Confusion matrix of the best Unit 2 classifier — `figures/fig_2_11_confusion_unit2.png`
- Figure 3.1: K-means validation accuracy versus the number of clusters — `figures/fig_3_1_kmeans_accuracy.png`
- Figure 3.2: Elbow plot of inertia versus the number of clusters — `figures/fig_3_2_elbow.png`
- Figure 3.3: Confusion matrix of the K-means classification on the test set — `figures/fig_3_3_kmeans_confusion.png`
- Figure 3.4: Cluster composition: share of each category per K-means cluster — `figures/fig_3_4_cluster_profile.png`
- Figure 3.5: K-means clusters and true categories on the first two principal components — `figures/fig_3_5_pca.png`
- Figure 3.6: Workflow of the Unit 3 K-means experiment — `figures/fig_3_6_workflow.png`
- Figure 3.7: Markov transition matrix between categories of consecutive connections — `figures/fig_3_7_markov.png`
- Figure 3.8: HMM: Baum–Welch training and learned emissions (hidden states → protocol/flag) — `figures/fig_3_8_hmm_training.png`
- Figure 3.9: HMM decoding: true attack phase, Viterbi path and posterior P(attack) — `figures/fig_3_9_hmm_decoding.png`
- Figure 4.1: Decision tree (top 3 levels) — `figures/fig_4_1_tree.png`
- Figure 4.2: Accuracy and macro-F1 of all classifiers — `figures/fig_4_2_model_comparison.png`
- Figure 4.3: Recall per attack category for every classifier — `figures/fig_4_3_recall.png`
- Figure 4.4: Confusion matrix of the best classifier — `figures/fig_4_4_confusion_best.png`
- Figure 4.5: Feature importance (one-hot columns summed per attribute) — `figures/fig_4_5_importance.png`
- Figure 4.6: Workflow of the Unit 5 tree-based and ensemble experiments — `figures/fig_4_6_workflow.png`
- Figure E.1: Training versus test accuracy (over-fitting check) — `figures/fig_e1_train_test.png`
- Figure E.2: 5-fold cross-validated macro-F1 — `figures/fig_e2_cv.png`
- Figure E.3: Clustering comparison: K-means, Gaussian mixture and hierarchical — `figures/fig_e3_clustering.png`

## Code screenshots

- Chapter 2 – Units 1 and 2: Loading, preprocessing pipeline and split — `code/code_ch2_1_loading_preprocessing.png`
- Chapter 2 – Unit 1: Bayes' rule on connection evidence — `code/code_ch2_2_bayes.png`
- Chapter 2 – Unit 1: Polynomial curve fitting of the attack rate — `code/code_ch2_3_polynomial.png`
- Chapter 2 – Unit 2: Logistic regression, LDA, naive Bayes and SVM — `code/code_ch2_4_classifiers.png`
- Chapter 2 – Unit 2: Evaluation: accuracy, macro-F1, recall, confusion, ROC — `code/code_ch2_5_evaluation.png`
- Chapter 3 – Unit 3: K-means: K selection, test evaluation and PCA — `code/code_ch3_1_kmeans.png`
- Chapter 3 – Unit 4: Markov chain over connection categories — `code/code_ch3_2_markov.png`
- Chapter 3 – Unit 4: HMM of attack phases: observations, Baum–Welch and Viterbi — `code/code_ch3_3_hmm.png`
- Chapter 3 – Unit 4: Scaled forward, backward, sum-product and Viterbi — `code/code_ch3_4_hmm_algorithms.png`
- Chapter 4 – Unit 5: Decision tree, random forest, AdaBoost and XGBoost — `code/code_ch4_1_ensembles.png`
- Extended analysis: Cross-validation and clustering comparison — `code/code_ch4_2_extended.png`

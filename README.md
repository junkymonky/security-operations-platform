# Grep Analyst — SOC console with ML Analytics

A security-operations console (Next.js) with an integrated machine-learning lab
(Python / FastAPI) that implements the five units of the ML syllabus on SOC
datasets. All results are computed on real data by trained models. Datasets,
experiments, model configurations, metrics and predictions are persisted in
the project's existing PostgreSQL (Neon) database.

```
Browser ──► Next.js 16 (port 3000)
              ├─ /                 SOC console (Overview, …, ML Analytics)
              ├─ /incidents/[id]   incident investigation (Postgres)
              ├─ /api/incidents/*  incident API routes (Postgres)
              └─ /ml-api/*  ──rewrite──► FastAPI ML service (port 8000)
                                          ├─ scikit-learn / NumPy / SciPy
                                          ├─ storage/  (CSV files, joblib models)
                                          └─ Postgres tables ml_datasets,
                                             ml_experiments, ml_predictions
```

## Requirements

- Node.js 20+ and **pnpm** (the repo uses `pnpm-lock.yaml`; `npm install` fails on
  pnpm's `node_modules` layout with `Cannot read properties of null (reading 'matches')`)
- Python 3.11+ (tested on 3.12)
- A PostgreSQL database URL (the existing Neon `DATABASE_URL`)

## Configuration

`.env.local` (project root, already used by the Next.js app):

```
DATABASE_URL=postgresql://...?...sslmode=require
# identity shown in the console (there is no auth/users table yet)
ANALYST_NAME=Divya Pandey
ANALYST_ROLE=Security analyst
ORG_NAME=Your organisation        # optional; shown on the workspace card
# optional: where the ML service runs (default http://127.0.0.1:8000)
ML_API_URL=http://127.0.0.1:8000
```

See `.env.example` for the full list.

The ML service reads `DATABASE_URL` from its own environment, then
`ml-service/.env`, then the root `.env.local`, so no secret is duplicated.
Optional ML settings (see `ml-service/.env.example`): `ML_STORAGE_DIR`,
`MAX_UPLOAD_MB` (10), `MAX_ROWS` (200 000), `MAX_COLUMNS` (200), `CORS_ORIGINS`.

The ML tables (`ml_datasets`, `ml_experiments`, `ml_predictions`) are created
automatically on first start; the existing SOC tables are not touched.

## Install and run

```bash
pnpm install           # frontend dependencies
pnpm ml:setup          # creates ml-service/.venv and installs Python requirements
pnpm db:seed-mitre     # one-time: MITRE ATT&CK technique names into mitre_techniques
pnpm db:seed-demo      # optional: 7 demo attack scenarios (INC-2001…2007); --remove deletes them

# terminal 1 — ML service on http://127.0.0.1:8000 (API docs at /docs)
pnpm ml

# terminal 2 — web app on http://localhost:3000
pnpm dev
```

If you ever run `pnpm install` while `pnpm dev` is running, restart the dev
server (it keeps references to the old `node_modules` links).

## SOC console data

Every value on the Overview, Incidents, Alerts, Events, MITRE ATT&CK and AI
Analyst views, and on the incident investigation page, is read from Postgres:

- **Critical alerts**: events with severity `critical`. **Alerts**: `high` + `critical` events.
- **Active incidents**: status `new` / `investigating` / `contained`.
- **Mean time to detect**: average of `detected_at − first event timestamp` per incident.
- **Threat activity**: events (or incidents) bucketed over the chosen range; "ML Anomalies"
  are events whose `detection_method` indicates an ML/anomaly detector.
- **System health**: live database latency, ML service `/health`, time since the last ingested event.
- **Investigation actions** (take ownership, change status, resolve, notes, recommended
  actions, export) are persisted to `incidents`, `investigation_notes` and `audit_logs`.
  The recommended actions and the AI evidence list are derived from the incident's events.

Open http://localhost:3000 (use `localhost`, not `127.0.0.1`: Next.js dev
blocks other origins) and choose **ML Analytics** in the sidebar. On the
**Datasets** tab, import a sample dataset or upload your own CSV.

Run the backend test suite (uses the configured database and cleans up after itself):

```bash
pnpm ml:test
```

## Project report — network-attack study (KDD Cup 1999)

**ML Analytics → Project Report → "Network attacks"** runs the five units on real intrusion data
(`ml-service/data/kdd_attacks_full.csv`, built by `python -m sample_data.make_kdd`) and saves every
diagram on your PC in `report_assets/attacks/` (`figures/`, `code/`, `results.json`, `summary.md`).
Run it from the website or a terminal:

```bash
cd ml-service && .venv\Scripts\python -m app.study.attack_report     # a few minutes
```

Figures: category and attack-type distributions, protocol by category, bytes-sent histogram, count
box plot, feature correlation, Bayes' rule posteriors, polynomial curve fitting, ROC curves, confusion
matrices, K-means accuracy/elbow/silhouette/cluster profiles/PCA, Markov transition matrix, HMM
(Baum–Welch training, emissions, Viterbi decoding of attack phases), decision-tree plot, model
comparison, per-category recall, feature importance, workflow diagrams, train vs test and
cross-validation. Code screenshots for every unit are saved in `report_assets/attacks/code/`.

## Project report (flight-fare study)

The **ML Analytics → Project Report** tab reproduces every experiment, table and figure of the
report on the Kaggle dataset *Airfare ML: Predicting Flight Fares* and extends it.

1. Download `Cleaned_dataset.csv` (or the zip) from
   https://www.kaggle.com/datasets/yashdharme36/airfare-ml-predicting-flight-fares
   and put it in `ml-service/data/` (or set `FLIGHT_FARE_DATA=<path>`).
2. Press **Run study** on the Project Report tab, or run it from a terminal:
   ```bash
   pnpm study                  # full run (about 10–15 min on 452k rows)
   pnpm study --skip-extended  # only the report's experiments
   ```
3. Everything is saved on your PC in `report_assets/`:

| Path | Contents |
|---|---|
| `report_assets/figures/fig_2_1_*.png` … `fig_4_3_*.png` | The report's figures, same numbering (2.1–4.3, plus 3.12 transition matrix) |
| `report_assets/figures/fig_e*.png` | Extended analysis: feature importance, train vs test, cross-validation, clustering comparison, Baum–Welch |
| `report_assets/code/code_ch*.png` | Code screenshots for Appendix A, rendered from the exact source that produced the results |
| `report_assets/results.json` | Every number behind the tables and figures |
| `report_assets/summary.md` | Plain-language findings + model comparison table |

What it runs (same settings as the report: 80:20 split, `random_state = 42`):

- **Units 1–2**: Tables 2.1–2.4; linear, degree-2 polynomial and ridge regression on label-encoded and
  one-hot features. On the one-hot pipeline the polynomial expansion is applied to the two numeric
  attributes (squares and product) so the design matrix fits in memory.
- **Unit 3**: exploratory K-means (K = 6, K = 3 + regression), then the 1,497-feature pipeline with the
  K sweep {2,3,4,5,6,8,10} on a validation split, test accuracy, classification report, confusion
  matrix, cluster profiles and PCA projection.
- **Unit 4**: flight sequences, tertile fare states, transition matrix, and the report's HMM
  (Table 3.10) through forward, backward, scaled forward, sum-product and Viterbi. These reproduce the
  report's Table 3.11 exactly (P(O|λ) = 0.0119528, log = −4.4268, path [0, 1, 1, 1]).
- **Unit 5**: decision tree, random forest and XGBoost with the report's hyperparameters (Table 4.1).
- **Extended** (the report's future scope): attribute-level feature importance, train vs test R²,
  5-fold cross-validation, log-fare target, K-means vs Gaussian mixture vs hierarchical clustering, and a
  Baum–Welch HMM learned from the fare sequences.

## ML-generated investigations

**AI Analyst → Run ML detection** (or `POST /ml-api/soc/detect`) trains a random forest on the
`network_intrusion` sample with 5-fold *out-of-fold* predictions, so every flow is scored by a model
that never saw it. Flows predicted as malicious are grouped by attack type and target host into
incidents `INC-3001…` with the model's confidence, MITRE technique and per-flow events. The sample
flows have no hosts, IPs or times, so a seeded, simulated network context is attached; each incident
summary says so. Re-running replaces the previous ML incidents.

## Using the ML workspace

1. **Datasets**: upload a CSV (drag and drop, 10 MB max) or import a sample.
   Preview rows, inspect inferred column types and missing values, view summary
   statistics, and create a preprocessed copy (drop columns or duplicates,
   impute or drop missing values, z-score outlier removal, scaling, one-hot encoding).
   The original dataset is never modified, and derived datasets record their lineage.
2. Pick the **active dataset** in the toolbar, then open a unit tab.
3. Configure the algorithm and its hyper-parameters, then run it. Every run is saved
   to **Experiments**. Supervised and clustering models are saved with their
   preprocessing pipeline, so you can predict from the **Predict** tab;
   each prediction is logged.

### Sample datasets (`ml-service/sample_data`, regenerate with `python -m sample_data.generate`)

| File | Contents |
|---|---|
| `kdd_attacks.csv` | **Real attack data**: 17,842 connections from KDD Cup 1999 (DARPA 1998 traffic), de-duplicated and stratified. Labels: `attack_type` (normal + 22 attacks), `attack_category` (normal / DoS / Probe / R2L / U2R), `is_attack`; 40 features. Rebuild with `python -m sample_data.make_kdd`, which also writes the full 145,586-row set to `data/kdd_attacks_full.csv`. |
| `network_intrusion.csv` | 805 network flows: 10 numeric features, `protocol`, `attack_type` (benign / brute_force / port_scan / exfiltration / c2_beacon), binary `verdict`, continuous `risk_score` with ~3% gross outliers. It includes overlapping behaviour (backup jobs, low-and-slow brute force), ~3% label noise, 8 missing values and 5 duplicate rows. |
| `auth_sequences.csv` | 24 sessions × 60 events sampled from a known 3-state HMM (hidden `phase`: normal / recon / compromise), plus a noisy random-walk `bytes_out_kb` signal. |

## Syllabus coverage

| Unit | Topic | Where / how |
|---|---|---|
| 1 | Supervised vs unsupervised, dataset handling | Datasets tab: import, validation, preview, column info, summary statistics, preprocessing |
| 1 | Discrete random variables, PMF, entropy | Probability › Random variables (empirical PMF for discrete columns) |
| 1 | Continuous RVs, mean / median / mode, variance, quantiles, skew, kurtosis | Probability › Random variables, Probability density |
| 1 | Expectation and covariance | Probability › Covariance: covariance and correlation (Pearson / Spearman / Kendall); numeric check of Cov = E[XY] − E[X]E[Y] |
| 1 | Probability densities | Histogram density, Gaussian KDE, MLE fits (normal, Laplace, logistic, exponential, lognormal, gamma) ranked by AIC; ECDF; P(a ≤ X ≤ b); Shapiro–Wilk |
| 1 | Sum and product rules, Bayes' rule, independence, conditional independence | Probability › Bayes' rule: prior, likelihood, evidence via the sum rule, posterior (checked against direct counts); joint table; χ² and mutual information; I(A;B\|C) |
| 1 | Polynomial curve fitting | Closed-form regularised least squares, degree 0–15, λ; train/test RMSE vs degree (over-fitting) and vs ln λ |
| 2 | Least squares and MLE | `least_squares`: explicit lstsq, σ²_ML, β_ML, log-likelihood; matches `linear` |
| 2 | Ridge regression | `ridge` plus the coefficient shrinkage path |
| 2 | Robust regression | `huber` (Huber loss); count of down-weighted outliers |
| 2 | Bayesian linear regression | Custom posterior N(m_N, S_N), α and β by evidence maximisation, ±σ predictive error bars |
| 2 | Linear classification, discriminant functions | `perceptron`, `lda` (Fisher / shared covariance), `qda` |
| 2 | Generative vs discriminative models | `lda`, `qda`, `naive_bayes` (generative) vs `logistic`, `bayesian_logistic` (discriminative); family shown per model |
| 2 | Laplace approximation, Bayesian logistic regression | Custom IRLS MAP, Hessian posterior covariance, probit-moderated predictive, Laplace evidence; one-vs-rest for multiclass |
| 2 | Kernels, kernel trick, SVM | `svm` (linear / poly / RBF / sigmoid; C, γ, degree, coef0), Gram-matrix heat map with PSD check, support-vector counts, 2-D decision regions |
| 2 | Evaluation | Accuracy, precision, recall, F1, confusion matrix, per-class table, ROC/AUC, log loss; MAE / MSE / RMSE / R²; optional k-fold CV; side-by-side comparison chart |
| 3 | K-means | Lloyd's algorithm, centroids (2-D and standardised), elbow and silhouette scan over K |
| 3 | Mixtures of Gaussians, EM | EM run one iteration at a time to plot the monotone log-likelihood; responsibilities γ(z_nk); BIC/AIC over K |
| 3 | Hierarchical clustering, dissimilarity measures | Ward / complete / average / single × Euclidean / Manhattan / cosine / Chebyshev / correlation; dendrogram with cut line; cophenetic correlation |
| 3 | Clustering evaluation | Silhouette, Davies–Bouldin, Calinski–Harabasz, inertia; ARI / NMI / homogeneity against reference labels; three-way comparison |
| 3 | PCA, choosing the latent dimension | Scree plot, cumulative variance, loadings, reconstruction error, 90/95% rule, Kaiser rule, elbow, probabilistic-PCA CV likelihood; 2-D and rotatable 3-D projections |
| 3 | Factor analysis | Loadings (varimax / quartimax), noise variances ψ, communalities, CV log-likelihood vs number of factors |
| 4 | Markov models | MLE transition matrix (optional Laplace smoothing), transition graph, n-step matrix, stationary distribution, simulation, log-likelihood |
| 4 | HMM, forward / backward, scaling factors | NumPy implementation: unscaled forward pass (shows underflow) and scaled forward/backward with c_t, ln p(X) = Σ ln c_t, α̂/β̂ tables |
| 4 | Sum-product | Separate factor-graph message passing; checked against forward–backward γ (difference ≈ 1e-16) |
| 4 | Viterbi | Max-sum decoding in log space; accuracy against true hidden states (Hungarian state matching) |
| 4 | Maximum likelihood for HMMs | Baum–Welch EM over multiple sequences with restarts; monotone likelihood trace |
| 4 | Linear dynamical systems | Kalman filter and RTS smoother (random-walk or constant-velocity model), q and r estimated by MLE, ±2σ bands, forecast |
| 5 | CART | Classification and regression trees, interactive tree view, cost-complexity pruning path |
| 5 | Boosting, adaptive basis functions | AdaBoost on stumps (weights α_m, weighted errors, staged accuracy vs one weak learner); gradient boosting |
| 5 | Ensemble learning | Random forest (OOB score, ensemble vs individual-tree scores), bagging, soft-voting ensemble |
| 5 | Bayesian model averaging | Every feature subset as a model, BIC-approximate posterior p(M\|D), inclusion probabilities, BMA vs best single vs full model |
| 5 | Generalized additive models | Cubic B-spline basis per feature with an L2 penalty (Ridge or logistic); plotted shape functions f_j |
| 5 | Evaluation | Impurity and permutation feature importance, comparison table and chart |

## SOC API (Next.js route handlers)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/overview?range=1h\|6h\|24h\|7d\|all&metric=events\|detections\|alerts\|incidents\|critical\|ml` | Metrics, activity series, incidents, techniques, DB latency |
| GET / POST | `/api/incidents` | List incidents / create one (`{title, severity, summary?}`) |
| GET / PATCH | `/api/incidents/{id}` | Incident / update `status` and/or `assigned_analyst` (audited) |
| GET | `/api/incidents/{id}/events` | Events linked to an incident |
| GET / POST | `/api/incidents/{id}/notes` | Investigation notes |
| GET / POST | `/api/incidents/{id}/audit` | Audit history / record an analyst action |
| GET | `/api/events?severity=high,critical&limit=` | Events, newest first |

## ML API (FastAPI, interactive docs at http://127.0.0.1:8000/docs)

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Service and database status |
| GET / POST | `/datasets` | List datasets / upload a CSV (multipart `file`) |
| GET / POST | `/datasets/samples`, `/datasets/samples/{key}` | List / import bundled sample datasets |
| GET / DELETE | `/datasets/{id}` | Metadata with column info and summary / delete |
| GET | `/datasets/{id}/preview?offset&limit` | Paged rows |
| POST | `/datasets/{id}/preprocess` | Create a cleaned, derived dataset |
| POST | `/datasets/{id}/stats/{describe,covariance,distribution,bayes,polynomial}` | Unit 1 |
| GET | `/algorithms` | Supervised algorithm catalogue |
| POST | `/datasets/{id}/train` | Train and compare up to 10 supervised models (Units 2 and 5) |
| POST | `/datasets/{id}/bma` | Bayesian model averaging |
| POST | `/datasets/{id}/cluster` | `kmeans`, `gmm`, `hierarchical`, `compare` |
| POST | `/datasets/{id}/decompose` | `pca`, `factor_analysis` |
| POST | `/datasets/{id}/markov` | Markov chain estimation |
| POST | `/datasets/{id}/hmm` | Baum–Welch HMM learning and decoding |
| POST | `/hmm/decode` | Inference with user-specified π, A, B (observed or sampled sequence) |
| POST | `/datasets/{id}/lds` | Kalman filter / smoother |
| GET | `/experiments?unit&dataset_id` | Experiment history |
| GET / DELETE | `/experiments/{id}` | Full experiment record / delete (also removes the saved model) |
| POST | `/experiments/{id}/predict` | `{"records": [{feature: value, …}]}` → predictions, stored in `ml_predictions` |
| GET | `/experiments/{id}/predictions` | Prediction history |
| GET | `/study/flight-fare` | Dataset status, job progress and study results |
| POST | `/study/flight-fare/run` | Start the study in the background (`{"extended": true}`) |
| GET | `/study/flight-fare/figures/{file}` · `/code/{file}` | Generated PNG figures / code screenshots |
| POST | `/soc/detect` | Score network flows and (re)create ML-generated incidents |

Errors return `{"detail": "..."}` with 404 (unknown id), 413 (too large),
422 (invalid input) or 503 (database unavailable).

## Project layout

```
app/, components/          Next.js app (existing console; ML UI in components/ml/)
components/ml/             api.ts client, ui.tsx primitives, one file per unit tab
scripts/ml.mjs             cross-platform ML service launcher (pnpm ml / ml:setup / ml:test)
ml-service/
  app/main.py              FastAPI app, error handling, lifespan (DB init)
  app/config.py, db.py     settings and Postgres pool and schema
  app/schemas.py           request validation (Pydantic)
  app/routers/             datasets, statistics, models, experiments
  app/services/            datasets, experiments, supervised, unsupervised, sequential
  app/ml/                  statistics, polynomial, regression, classification, svm,
                           clustering, decomposition, hmm, lds, ensemble, metrics, preprocessing
  sample_data/             generator and generated CSVs
  app/study/               flight-fare study: data, unit12, unit3, unit4, unit5, figures, codeshots, runner
  app/services/soc_detection.py   ML predictions → SOC incidents
  tests/test_api.py        end-to-end API tests (22)
  tests/test_study.py      study pipeline tests on a TEST-ONLY stand-in (5)
```

## Security notes

- Uploads: extension, size and binary-content checks; rows and columns are capped;
  files are stored under random ids (client file names are never used as paths);
  column names are sanitised and de-duplicated.
- All SQL is parameterised; ids are validated before use as file names.
- Saved models are loaded with joblib. Only load model files this service created
  (never place untrusted `.joblib` files in `storage/models`).
- The ML service has no authentication of its own. Keep it bound to 127.0.0.1
  behind the Next.js proxy; do not expose port 8000 publicly.

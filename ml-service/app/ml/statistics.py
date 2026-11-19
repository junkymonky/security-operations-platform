"""Unit 1 — probability & descriptive statistics.

Covers: discrete and continuous random variables, mean/median/mode, variance,
quantiles, expectation, covariance/correlation, probability densities (histogram,
KDE, MLE-fitted parametric densities), Bayes' rule, the sum/product rules,
independence and conditional independence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..errors import ValidationFailed

QUANTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        raise ValidationFailed(f"Column '{column}' not found.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValidationFailed(f"Column '{column}' must be numeric.")
    s = df[column].dropna().astype(float)
    if len(s) < 3:
        raise ValidationFailed(f"Column '{column}' needs at least 3 non-missing values.")
    return s


def describe(df: pd.DataFrame, columns: list[str] | None = None) -> dict:
    columns = columns or list(df.columns)
    numeric, discrete = [], []
    for col in columns:
        if col not in df.columns:
            raise ValidationFailed(f"Column '{col}' not found.")
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 15:
            x = s.dropna().astype(float)
            if x.empty:
                continue
            mode = x.mode()
            numeric.append(
                {
                    "column": col,
                    "count": int(x.count()),
                    "missing": int(s.isna().sum()),
                    "mean": x.mean(),
                    "median": x.median(),
                    "mode": mode.iloc[0] if not mode.empty else None,
                    "variance": x.var(ddof=1),
                    "variance_population": x.var(ddof=0),
                    "std": x.std(ddof=1),
                    "min": x.min(),
                    "max": x.max(),
                    "range": x.max() - x.min(),
                    "iqr": x.quantile(0.75) - x.quantile(0.25),
                    "skewness": stats.skew(x, bias=False) if len(x) > 2 else None,
                    "kurtosis": stats.kurtosis(x, bias=False) if len(x) > 3 else None,
                    "quantiles": {f"q{int(q * 100)}": x.quantile(q) for q in QUANTILES},
                }
            )
        else:
            # Treat as a discrete random variable: empirical probability mass function.
            counts = s.astype("string").value_counts(dropna=True)
            total = counts.sum()
            pmf = [{"value": str(k), "count": int(v), "p": v / total} for k, v in counts.head(30).items()]
            entry = {
                "column": col,
                "count": int(total),
                "missing": int(s.isna().sum()),
                "cardinality": int(len(counts)),
                "pmf": pmf,
                "entropy_bits": float(stats.entropy(counts.to_numpy(), base=2)) if total else None,
            }
            if pd.api.types.is_numeric_dtype(s):
                x = s.dropna().astype(float)
                entry["expectation"] = x.mean()
                entry["variance"] = x.var(ddof=0)
            discrete.append(entry)
    return {"continuous": numeric, "discrete": discrete}


def covariance(df: pd.DataFrame, columns: list[str] | None, method: str = "pearson") -> dict:
    num = df.select_dtypes(include="number")
    columns = columns or list(num.columns)
    bad = [c for c in columns if c not in num.columns]
    if bad:
        raise ValidationFailed(f"Non-numeric or unknown column(s): {', '.join(bad)}")
    if len(columns) < 2:
        raise ValidationFailed("Select at least two numeric columns.")
    if method not in ("pearson", "spearman", "kendall"):
        raise ValidationFailed("method must be pearson, spearman or kendall.")
    data = num[columns].dropna()
    if len(data) < 3:
        raise ValidationFailed("Not enough complete rows to compute covariance.")
    cov = data.cov()
    corr = data.corr(method=method)
    # Expectation of products check: Cov(X,Y) = E[XY] - E[X]E[Y] (population form)
    first, second = columns[0], columns[1]
    exy = float((data[first] * data[second]).mean())
    ex, ey = float(data[first].mean()), float(data[second].mean())
    return {
        "columns": columns,
        "covariance": cov.to_numpy().tolist(),
        "correlation": corr.to_numpy().tolist(),
        "method": method,
        "n": int(len(data)),
        "identity_check": {
            "x": first,
            "y": second,
            "E[XY]": exy,
            "E[X]E[Y]": ex * ey,
            "cov_population": exy - ex * ey,
            "cov_population_numpy": float(np.cov(data[first], data[second], ddof=0)[0, 1]),
        },
    }


def _fit_families(x: np.ndarray) -> list[dict]:
    """Maximum-likelihood fits of candidate densities; ranked by AIC."""
    fits = []
    candidates = {"normal": stats.norm, "laplace": stats.laplace, "logistic": stats.logistic}
    if (x > 0).all():
        candidates.update({"exponential": stats.expon, "lognormal": stats.lognorm, "gamma": stats.gamma})
    for name, dist in candidates.items():
        try:
            if name in ("exponential", "lognormal", "gamma"):
                params = dist.fit(x, floc=0)
                k = len(params) - 1
            else:
                params = dist.fit(x)
                k = len(params)
            ll = float(np.sum(dist.logpdf(x, *params)))
            if not np.isfinite(ll):
                continue
            fits.append({"family": name, "params": list(params), "log_likelihood": ll, "aic": 2 * k - 2 * ll, "_dist": dist})
        except Exception:
            continue
    fits.sort(key=lambda f: f["aic"])
    return fits


def distribution(
    df: pd.DataFrame, column: str, bins: int = 30, lower: float | None = None, upper: float | None = None
) -> dict:
    s = _numeric(df, column)
    x = s.to_numpy()
    bins = int(min(max(bins, 5), 100))
    counts, edges = np.histogram(x, bins=bins, density=True)
    lo, hi = float(x.min()), float(x.max())
    pad = (hi - lo) * 0.05 or 1.0
    grid = np.linspace(lo - pad, hi + pad, 200)

    curves = {"x": grid}
    if np.std(x) > 0:
        kde = stats.gaussian_kde(x)
        curves["kde"] = kde(grid)
    fits = _fit_families(x)
    for f in fits[:3]:
        curves[f["family"]] = f["_dist"].pdf(grid, *f["params"])
    mu, sigma = float(np.mean(x)), float(np.std(x, ddof=0))  # Gaussian MLE

    ecdf_x = np.sort(x)
    idx = np.linspace(0, len(ecdf_x) - 1, min(len(ecdf_x), 150)).astype(int)
    ecdf = [{"x": ecdf_x[i], "p": (i + 1) / len(ecdf_x)} for i in idx]

    lower = lo if lower is None else float(lower)
    upper = hi if upper is None else float(upper)
    if lower > upper:
        raise ValidationFailed("lower must be <= upper.")
    interval = {
        "lower": lower,
        "upper": upper,
        "empirical": float(np.mean((x >= lower) & (x <= upper))),
        "normal_model": float(stats.norm.cdf(upper, mu, sigma) - stats.norm.cdf(lower, mu, sigma)) if sigma > 0 else None,
    }
    best = fits[0] if fits else None
    if best:
        interval[f"{best['family']}_model"] = float(
            best["_dist"].cdf(upper, *best["params"]) - best["_dist"].cdf(lower, *best["params"])
        )

    return {
        "column": column,
        "n": int(len(x)),
        "histogram": [
            {"x0": edges[i], "x1": edges[i + 1], "mid": (edges[i] + edges[i + 1]) / 2, "density": counts[i]}
            for i in range(len(counts))
        ],
        "curve": [{k: v[i] for k, v in curves.items()} for i in range(len(grid))],
        "curve_series": [k for k in curves if k != "x"],
        "fits": [{k: v for k, v in f.items() if k != "_dist"} for f in fits],
        "moments": {
            "expectation": mu,
            "variance_mle": sigma**2,
            "variance_unbiased": float(np.var(x, ddof=1)),
            "std": sigma,
            "median": float(np.median(x)),
        },
        "quantiles": {f"q{int(q * 100)}": float(np.quantile(x, q)) for q in QUANTILES},
        "normal_quantiles": {f"q{int(q * 100)}": float(stats.norm.ppf(q, mu, sigma)) for q in QUANTILES} if sigma > 0 else {},
        "ecdf": ecdf,
        "interval_probability": interval,
        "normality_test": _normality(x),
    }


def _normality(x: np.ndarray) -> dict:
    sample = x if len(x) <= 5000 else np.random.default_rng(0).choice(x, 5000, replace=False)
    stat, p = stats.shapiro(sample) if len(sample) >= 3 else (None, None)
    return {"test": "Shapiro-Wilk", "statistic": stat, "p_value": p}


def _event(df: pd.DataFrame, column: str, value: str | None, threshold: float | None, op: str) -> pd.Series:
    if column not in df.columns:
        raise ValidationFailed(f"Column '{column}' not found.")
    s = df[column]
    if threshold is not None:
        if not pd.api.types.is_numeric_dtype(s):
            raise ValidationFailed(f"Threshold events need a numeric column; '{column}' is not numeric.")
        ops = {">": s > threshold, ">=": s >= threshold, "<": s < threshold, "<=": s <= threshold}
        if op not in ops:
            raise ValidationFailed("op must be one of >, >=, <, <=")
        return ops[op].fillna(False)
    if value is None:
        raise ValidationFailed(f"Provide a value or threshold for '{column}'.")
    return (s.astype(str) == str(value)).fillna(False)


def _label(column, value, threshold, op):
    return f"{column} {op} {threshold}" if threshold is not None else f"{column} = {value}"


def bayes(df: pd.DataFrame, req: dict) -> dict:
    """Bayes' rule P(A|B) = P(B|A)P(A)/P(B) estimated from data, plus independence tests."""
    A = _event(df, req["hypothesis_column"], req.get("hypothesis_value"), req.get("hypothesis_threshold"), req.get("hypothesis_op", ">="))
    B = _event(df, req["evidence_column"], req.get("evidence_value"), req.get("evidence_threshold"), req.get("evidence_op", ">="))
    n = len(df)
    if n == 0:
        raise ValidationFailed("Dataset is empty.")
    pA, pB = A.mean(), B.mean()
    pAB = (A & B).mean()
    if pA == 0 or pB == 0:
        raise ValidationFailed("One of the events never occurs in the data; choose a different value/threshold.")
    p_B_given_A = pAB / pA
    p_B_given_notA = ((~A) & B).mean() / (1 - pA) if pA < 1 else 0.0
    # Sum rule: P(B) = P(B|A)P(A) + P(B|¬A)P(¬A)
    pB_sum_rule = p_B_given_A * pA + p_B_given_notA * (1 - pA)
    posterior = p_B_given_A * pA / pB_sum_rule
    direct = pAB / pB

    table = pd.crosstab(A.rename("A"), B.rename("B"))
    table = table.reindex(index=[False, True], columns=[False, True], fill_value=0)
    counts = table.to_numpy()
    chi2 = chi_p = None
    if (counts.sum(axis=0) > 0).all() and (counts.sum(axis=1) > 0).all():
        chi2, chi_p, _, _ = stats.chi2_contingency(counts, correction=False)

    result = {
        "hypothesis": _label(req["hypothesis_column"], req.get("hypothesis_value"), req.get("hypothesis_threshold"), req.get("hypothesis_op", ">=")),
        "evidence": _label(req["evidence_column"], req.get("evidence_value"), req.get("evidence_threshold"), req.get("evidence_op", ">=")),
        "n": n,
        "prior": pA,
        "likelihood": p_B_given_A,
        "likelihood_not": p_B_given_notA,
        "evidence_prob": pB,
        "evidence_sum_rule": pB_sum_rule,
        "posterior": posterior,
        "posterior_direct": direct,
        "joint": pAB,
        "joint_table": {
            "rows": ["not A", "A"],
            "cols": ["not B", "B"],
            "counts": table.to_numpy().tolist(),
            "probabilities": (table.to_numpy() / n).tolist(),
        },
        "independence": {
            "p_joint": pAB,
            "p_product": pA * pB,
            "difference": pAB - pA * pB,
            "mutual_information_bits": _mutual_information(A, B),
            "chi2": chi2,
            "p_value": chi_p,
            "independent_at_5pct": None if chi_p is None else bool(chi_p > 0.05),
        },
    }

    cond_col = req.get("condition_column")
    if cond_col:
        if cond_col not in df.columns:
            raise ValidationFailed(f"Column '{cond_col}' not found.")
        C = df[cond_col]
        if pd.api.types.is_numeric_dtype(C) and C.nunique() > 10:
            C = pd.qcut(C, q=4, duplicates="drop").astype(str)
        rows = []
        cmi = 0.0
        for c_val, idx in C.astype(str).groupby(C.astype(str)).groups.items():
            a, b = A.loc[idx], B.loc[idx]
            pc = len(idx) / n
            mi = _mutual_information(a, b)
            cmi += pc * mi
            rows.append(
                {
                    "condition": c_val,
                    "p_c": pc,
                    "p_ab_given_c": (a & b).mean(),
                    "p_a_given_c_times_p_b_given_c": a.mean() * b.mean(),
                    "mutual_information_bits": mi,
                }
            )
        result["conditional_independence"] = {
            "condition_column": cond_col,
            "by_condition": rows,
            "conditional_mutual_information_bits": cmi,
        }
    return result


def _mutual_information(a: pd.Series, b: pd.Series) -> float:
    a, b = a.to_numpy(bool), b.to_numpy(bool)
    mi = 0.0
    for va in (False, True):
        for vb in (False, True):
            pab = np.mean((a == va) & (b == vb))
            pa, pb = np.mean(a == va), np.mean(b == vb)
            if pab > 0:
                mi += pab * np.log2(pab / (pa * pb))
    return float(mi)

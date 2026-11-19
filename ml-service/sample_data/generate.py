"""Generate the bundled SOC sample datasets (deterministic, seed 7).

    python -m sample_data.generate      (from the ml-service directory)

network_intrusion.csv
    800 synthetic network flows. Five behaviour classes (benign, brute_force,
    port_scan, exfiltration, c2_beacon) with distinct feature distributions, a
    binary `verdict`, and a continuous `risk_score` regression target containing
    ~3% gross outliers (useful for robust regression). Benign backup jobs and
    low-and-slow brute force overlap other classes, and ~3% of verdicts are
    mislabelled, so classifiers cannot reach 100%. ~1% missing payload_entropy
    values and a handful of duplicate rows exercise preprocessing.

auth_sequences.csv
    24 sessions x 60 steps sampled from a known 3-state HMM (hidden `phase`:
    normal / recon / compromise) emitting authentication/endpoint events, plus a
    noisy random-walk `bytes_out_kb` signal for Kalman filtering.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
rng = np.random.default_rng(7)


def network_intrusion(n: int = 800) -> pd.DataFrame:
    classes = ["benign", "brute_force", "port_scan", "exfiltration", "c2_beacon"]
    probs = [0.55, 0.15, 0.12, 0.10, 0.08]
    labels = rng.choice(classes, size=n, p=probs)
    rows = []
    for lab in labels:
        if lab == "benign" and rng.random() < 0.08:
            # Nightly backup jobs: large, high-entropy uploads that resemble exfiltration.
            r = dict(duration_s=rng.gamma(4, 25), src_bytes=rng.lognormal(12.5, 1.2), dst_bytes=rng.lognormal(7.5, 0.8),
                     packets=rng.poisson(500), failed_logins=rng.poisson(0.3), distinct_ports=rng.poisson(1) + 1,
                     conn_rate=rng.normal(4, 1.5), payload_entropy=rng.normal(7.0, 0.5),
                     hour=int(rng.normal(1, 2)) % 24, protocol="tcp")
        elif lab == "brute_force" and rng.random() < 0.3:
            # Low-and-slow password spraying that stays under rate thresholds.
            r = dict(duration_s=rng.gamma(2, 6), src_bytes=rng.lognormal(7, 0.8), dst_bytes=rng.lognormal(7.5, 0.8),
                     packets=rng.poisson(25), failed_logins=rng.poisson(2.5), distinct_ports=1,
                     conn_rate=rng.normal(8, 3), payload_entropy=rng.normal(4.0, 0.6),
                     hour=int(rng.normal(10, 4)) % 24, protocol="tcp")
        elif lab == "benign":
            r = dict(duration_s=rng.gamma(2, 8), src_bytes=rng.lognormal(8, 1), dst_bytes=rng.lognormal(9, 1),
                     packets=rng.poisson(40), failed_logins=rng.poisson(0.1), distinct_ports=rng.poisson(2) + 1,
                     conn_rate=rng.normal(5, 2), payload_entropy=rng.normal(4.5, 0.6),
                     hour=int(rng.normal(13, 3)) % 24, protocol=rng.choice(["tcp", "udp"], p=[0.8, 0.2]))
        elif lab == "brute_force":
            r = dict(duration_s=rng.gamma(2, 2), src_bytes=rng.lognormal(6, 0.5), dst_bytes=rng.lognormal(6, 0.5),
                     packets=rng.poisson(15), failed_logins=rng.poisson(25), distinct_ports=1,
                     conn_rate=rng.normal(40, 8), payload_entropy=rng.normal(3.5, 0.4),
                     hour=int(rng.normal(3, 2)) % 24, protocol="tcp")
        elif lab == "port_scan":
            r = dict(duration_s=rng.gamma(1, 0.5), src_bytes=rng.lognormal(4, 0.4), dst_bytes=rng.lognormal(3, 0.5),
                     packets=rng.poisson(3) + 1, failed_logins=0, distinct_ports=rng.poisson(120) + 20,
                     conn_rate=rng.normal(80, 15), payload_entropy=rng.normal(1.5, 0.5),
                     hour=int(rng.integers(0, 24)), protocol=rng.choice(["tcp", "udp", "icmp"], p=[0.6, 0.2, 0.2]))
        elif lab == "exfiltration":
            r = dict(duration_s=rng.gamma(4, 30), src_bytes=rng.lognormal(14, 0.8), dst_bytes=rng.lognormal(7, 0.6),
                     packets=rng.poisson(900), failed_logins=rng.poisson(0.2), distinct_ports=rng.poisson(1) + 1,
                     conn_rate=rng.normal(3, 1), payload_entropy=rng.normal(7.6, 0.2),
                     hour=int(rng.normal(23, 2)) % 24, protocol="tcp")
        else:  # c2_beacon
            r = dict(duration_s=rng.normal(2, 0.3), src_bytes=rng.lognormal(6.5, 0.2), dst_bytes=rng.lognormal(6.8, 0.2),
                     packets=rng.poisson(8), failed_logins=0, distinct_ports=1,
                     conn_rate=rng.normal(12, 1), payload_entropy=rng.normal(6.8, 0.3),
                     hour=int(rng.integers(0, 24)), protocol=rng.choice(["tcp", "udp"], p=[0.7, 0.3]))
        r["attack_type"] = lab
        rows.append(r)
    df = pd.DataFrame(rows)
    df["duration_s"] = df["duration_s"].clip(lower=0.05).round(3)
    df["src_bytes"] = df["src_bytes"].round().astype(int)
    df["dst_bytes"] = df["dst_bytes"].round().astype(int)
    df["conn_rate"] = df["conn_rate"].clip(lower=0.1).round(2)
    df["payload_entropy"] = df["payload_entropy"].clip(0, 8).round(3)
    df["verdict"] = np.where(df["attack_type"] == "benign", "benign", "malicious")
    # ~3% analyst labelling errors on the binary verdict.
    flip = rng.choice(n, size=int(0.03 * n), replace=False)
    df.loc[flip, "verdict"] = np.where(df.loc[flip, "verdict"] == "benign", "malicious", "benign")
    # Continuous risk score: linear in a few features + a non-linear term + Gaussian noise.
    risk = (
        10
        + 1.2 * df["failed_logins"]
        + 0.25 * df["distinct_ports"]
        + 4.0 * np.log1p(df["src_bytes"]) * (df["payload_entropy"] > 6)
        + 0.3 * df["conn_rate"]
        + 2.0 * (df["hour"].between(0, 5))
        + rng.normal(0, 3, n)
    )
    outliers = rng.choice(n, size=int(0.03 * n), replace=False)
    risk[outliers] += rng.choice([-1, 1], size=len(outliers)) * rng.uniform(40, 80, len(outliers))
    df["risk_score"] = risk.round(2)
    missing = rng.choice(n, size=8, replace=False)
    df.loc[missing, "payload_entropy"] = np.nan
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)  # duplicate rows
    cols = ["duration_s", "src_bytes", "dst_bytes", "packets", "failed_logins", "distinct_ports", "conn_rate",
            "payload_entropy", "hour", "protocol", "attack_type", "verdict", "risk_score"]
    return df[cols]


def auth_sequences(n_sessions: int = 24, length: int = 60) -> pd.DataFrame:
    states = ["normal", "recon", "compromise"]
    symbols = ["login_success", "login_failure", "process_exec", "net_conn", "file_access", "priv_escalation"]
    pi = np.array([0.85, 0.12, 0.03])
    A = np.array([[0.92, 0.07, 0.01], [0.10, 0.75, 0.15], [0.02, 0.05, 0.93]])
    B = np.array([
        [0.35, 0.05, 0.20, 0.20, 0.19, 0.01],
        [0.05, 0.55, 0.05, 0.30, 0.04, 0.01],
        [0.05, 0.05, 0.30, 0.25, 0.20, 0.15],
    ])
    rows = []
    t_global = 0
    level = 120.0
    for s in range(n_sessions):
        z = rng.choice(3, p=pi)
        for step in range(length):
            if step:
                z = rng.choice(3, p=A[z])
            x = rng.choice(len(symbols), p=B[z])
            drift = {0: 0.0, 1: 0.5, 2: 3.0}[z]
            level = max(10.0, level + drift + rng.normal(0, 4))
            rows.append({"t": t_global, "session_id": f"S{s + 1:02d}", "step": step, "event": symbols[x],
                         "phase": states[z], "bytes_out_kb": round(level + rng.normal(0, 15), 2)})
            t_global += 1
    return pd.DataFrame(rows)


if __name__ == "__main__":
    network_intrusion().to_csv(OUT / "network_intrusion.csv", index=False)
    auth_sequences().to_csv(OUT / "auth_sequences.csv", index=False)
    print("Wrote", OUT / "network_intrusion.csv", "and", OUT / "auth_sequences.csv")

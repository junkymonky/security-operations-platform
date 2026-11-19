"""Build the network-attack datasets from KDD Cup 1999 (10% subset, DARPA 1998 traffic).

    python -m sample_data.make_kdd          (from ml-service/; downloads ~2 MB via scikit-learn)

Outputs
    data/kdd_attacks_full.csv        all de-duplicated connections (~145k rows)
    sample_data/kdd_attacks.csv      stratified sample (~20k rows) importable in the website

Cleaning follows the NSL-KDD rationale: the raw 10% file is ~70% exact duplicates (mostly smurf /
neptune floods), which inflates accuracy, so duplicates are removed. Each attack is mapped to its
standard category (DoS, Probe, R2L, U2R). The constant column num_outbound_cmds is dropped.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_kddcup99

ROOT = Path(__file__).resolve().parent.parent
CATEGORY = {
    "normal": "normal",
    **dict.fromkeys(["back", "land", "neptune", "pod", "smurf", "teardrop"], "DoS"),
    **dict.fromkeys(["ipsweep", "nmap", "portsweep", "satan"], "Probe"),
    **dict.fromkeys(["ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy", "warezclient",
                     "warezmaster"], "R2L"),
    **dict.fromkeys(["buffer_overflow", "loadmodule", "perl", "rootkit"], "U2R"),
}
# Rows kept per attack label in the sample; rare labels are kept in full.
SAMPLE_CAP = {"normal": 9000, "neptune": 3000, "smurf": 1500, "back": 900, "satan": 900,
              "ipsweep": 900, "portsweep": 900, "warezclient": 900, "teardrop": 900}


def build(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = fetch_kddcup99(percent10=True, as_frame=True, data_home=str(ROOT / "data" / "kddcup99_cache")).frame
    df = raw.copy()
    for col in ["protocol_type", "service", "flag", "labels"]:
        df[col] = df[col].map(lambda v: v.decode() if isinstance(v, bytes) else str(v))
    df["attack_type"] = df.pop("labels").str.rstrip(".")
    unknown = sorted(set(df["attack_type"]) - set(CATEGORY))
    if unknown:
        raise ValueError(f"Unmapped attack labels: {unknown}")
    df["attack_category"] = df["attack_type"].map(CATEGORY)
    df["is_attack"] = np.where(df["attack_type"] == "normal", "normal", "attack")
    numeric = [c for c in df.columns if c not in ("protocol_type", "service", "flag", "attack_type",
                                                  "attack_category", "is_attack")]
    df[numeric] = df[numeric].apply(pd.to_numeric)
    df = df.drop(columns=["num_outbound_cmds"])  # always 0 in this subset
    full = df.drop_duplicates().reset_index(drop=True)

    parts = []
    for label, g in full.groupby("attack_type"):
        cap = SAMPLE_CAP.get(label, len(g))
        parts.append(g.sample(n=min(cap, len(g)), random_state=seed))
    sample = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return full, sample


if __name__ == "__main__":
    full, sample = build()
    (ROOT / "data").mkdir(exist_ok=True)
    full.to_csv(ROOT / "data" / "kdd_attacks_full.csv", index=False)
    sample.to_csv(ROOT / "sample_data" / "kdd_attacks.csv", index=False)
    print(f"full: {len(full):,} rows -> data/kdd_attacks_full.csv")
    print(f"sample: {len(sample):,} rows -> sample_data/kdd_attacks.csv")
    print(sample["attack_category"].value_counts().to_string())

"""Turn ML predictions on network flows into SOC investigations (incidents + events).

A random forest classifies each flow in the network_intrusion sample by attack type using
5-fold out-of-fold predictions (each flow is scored by a model that never saw it). Flows
predicted as malicious are grouped by (attack type, target host) into incidents.

The sample flows carry no hosts/IPs/timestamps, so a deterministic *simulated* network context
(seeded) is attached to each flow; this is stated in every incident summary.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .. import db
from ..config import settings
from ..errors import ValidationFailed
from ..utils import to_jsonable

SOURCE = "ML detection"
MODEL_NAME = "rf-attack-type (5-fold out-of-fold)"
NUMERIC = ["duration_s", "src_bytes", "dst_bytes", "packets", "failed_logins", "distinct_ports",
           "conn_rate", "payload_entropy", "hour"]
PROFILE = {
    # attack type: title, severity, MITRE technique, event type, typical destination ports
    "brute_force": ("Brute-force authentication attack", "high", "T1110", "AUTH_FAILURE", [22, 3389, 445]),
    "port_scan": ("Network port scan", "medium", "T1046", "NETWORK_CONNECTION", [21, 23, 135, 139, 1433, 3306, 8080]),
    "exfiltration": ("Possible data exfiltration", "critical", "T1048", "NETWORK_CONNECTION", [443, 8443]),
    "c2_beacon": ("Command-and-control beaconing", "high", "T1071.001", "NETWORK_CONNECTION", [443, 80]),
}
HOSTS = ["WIN-SRV-01", "WIN-SRV-02", "LINUX-SRV-02", "LINUX-WEB-01", "WIN-DEV-04", "WIN-FIN-03", "DB-PROD-01", "FW-01"]
USERS = ["john.smith", "admin", "james", "svc_backup", "a.kumar", "m.lee", "root", "s.patel"]
EXTERNAL = ["185.225.73.14", "45.137.21.9", "91.240.118.172", "193.142.146.35", "103.75.190.11", "5.188.206.14"]


def _context(n: int, attack: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """Simulated, reproducible network context for each flow."""
    hosts = rng.choice(HOSTS, n)
    # Malicious flows concentrate on a couple of targets so incidents group naturally.
    for a in PROFILE:
        idx = np.flatnonzero(attack == a)
        targets = rng.choice(HOSTS, 2, replace=False)
        hosts[idx] = rng.choice(targets, len(idx), p=[0.7, 0.3])
    ports = np.array([rng.choice(PROFILE[a][4]) if a in PROFILE else rng.choice([443, 80, 53]) for a in attack])
    src = np.where(attack != "benign", rng.choice(EXTERNAL, n), [f"10.24.{rng.integers(1, 30)}.{rng.integers(2, 250)}" for _ in range(n)])
    return pd.DataFrame({
        "hostname": hosts, "username": rng.choice(USERS, n), "source_ip": src,
        "destination_ip": [f"10.24.8.{40 + HOSTS.index(h)}" for h in hosts], "destination_port": ports,
    })


def _score(df: pd.DataFrame):
    features = NUMERIC + ["protocol"]
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["protocol"])], remainder="passthrough")
    model = Pipeline([("pre", pre), ("rf", RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1))])
    X = df[features].copy()
    X[NUMERIC] = X[NUMERIC].fillna(X[NUMERIC].median())
    y = df["attack_type"].astype(str).to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
    classes = np.array(sorted(set(y)))
    pred = classes[proba.argmax(1)]
    return pred, proba.max(1), y, float((pred == y).mean())


def _delete_previous(conn) -> int:
    ids = [r["id"] for r in conn.execute("SELECT id FROM incidents WHERE detection_source LIKE %s", (SOURCE + "%",))]
    if ids:
        conn.execute("DELETE FROM events WHERE incident_id = ANY(%s)", (ids,))
        conn.execute("DELETE FROM incidents WHERE id = ANY(%s)", (ids,))
    conn.execute("DELETE FROM events WHERE id LIKE 'EVT-ML-%%' AND incident_id IS NULL")
    return len(ids)


def generate_incidents(max_events_per_incident: int = 25, min_flows: int = 3) -> dict:
    path = settings.samples_dir / "network_intrusion.csv"
    if not path.is_file():
        raise ValidationFailed("Sample network_intrusion.csv not found; run python -m sample_data.generate")
    df = pd.read_csv(path).drop_duplicates().reset_index(drop=True)
    pred, conf, truth, accuracy = _score(df)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    rng = np.random.default_rng(7)
    ctx = _context(len(df), pred, rng)
    flows = pd.concat([df, ctx], axis=1).assign(predicted=pred, confidence=conf, truth=truth)
    flagged = flows[flows["predicted"] != "benign"]

    created = []
    with db.connection() as conn:
        with conn.transaction():  # all-or-nothing replace of the ML incidents
            removed = _delete_previous(conn)
            (start,) = conn.execute(
                "SELECT coalesce(max(NULLIF(regexp_replace(id, '\\D', '', 'g'), '')::int), 1000) AS n FROM incidents"
            ).fetchone().values()
            number = max(int(start), 3000)
            event_no = 0
            for (attack, host), group in flagged.groupby(["predicted", "hostname"]):
                if len(group) < min_flows:
                    continue
                title, severity, technique, event_type, _ = PROFILE[attack]
                number += 1
                inc_id = f"INC-{number}"
                # Each attack is a burst: flows within ~30 minutes, detected a few minutes after the last one.
                burst_start = now - timedelta(hours=float(rng.uniform(0.5, 72)))
                offsets = np.sort(rng.uniform(0, 30 * 60, len(group)))
                group = group.assign(timestamp=[burst_start + timedelta(seconds=float(o)) for o in offsets])
                detected_at = group["timestamp"].max() + timedelta(minutes=float(rng.uniform(1, 8)))
                group = group.sort_values("confidence", ascending=False)
                mean_conf = float(group["confidence"].mean())
                agree = int((group["truth"] == attack).sum())
                risk = int(min(99, round(mean_conf * {"critical": 100, "high": 90, "medium": 75}[severity])))
                first_seen = group["timestamp"].min()
                summary = (f"{len(group)} network flows to {host} were classified as {attack.replace('_', ' ')} by the "
                           f"random forest (mean confidence {mean_conf:.0%}). Ground-truth labels in the sample agree for "
                           f"{agree}/{len(group)} flows. Hosts, users, IPs and times are simulated context for the "
                           f"sample dataset.")
                conn.execute(
                    """INSERT INTO incidents (id, title, severity, status, risk_score, ai_confidence, detected_at, summary,
                       detection_source, detection_model, correlated_event_count, affected_entity_count)
                       VALUES (%s,%s,%s,'new',%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (inc_id, f"{title} on {host}", severity, risk, round(mean_conf * 100, 2),
                     detected_at, summary, f"{SOURCE} (random forest)", MODEL_NAME, len(group),
                     int(1 + group["source_ip"].nunique() + group["username"].nunique())),
                )
                for _, f in group.head(max_events_per_incident).iterrows():
                    event_no += 1
                    raw = to_jsonable({**{k: f[k] for k in NUMERIC + ["protocol"]}, "predicted": attack,
                                       "confidence": f["confidence"], "true_label": f["truth"],
                                       "direction": "inbound" if attack in ("brute_force", "port_scan") else "outbound"})
                    conn.execute(
                        """INSERT INTO events (id, incident_id, timestamp, event_type, source, username, hostname, source_ip,
                           destination_ip, destination_port, severity, detection_method, mitre_technique, raw_event)
                           VALUES (%s,%s,%s,%s,'Network sensor',%s,%s,%s,%s,%s,%s,'ml_classifier',%s,%s)""",
                        (f"EVT-ML-{event_no:04d}", inc_id, f["timestamp"], event_type, f["username"], host,
                         f["source_ip"], f["destination_ip"], int(f["destination_port"]),
                         "critical" if f["confidence"] > 0.95 and severity == "critical" else
                         "high" if f["confidence"] > 0.8 else "medium",
                         technique, json.dumps(raw)),
                    )
                conn.execute("INSERT INTO audit_logs (incident_id, actor, action, details) VALUES (%s,'ml-detector',"
                             "'incident_created',%s)", (inc_id, json.dumps({"severity": severity, "flows": len(group)})))
                created.append({"id": inc_id, "title": f"{title} on {host}", "severity": severity, "flows": len(group),
                                "mean_confidence": mean_conf, "label_agreement": agree / len(group),
                                "first_seen": first_seen.isoformat()})
    return {"model": MODEL_NAME, "flows_scored": int(len(df)), "out_of_fold_accuracy": accuracy,
            "flows_flagged": int(len(flagged)), "incidents_created": created, "previous_ml_incidents_removed": removed}

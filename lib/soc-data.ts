// Server-side queries for the SOC console. Every value shown on the dashboards is
// computed here from the incidents / events / reference tables; nothing is hardcoded.
import { sql } from "@/lib/db";
import { analystProfile } from "@/lib/profile";

export const ACTIVE_STATUSES = ["new", "investigating", "contained"];
export const ALERT_SEVERITIES = ["high", "critical"];

export type Range = "1h" | "6h" | "24h" | "7d" | "all";
export type SeriesMetric = "events" | "detections" | "alerts" | "incidents" | "critical" | "ml";

const RANGE_SECONDS: Record<Exclude<Range, "all">, number> = { "1h": 3600, "6h": 21600, "24h": 86400, "7d": 604800 };

export { analystProfile } from "@/lib/profile";

export async function listIncidents() {
  return sql`
    SELECT i.id, i.title, i.severity, i.status, i.risk_score, i.ai_confidence, i.detected_at,
           i.assigned_analyst, i.summary, i.detection_source, i.detection_model,
           i.correlated_event_count, i.affected_entity_count,
           (SELECT e.username FROM events e WHERE e.incident_id = i.id AND e.username IS NOT NULL
              GROUP BY e.username ORDER BY count(*) DESC LIMIT 1) AS primary_user,
           (SELECT e.hostname FROM events e WHERE e.incident_id = i.id AND e.hostname IS NOT NULL
              GROUP BY e.hostname ORDER BY count(*) DESC LIMIT 1) AS primary_host
    FROM incidents i
    ORDER BY i.risk_score DESC, i.detected_at DESC
  `;
}

export async function listEvents(opts: { severities?: string[]; limit?: number } = {}) {
  const limit = Math.min(Math.max(opts.limit ?? 500, 1), 2000);
  if (opts.severities?.length) {
    return sql`
      SELECT id, incident_id, timestamp, event_type, source, username, hostname, host(source_ip) AS source_ip,
             host(destination_ip) AS destination_ip, destination_port, process_name, command_line, severity,
             detection_method, mitre_technique
      FROM events WHERE lower(severity) = ANY(${opts.severities}) ORDER BY timestamp DESC LIMIT ${limit}`;
  }
  return sql`
    SELECT id, incident_id, timestamp, event_type, source, username, hostname, host(source_ip) AS source_ip,
           host(destination_ip) AS destination_ip, destination_port, process_name, command_line, severity,
           detection_method, mitre_technique
    FROM events ORDER BY timestamp DESC LIMIT ${limit}`;
}

/** Headline metrics with the change versus the previous 24 hours. */
export async function overviewMetrics() {
  const [m] = await sql`
    SELECT
      (SELECT count(*) FROM events WHERE lower(severity) = 'critical')::int AS critical_alerts,
      (SELECT count(*) FROM events WHERE lower(severity) = 'critical' AND timestamp >= now() - interval '24 hours')::int AS critical_alerts_24h,
      (SELECT count(*) FROM events WHERE lower(severity) = ANY(${ALERT_SEVERITIES}))::int AS alerts,
      (SELECT count(*) FROM incidents WHERE status = ANY(${ACTIVE_STATUSES}))::int AS active_incidents,
      (SELECT count(*) FROM incidents WHERE status = 'new')::int AS new_incidents,
      (SELECT count(*) FROM incidents WHERE detected_at >= now() - interval '24 hours')::int AS incidents_24h,
      (SELECT count(*) FROM incidents)::int AS total_incidents,
      (SELECT count(*) FROM events)::int AS events_total,
      (SELECT count(*) FROM events WHERE timestamp >= now() - interval '24 hours')::int AS events_24h,
      (SELECT count(*) FROM events WHERE timestamp >= now() - interval '48 hours'
                                    AND timestamp <  now() - interval '24 hours')::int AS events_prev_24h,
      (SELECT max(created_at) FROM events) AS last_event_ingested_at,
      (SELECT avg(EXTRACT(EPOCH FROM (i.detected_at - f.first_seen)))
         FROM incidents i
         JOIN (SELECT incident_id, min(timestamp) AS first_seen FROM events GROUP BY incident_id) f
           ON f.incident_id = i.id
        WHERE i.detected_at >= f.first_seen) AS mttd_seconds,
      (SELECT count(DISTINCT i.id) FROM incidents i JOIN events e ON e.incident_id = i.id)::int AS mttd_sample
  `;
  return m;
}

/** Activity time series for the overview chart: the chosen metric plus critical-severity activity. */
export async function activitySeries(range: Range, metric: SeriesMetric, buckets = 24) {
  let start: Date, end: Date;
  if (range === "all") {
    const [b] = await sql`
      SELECT least((SELECT min(timestamp) FROM events), (SELECT min(detected_at) FROM incidents)) AS lo,
             greatest((SELECT max(timestamp) FROM events), (SELECT max(detected_at) FROM incidents)) AS hi`;
    if (!b.lo) return { start: null, end: null, bucketSeconds: 0, points: [] };
    start = new Date(b.lo);
    end = new Date(new Date(b.hi).getTime() + 1000);
  } else {
    end = new Date();
    start = new Date(end.getTime() - RANGE_SECONDS[range] * 1000);
  }
  const span = Math.max(end.getTime() - start.getTime(), 1000);
  const bucketMs = Math.ceil(span / buckets);

  const rows =
    metric === "incidents"
      ? await sql`SELECT detected_at AS ts, lower(severity) AS severity, NULL AS detection_method FROM incidents
                  WHERE detected_at >= ${start.toISOString()} AND detected_at < ${end.toISOString()}`
      : await sql`SELECT timestamp AS ts, lower(severity) AS severity, detection_method FROM events
                  WHERE timestamp >= ${start.toISOString()} AND timestamp < ${end.toISOString()}`;

  const keep = (r: any) =>
    metric === "events" || metric === "incidents" ? true
    : metric === "detections" ? !!r.detection_method
    : metric === "alerts" ? ALERT_SEVERITIES.includes(r.severity)
    : metric === "critical" ? r.severity === "critical"
    : /ml|anomal|model/i.test(r.detection_method || "");

  const points = Array.from({ length: buckets }, (_, i) => ({
    t: new Date(start.getTime() + i * bucketMs).toISOString(), v: 0, critical: 0,
  }));
  for (const r of rows) {
    const i = Math.min(Math.floor((new Date(r.ts).getTime() - start.getTime()) / bucketMs), buckets - 1);
    if (i < 0) continue;
    if (keep(r)) points[i].v += 1;
    if (r.severity === "critical") points[i].critical += 1;
  }
  return { start: start.toISOString(), end: end.toISOString(), bucketSeconds: bucketMs / 1000, points };
}

/** MITRE ATT&CK techniques observed in events, joined to reference names where available. */
export async function techniqueSummary(incidentId?: string) {
  if (incidentId) {
    return sql`
      SELECT e.mitre_technique AS id, t.name, t.tactic, count(*)::int AS evidence_count,
             im.confidence, min(e.timestamp) AS first_seen
      FROM events e
      LEFT JOIN mitre_techniques t ON t.id = e.mitre_technique
      LEFT JOIN incident_mitre im ON im.incident_id = e.incident_id AND im.technique_id = e.mitre_technique
      WHERE e.incident_id = ${incidentId} AND e.mitre_technique IS NOT NULL
      GROUP BY e.mitre_technique, t.name, t.tactic, im.confidence
      ORDER BY min(e.timestamp)`;
  }
  return sql`
    SELECT e.mitre_technique AS id, t.name, t.tactic, count(*)::int AS evidence_count,
           count(DISTINCT e.incident_id)::int AS incident_count, max(e.timestamp) AS last_seen
    FROM events e LEFT JOIN mitre_techniques t ON t.id = e.mitre_technique
    WHERE e.mitre_technique IS NOT NULL
    GROUP BY e.mitre_technique, t.name, t.tactic
    ORDER BY count(*) DESC`;
}

export async function databaseLatencyMs() {
  const t = Date.now();
  await sql`SELECT 1`;
  return Date.now() - t;
}

export async function writeAudit(incidentId: string, actor: string, action: string, details: Record<string, unknown> = {}) {
  await sql`INSERT INTO audit_logs (incident_id, actor, action, details)
            VALUES (${incidentId}, ${actor}, ${action}, ${JSON.stringify(details)}::jsonb)`;
}

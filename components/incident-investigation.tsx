"use client";

import { useMemo, useState } from "react";
import {
  ArrowLeft,
  Check,
  ChevronDown,
  Download,
  ExternalLink,
  MoreHorizontal,
  Plus,
  ShieldAlert,
  UserRound,
  Server,
  Globe2,
  Terminal,
  X,
} from "lucide-react";

export interface Incident {
  id: string;
  title: string;
  severity: string;
  status: string;
  risk_score: number;
  ai_confidence: number | string | null;
  detected_at: string;
  assigned_analyst: string | null;
  summary: string | null;
  detection_source: string | null;
  correlated_event_count: number;
  affected_entity_count: number;
  detection_model: string | null;
}

export interface Event {
  id: string;
  timestamp: string;
  event_type: string;
  source: string;
  username: string | null;
  hostname: string | null;
  source_ip: string | null;
  destination_ip: string | null;
  destination_port: number | null;
  severity: string | null;
  process_name: string | null;
  command_line: string | null;
  detection_method: string | null;
  mitre_technique: string | null;
  raw_event?: Record<string, any> | null;
}

export interface Entity {
  id: string;
  type: string;
  name: string;
  metadata: string[];
}

export interface MITRETechnique {
  id: string;
  name: string | null;
  tactic: string | null;
  confidence: number | string | null;
  evidence_count: number;
}

export interface InvestigationNote {
  id: number | string;
  analyst: string;
  note: string;
  created_at: string;
}

export interface AuditEntry {
  id: number | string;
  actor: string;
  action: string;
  details: Record<string, any> | null;
  created_at: string;
}

// Status values as stored in the database, with display labels.
const STATUSES: [string, string][] = [
  ["new", "New"],
  ["investigating", "Investigating"],
  ["contained", "Contained"],
  ["resolved", "Resolved"],
  ["false_positive", "False Positive"],
];

// IANA well-known ports, used to label network evidence when the raw event has no protocol.
const WELL_KNOWN_PORTS: Record<number, string> = {
  20: "FTP-DATA", 21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP", 53: "DNS", 80: "HTTP", 88: "Kerberos",
  110: "POP3", 123: "NTP", 135: "MS-RPC", 139: "NetBIOS", 143: "IMAP", 389: "LDAP", 443: "HTTPS",
  445: "SMB", 636: "LDAPS", 993: "IMAPS", 995: "POP3S", 1433: "MSSQL", 3306: "MySQL", 3389: "RDP",
  5432: "PostgreSQL", 5985: "WinRM", 5986: "WinRM-TLS", 8080: "HTTP-ALT", 8443: "HTTPS-ALT",
};

// Deterministic date formatting to prevent SSR hydration mismatches
function formatDateTime(value: string) {
  const date = new Date(value);

  return (
    new Intl.DateTimeFormat("en-GB", {
      day: "2-digit",
      month: "2-digit",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
      timeZone: "UTC",
    }).format(date) + " UTC"
  );
}

// Deterministic time formatting for event timelines
function formatTime(value: string) {
  const date = new Date(value);

  return (
    new Intl.DateTimeFormat("en-GB", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
      timeZone: "UTC",
    }).format(date) + " UTC"
  );
}

const label = (value: string) => value.replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());
const percent = (value: number | string | null) =>
  value === null || value === undefined ? "—" : `${Math.round(Number(value))}%`;

function isPrivateIp(ip: string) {
  return /^(10\.|127\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|169\.254\.|::1|f[cd])/i.test(ip);
}

function protocolOf(event: Event) {
  const raw = event.raw_event || {};
  if (raw.protocol) return String(raw.protocol).toUpperCase();
  if (event.destination_port && WELL_KNOWN_PORTS[event.destination_port]) return WELL_KNOWN_PORTS[event.destination_port];
  return "—";
}

function directionOf(event: Event) {
  const raw = event.raw_event || {};
  if (raw.direction) return label(String(raw.direction));
  if (!event.destination_ip) return "—";
  const srcPrivate = event.source_ip ? isPrivateIp(event.source_ip) : true;
  const dstPrivate = isPrivateIp(event.destination_ip);
  if (srcPrivate && !dstPrivate) return "Outbound";
  if (!srcPrivate && dstPrivate) return "Inbound";
  return "Internal";
}

const AUDIT_TEXT: Record<string, (d: any) => string> = {
  incident_created: (d) => `Incident created (${d?.severity ?? ""} severity)`,
  status_changed: (d) => `Status changed from ${label(d?.from ?? "—")} to ${label(d?.to ?? "—")}`,
  owner_changed: (d) => `Ownership changed from ${d?.from ?? "unassigned"} to ${d?.to ?? "unassigned"}`,
  note_added: () => "Investigation note added",
  action_requested: (d) => `Requested: ${d?.action ?? "action"}`,
  action_confirmed: (d) => `Confirmed: ${d?.action ?? "action"}`,
  incident_exported: (d) => `Incident exported (${d?.format ?? "json"})`,
};

async function call<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  const body = await res.json().catch(() => null);
  if (!res.ok || body?.success === false) throw new Error(body?.error || `Request failed (${res.status})`);
  return body?.data as T;
}

function Pill({
  children,
  tone = "muted",
}: {
  children: React.ReactNode;
  tone?: string;
}) {
  return (
    <span className={`pill pill-${tone.toLowerCase()}`}>
      {children}
    </span>
  );
}

function Section({
  title,
  children,
  action,
}: {
  title: string;
  children: React.ReactNode;
  action?: React.ReactNode;
}) {
  return (
    <section className="investigation-section">
      <div className="investigation-section-head">
        <h2>{title}</h2>
        {action}
      </div>

      {children}
    </section>
  );
}

interface IncidentInvestigationProps {
  incident: Incident;
  events: Event[];
  techniques: MITRETechnique[];
  initialNotes: InvestigationNote[];
  initialAudit: AuditEntry[];
  analyst: string;
}

export default function IncidentInvestigation({
  incident,
  events,
  techniques,
  initialNotes,
  initialAudit,
  analyst,
}: IncidentInvestigationProps) {
  const [status, setStatus] = useState(incident.status);
  const [owner, setOwner] = useState(incident.assigned_analyst || "Unassigned");
  const [selectedEvent, setSelectedEvent] = useState<Event | null>(
    events.length > 0 ? events[events.length - 1] : null
  );
  const [tab, setTab] = useState("Overview");
  const [notes, setNotes] = useState<InvestigationNote[]>(initialNotes);
  const [audit, setAudit] = useState<AuditEntry[]>(initialAudit);
  const [note, setNote] = useState("");
  const [toast, setToast] = useState("");
  const [modal, setModal] = useState("");
  const [busy, setBusy] = useState(false);

  const notify = (message: string) => {
    setToast(message);
    window.setTimeout(() => setToast(""), 2600);
  };

  const base = `/api/incidents/${encodeURIComponent(incident.id)}`;
  const refreshAudit = async () => {
    try {
      setAudit(await call<AuditEntry[]>(`${base}/audit`));
    } catch {
      /* the action itself succeeded; the list refreshes on next load */
    }
  };

  const updateIncident = async (patch: { status?: string; assigned_analyst?: string }, success: string) => {
    setBusy(true);
    try {
      const row = await call<{ status: string; assigned_analyst: string | null }>(base, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patch),
      });
      setStatus(row.status);
      setOwner(row.assigned_analyst || "Unassigned");
      notify(success);
      refreshAudit();
    } catch (error: any) {
      notify(`Update failed: ${error.message}`);
    } finally {
      setBusy(false);
    }
  };

  const recordAction = async (action: "action_requested" | "action_confirmed", what: string) => {
    try {
      await call(`${base}/audit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, details: { action: what } }),
      });
      notify(`${what} recorded in the audit log`);
      refreshAudit();
    } catch (error: any) {
      notify(`Could not record action: ${error.message}`);
    }
  };

  const exportIncident = async () => {
    const payload = { exported_at: new Date().toISOString(), incident: { ...incident, status, assigned_analyst: owner }, events, techniques, notes };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${incident.id}.json`;
    a.click();
    URL.revokeObjectURL(url);
    try {
      await call(`${base}/audit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "incident_exported", details: { format: "json", events: events.length } }),
      });
      refreshAudit();
    } catch {
      /* export already downloaded */
    }
    notify(`Exported ${incident.id}.json`);
  };

  const saveNote = async () => {
    if (!note.trim()) return;
    setBusy(true);
    try {
      const saved = await call<InvestigationNote>(`${base}/notes`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ note, analyst: analyst || (owner !== "Unassigned" ? owner : "") }),
      });
      setNotes([saved, ...notes]);
      setNote("");
      notify("Note saved");
      refreshAudit();
    } catch (error: any) {
      notify(`Note not saved: ${error.message}`);
    } finally {
      setBusy(false);
    }
  };

  /*
   * Entities derived from the incident's events, with facts computed from those events.
   */
  const entities = useMemo<Entity[]>(() => {
    const groups: [string, keyof Event][] = [
      ["User", "username"],
      ["Host", "hostname"],
      ["Source IP", "source_ip"],
      ["Destination IP", "destination_ip"],
      ["Process", "process_name"],
    ];
    const result: Entity[] = [];
    for (const [type, key] of groups) {
      const seen = new Map<string, Event[]>();
      events.forEach((event) => {
        const value = event[key];
        if (!value || value === "—") return;
        seen.set(String(value), [...(seen.get(String(value)) || []), event]);
      });
      seen.forEach((list, name) => {
        const kinds = Array.from(new Set(list.map((e) => label(e.event_type.toLowerCase()))));
        result.push({
          id: `${type}-${name}`,
          type,
          name,
          metadata: [
            `${list.length} event${list.length > 1 ? "s" : ""} · ${kinds.join(", ")}`,
            `First seen ${formatTime(list[0].timestamp)}`,
          ],
        });
      });
    }
    return result;
  }, [events]);

  /*
   * Analysis derived from the incident record and its events.
   */
  const analysis = useMemo(() => {
    const byType = new Map<string, Event[]>();
    events.forEach((e) => byType.set(e.event_type, [...(byType.get(e.event_type) || []), e]));
    const evidence = Array.from(byType.entries()).map(([type, list]) => {
      const extra = [
        ...new Set(list.map((e) => e.process_name || e.source_ip || e.destination_ip || e.username).filter(Boolean)),
      ];
      const methods = [...new Set(list.map((e) => e.detection_method).filter(Boolean))];
      return `${list.length} × ${label(type.toLowerCase())}${extra.length ? ` (${extra.join(", ")})` : ""}${
        methods.length ? ` · ${methods.join(", ")}` : ""
      }`;
    });

    const has = (type: string) => byType.has(type);
    const users = [...new Set(events.map((e) => e.username).filter(Boolean))] as string[];
    const hosts = [...new Set(events.map((e) => e.hostname).filter(Boolean))] as string[];
    const srcIps = [...new Set(events.map((e) => e.source_ip).filter(Boolean))] as string[];
    const dstIps = [...new Set(events.map((e) => e.destination_ip).filter(Boolean))] as string[];
    const procs = [...new Set(events.map((e) => e.process_name).filter(Boolean))] as string[];

    const recommendations: string[] = [];
    srcIps.forEach((ip) => recommendations.push(`Investigate source IP ${ip}`));
    users.forEach((u) => recommendations.push(`Review user activity for ${u}`));
    procs.forEach((p) => recommendations.push(`Review process tree for ${p}`));
    dstIps.forEach((ip) => recommendations.push(`Check host connections to ${ip}`));
    if (has("AUTH_SUCCESS") && (has("AUTH_FAILURE") || srcIps.length)) {
      users.forEach((u) => recommendations.push(`Revoke active sessions for ${u}`));
    }
    if (procs.length || dstIps.length) hosts.forEach((h) => recommendations.push(`Isolate host ${h}`));

    const coverage: [string, number][] = [
      ["Authentication events", (byType.get("AUTH_FAILURE")?.length || 0) + (byType.get("AUTH_SUCCESS")?.length || 0)],
      ["Process executions", byType.get("PROCESS_EXECUTION")?.length || 0],
      ["Network connections", events.filter((e) => e.destination_ip).length],
      ["Distinct source IPs", srcIps.length],
      ["MITRE techniques mapped", techniques.length],
    ];
    return { evidence, recommendations, coverage };
  }, [events, techniques]);

  const rows = useMemo(() => {
    if (tab === "Processes") {
      return events.filter(
        (event) =>
          event.event_type === "PROCESS_EXECUTION" ||
          Boolean(event.process_name)
      );
    }

    if (tab === "Network") {
      return events.filter(
        (event) =>
          event.event_type === "NETWORK_CONNECTION" ||
          Boolean(event.destination_ip)
      );
    }

    if (tab === "Authentication") {
      return events.filter(
        (event) =>
          event.event_type === "AUTH_FAILURE" ||
          event.event_type === "AUTH_SUCCESS"
      );
    }

    if (tab === "DNS") {
      return events.filter(
        (event) => event.event_type === "DNS_QUERY"
      );
    }

    return events;
  }, [tab, events]);

  const severityTone = (severity: string | null) => {
    if (!severity) return "muted";

    return severity.toLowerCase().replace(" ", "-");
  };

  const statusLabel = STATUSES.find(([value]) => value === status)?.[1] ?? label(status);
  const headers =
    tab === "Processes"
      ? ["Time", "Process", "PID", "Parent Process", "User", "Command", "Risk"]
      : tab === "Network"
        ? ["Timestamp", "Source", "Destination", "Port", "Protocol", "Direction", "Risk"]
        : ["Timestamp", "Event Type", "Source", "User", "Host", "IP", "Severity"];
  const critical = ["Isolate host", "Revoke active sessions"];

  return (
    <div className="investigation-page">

      {/* HEADER */}

      <div className="investigation-top">
        <div>
          <button
            className="back-link"
            onClick={() => {
              window.location.href = "/";
            }}
          >
            <ArrowLeft size={14} />
            Incidents
          </button>

          <div className="breadcrumb">
            Incidents / <b>{incident.id}</b>
          </div>

          <div className="investigation-title">
            <h1>{incident.title}</h1>

            <span>{incident.id}</span>

            <Pill tone={severityTone(incident.severity)}>
              {incident.severity.toUpperCase()}
            </Pill>

            <Pill tone={status === "new" ? "new" : "muted"}>
              {statusLabel.toUpperCase()}
            </Pill>
          </div>
        </div>

        <div className="investigation-stats">
          <div>
            <span>Risk score</span>
            <b>{incident.risk_score} / 100</b>
          </div>

          <div>
            <span>AI confidence</span>
            <b>{percent(incident.ai_confidence)}</b>
          </div>

          <div>
            <span>Detected</span>
            <b>{formatDateTime(incident.detected_at)}</b>
          </div>

          <div>
            <span>Assigned analyst</span>
            <b>{owner}</b>
          </div>
        </div>
      </div>

      {/* ACTIONS */}

      <div className="investigation-actions">
        <button
          className="filter"
          disabled={busy || !analyst || owner === analyst}
          title={!analyst ? "Set ANALYST_NAME in .env.local" : undefined}
          onClick={() => updateIncident({ assigned_analyst: analyst }, "Investigation ownership assigned")}
        >
          <UserRound size={14} />
          {owner === analyst ? "You own this" : "Take ownership"}
        </button>

        <label className="select-action">
          Change status

          <select
            value={status}
            disabled={busy}
            onChange={(event) =>
              updateIncident({ status: event.target.value }, `Status changed to ${label(event.target.value)}`)
            }
          >
            {STATUSES.map(([value, text]) => (
              <option key={value} value={value}>
                {text}
              </option>
            ))}
          </select>

          <ChevronDown size={13} />
        </label>

        <button
          className="filter"
          onClick={() =>
            document
              .getElementById("notes")
              ?.scrollIntoView({
                behavior: "smooth",
              })
          }
        >
          <Plus size={14} />
          Add note
        </button>

        <button className="filter" onClick={exportIncident}>
          <Download size={14} />
          Export
        </button>

        <button className="filter">
          <MoreHorizontal size={14} />
          More
        </button>

        <button
          className="primary"
          disabled={busy || status === "resolved"}
          onClick={() => updateIncident({ status: "resolved" }, "Incident marked resolved")}
        >
          {status === "resolved" ? "Resolved" : "Mark resolved"}
        </button>
      </div>

      {/* SUMMARY */}

      <Section title="Investigation Summary">
        <div className="summary-grid">
          <p>{incident.summary || "No summary recorded for this incident."}</p>

          <div className="summary-facts">
            {[
              ["Detection source", incident.detection_source || "—"],
              ["Risk", `${incident.risk_score}/100`],
              ["AI confidence", percent(incident.ai_confidence)],
              ["Correlated events", String(incident.correlated_event_count)],
              ["Affected entities", String(incident.affected_entity_count)],
              ["Detection model", incident.detection_model || "—"],
            ].map(([key, value]) => (
              <div key={key}>
                <span>{key}</span>
                <b>{value}</b>
              </div>
            ))}
          </div>
        </div>
      </Section>

      {/* TIMELINE + ENTITIES */}

      <div className="investigation-columns">
        <Section title="Attack Timeline">
          <p className="section-subtitle">
            Chronological sequence of correlated security events.
          </p>

          <div className="timeline">
            {events.length === 0 && <p className="section-subtitle">No events are linked to this incident.</p>}
            {events.map((event) => (
              <button
                className={`timeline-event ${selectedEvent?.id === event.id ? "selected" : ""}`}
                key={event.id}
                onClick={() => setSelectedEvent(event)}
              >
                <time>{formatTime(event.timestamp)}</time>

                <i className={severityTone(event.severity)} />

                <div>
                  <b>{event.event_type.replaceAll("_", " ")}</b>

                  <span>
                    {event.username
                      ? `User: ${event.username}`
                      : event.process_name
                        ? `Process: ${event.process_name}`
                        : event.hostname
                          ? `Host: ${event.hostname}`
                          : "Security event"}
                  </span>

                  <small>
                    {event.source_ip
                      ? `Source: ${event.source_ip}`
                      : event.destination_ip
                        ? `Destination: ${event.destination_ip}`
                        : event.hostname || event.source}
                  </small>
                </div>
              </button>
            ))}
          </div>
        </Section>

        <Section title="Entities Involved">
          <div className="entities">
            {entities.length === 0 && <p className="section-subtitle">No entities recorded on this incident's events.</p>}
            {entities.map((entity) => (
              <button
                className="entity-row"
                key={entity.id}
                onClick={() => {
                  const first = events.find((e) => Object.values(e).includes(entity.name));
                  if (first) setSelectedEvent(first);
                }}
                title="Show the first event involving this entity"
              >
                {entity.type === "User" ? (
                  <UserRound size={15} />
                ) : entity.type === "Host" ? (
                  <Server size={15} />
                ) : entity.type.endsWith("IP") ? (
                  <Globe2 size={15} />
                ) : (
                  <Terminal size={15} />
                )}

                <span>
                  <b>{entity.name}</b>
                  <small>{entity.type}</small>
                  {entity.metadata.map((value) => (
                    <small key={value}>{value}</small>
                  ))}
                </span>

                <ExternalLink size={12} />
              </button>
            ))}
          </div>
        </Section>
      </div>

      {/* EVENT DETAILS */}

      {selectedEvent && (
        <Section
          title="Event Details"
          action={
            <button className="icon-btn" onClick={() => setSelectedEvent(null)}>
              <X size={15} />
            </button>
          }
        >
          <div className="event-details">
            {[
              ["Event ID", selectedEvent.id],
              ["Timestamp", formatDateTime(selectedEvent.timestamp)],
              ["Event type", selectedEvent.event_type],
              ["Source", selectedEvent.source],
              ["User", selectedEvent.username || "—"],
              ["Host", selectedEvent.hostname || "—"],
              ["Source IP", selectedEvent.source_ip || "—"],
              ["Destination IP", selectedEvent.destination_ip || "—"],
              ["Process", selectedEvent.process_name || "—"],
              ["Command", selectedEvent.command_line || "—"],
              ["Detection", selectedEvent.detection_method || "—"],
              ["Severity", selectedEvent.severity || "—"],
              ["MITRE", selectedEvent.mitre_technique || "—"],
            ].map(([key, value]) => (
              <div key={key}>
                <span>{key}</span>
                <b>{value}</b>
              </div>
            ))}
          </div>
        </Section>
      )}

      {/* EVIDENCE */}

      <Section
        title="Evidence"
        action={
          <div className="evidence-tabs">
            {["Overview", "Events", "Authentication", "Processes", "Network", "DNS"].map((value) => (
              <button className={tab === value ? "active" : ""} key={value} onClick={() => setTab(value)}>
                {value}
              </button>
            ))}
          </div>
        }
      >
        <div className="table-wrap investigation-table">
          <table>
            <thead>
              <tr>
                {headers.map((value) => (
                  <th key={value}>{value}</th>
                ))}
              </tr>
            </thead>

            <tbody>
              {rows.length === 0 && (
                <tr>
                  <td colSpan={7} className="muted">No {tab.toLowerCase()} evidence for this incident.</td>
                </tr>
              )}
              {rows.map((event) => (
                <tr key={event.id} onClick={() => setSelectedEvent(event)}>
                  {tab === "Processes" ? (
                    <>
                      <td>{formatTime(event.timestamp)}</td>
                      <td>{event.process_name || "—"}</td>
                      <td>{event.raw_event?.pid ?? event.raw_event?.process_id ?? "—"}</td>
                      <td>{event.raw_event?.parent_process ?? event.raw_event?.parent_image ?? "—"}</td>
                      <td>{event.username || "—"}</td>
                      <td>{event.command_line || "—"}</td>
                      <td>
                        <Pill tone={severityTone(event.severity)}>{event.severity || "—"}</Pill>
                      </td>
                    </>
                  ) : tab === "Network" ? (
                    <>
                      <td>{formatTime(event.timestamp)}</td>
                      <td>{event.source_ip || "—"}</td>
                      <td>{event.destination_ip || "—"}</td>
                      <td>{event.destination_port || "—"}</td>
                      <td>{protocolOf(event)}</td>
                      <td>{directionOf(event)}</td>
                      <td>
                        <Pill tone={severityTone(event.severity)}>{event.severity || "—"}</Pill>
                      </td>
                    </>
                  ) : (
                    <>
                      <td>{formatTime(event.timestamp)}</td>
                      <td>
                        <b>{event.event_type}</b>
                      </td>
                      <td>{event.source}</td>
                      <td>{event.username || "—"}</td>
                      <td>{event.hostname || "—"}</td>
                      <td>{event.source_ip || "—"}</td>
                      <td>
                        <Pill tone={severityTone(event.severity)}>{event.severity || "—"}</Pill>
                      </td>
                    </>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      {/* MITRE */}

      <Section title="MITRE ATT&CK">
        <div className="mitre-chain">
          {techniques.length === 0 && <p className="section-subtitle">No techniques are mapped on this incident's events.</p>}
          {techniques.map((technique, index) => (
            <div className="mitre-item" key={technique.id}>
              <button
                onClick={() => {
                  const first = events.find((e) => e.mitre_technique === technique.id);
                  if (first) setSelectedEvent(first);
                }}
                title={technique.tactic || undefined}
              >
                <b>{technique.id}</b>

                <span>{technique.name || "Not in reference table (run pnpm db:seed-mitre)"}</span>

                <small>
                  {technique.tactic ? `${technique.tactic} · ` : ""}
                  {technique.confidence !== null && technique.confidence !== undefined
                    ? `Confidence ${percent(technique.confidence)} · `
                    : ""}
                  Evidence {technique.evidence_count} event{technique.evidence_count === 1 ? "" : "s"}
                </small>
              </button>

              {index < techniques.length - 1 && <span className="chain-arrow">↓</span>}
            </div>
          ))}
        </div>
      </Section>

      {/* AI */}

      <Section title="AI Investigation">
        <div className="ai-investigation">
          <div className="ai-assessment">
            <span>Assessment</span>

            <b>{incident.title}</b>

            <Pill tone="muted">Confidence {percent(incident.ai_confidence)}</Pill>

            {incident.detection_model && <small>Model {incident.detection_model}</small>}
          </div>

          <div>
            <h3>Evidence considered</h3>

            <ul>
              {analysis.evidence.length === 0 && <li>No events linked to this incident.</li>}
              {analysis.evidence.map((value) => (
                <li key={value}>
                  <Check size={14} />
                  {value}
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h3>Investigation coverage</h3>

            <ul className="ai-steps">
              {analysis.coverage.map(([key, count]) => (
                <li key={key}>
                  <Check size={14} />
                  {key}
                  <span>{count}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="recommendations">
            <h3>Recommended Actions</h3>

            {analysis.recommendations.length === 0 && <p className="section-subtitle">No actions derived from the current evidence.</p>}
            {analysis.recommendations.map((value) => (
              <button
                key={value}
                onClick={() =>
                  critical.some((c) => value.startsWith(c)) ? setModal(value) : recordAction("action_requested", value)
                }
              >
                {value}
                <ExternalLink size={12} />
              </button>
            ))}
          </div>
        </div>
      </Section>

      {/* NOTES */}

      <Section title="Investigation Notes">
        <div id="notes" className="note-form">
          <textarea
            value={note}
            maxLength={5000}
            onChange={(event) => setNote(event.target.value)}
            placeholder="Add investigation notes..."
          />

          <button className="primary" disabled={busy || !note.trim()} onClick={saveNote}>
            Save note
          </button>
        </div>

        {notes.map((savedNote) => (
          <div className="saved-note" key={savedNote.id}>
            <b>{savedNote.analyst}</b>
            <span>{formatDateTime(savedNote.created_at)}</span>
            <p>{savedNote.note}</p>
          </div>
        ))}
      </Section>

      {/* AUDIT */}

      <Section title="Activity / Audit History">
        <div className="audit-list">
          {audit.length === 0 && (
            <div>
              <span>No analyst activity recorded yet.</span>
            </div>
          )}
          {audit.map((entry) => (
            <div key={entry.id}>
              <time>{formatDateTime(entry.created_at)}</time>
              <span>
                {(AUDIT_TEXT[entry.action] || (() => label(entry.action)))(entry.details)} · {entry.actor}
              </span>
            </div>
          ))}
        </div>
      </Section>

      {/* CONFIRMATION MODAL */}

      {modal && (
        <div className="modal-backdrop">
          <div className="confirm-modal">
            <button className="icon-btn" onClick={() => setModal("")}>
              <X size={16} />
            </button>

            <ShieldAlert size={24} />

            <h2>Confirm {modal}</h2>

            <p>
              This action may affect production systems. Confirming records your decision in the incident audit log;
              execute the containment step in your EDR / identity platform.
            </p>

            <div>
              <button className="filter" onClick={() => setModal("")}>
                Cancel
              </button>

              <button
                className="primary"
                onClick={() => {
                  const action = modal;
                  setModal("");
                  recordAction("action_confirmed", action);
                }}
              >
                Confirm action
              </button>
            </div>
          </div>
        </div>
      )}

      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}

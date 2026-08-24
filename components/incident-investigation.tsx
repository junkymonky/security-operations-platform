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
  ai_confidence: number;
  detected_at: string;
  assigned_analyst: string;
  summary: string;
  detection_source: string;
  correlated_event_count: number;
  affected_entity_count: number;
  detection_model: string;
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
}

export interface Entity {
  id: string;
  type: string;
  name: string;
  metadata: string[];
}

export interface MITRETechnique {
  id: string;
  name: string;
  confidence: number;
  evidenceCount: number;
}

export interface AIAnalysis {
  assessment: string;
  confidence: number;
  evidence: string[];
  recommendations: string[];
}

export interface InvestigationNote {
  analyst: string;
  timestamp: string;
  note: string;
}

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
}

export default function IncidentInvestigation({
  incident,
  events,
}: IncidentInvestigationProps) {
  const [status, setStatus] = useState(incident.status);
  const [owner, setOwner] = useState(
    incident.assigned_analyst || "Unassigned"
  );

  const [selectedEvent, setSelectedEvent] = useState<Event | null>(
    events.length > 0 ? events[events.length - 1] : null
  );

  const [tab, setTab] = useState("Overview");

  const [notes, setNotes] = useState<InvestigationNote[]>([]);
  const [note, setNote] = useState("");
  const [toast, setToast] = useState("");
  const [modal, setModal] = useState("");

  const notify = (message: string) => {
    setToast(message);

    window.setTimeout(() => {
      setToast("");
    }, 2200);
  };

  /*
   * Build entities from the real event data for now.
   *
   * Later this will come directly from the entities table.
   */
  const entities = useMemo<Entity[]>(() => {
    const result: Entity[] = [];

    const usernames = Array.from(
      new Set(
        events
          .map((event) => event.username)
          .filter(
            (value): value is string =>
              Boolean(value) && value !== "—"
          )
      )
    );

    const hosts = Array.from(
      new Set(
        events
          .map((event) => event.hostname)
          .filter(
            (value): value is string =>
              Boolean(value) && value !== "—"
          )
      )
    );

    const sourceIps = Array.from(
      new Set(
        events
          .map((event) => event.source_ip)
          .filter(
            (value): value is string =>
              Boolean(value) && value !== "—"
          )
      )
    );

    const processes = Array.from(
      new Set(
        events
          .map((event) => event.process_name)
          .filter(
            (value): value is string =>
              Boolean(value) && value !== "—"
          )
      )
    );

    usernames.forEach((username) => {
      result.push({
        id: `user-${username}`,
        type: "User",
        name: username,
        metadata: ["Identity / Active Directory"],
      });
    });

    hosts.forEach((host) => {
      result.push({
        id: `host-${host}`,
        type: "Host",
        name: host,
        metadata: ["Security endpoint"],
      });
    });

    sourceIps.forEach((ip) => {
      result.push({
        id: `ip-${ip}`,
        type: "Source IP",
        name: ip,
        metadata: ["Observed source"],
      });
    });

    processes.forEach((process) => {
      result.push({
        id: `process-${process}`,
        type: "Process",
        name: process,
        metadata: ["Observed execution"],
      });
    });

    return result;
  }, [events]);

  /*
   * Temporary AI placeholder.
   *
   * This will later come from the AI investigation service.
   */
  const ai: AIAnalysis = {
    assessment: "Possible account compromise",
    confidence: incident.ai_confidence,
    evidence: [
      "Multiple authentication failures",
      "Successful authentication",
      "Previously unseen source",
      "PowerShell execution",
      "Outbound network connection",
    ],
    recommendations: [
      "Investigate source IP",
      "Review user activity",
      "Review process tree",
      "Check host connections",
      "Revoke active sessions",
      "Isolate host",
    ],
  };

  /*
   * Temporary MITRE representation generated from events.
   *
   * Later this will come from the incident_mitre table.
   */
  const techniques = useMemo<MITRETechnique[]>(() => {
    const techniqueMap = new Map<string, number>();

    events.forEach((event) => {
      if (!event.mitre_technique) return;

      techniqueMap.set(
        event.mitre_technique,
        (techniqueMap.get(event.mitre_technique) || 0) + 1
      );
    });

    const names: Record<string, string> = {
      T1110: "Brute Force",
      T1078: "Valid Accounts",
      "T1059.001": "PowerShell",
      "T1071.001": "Web Protocols",
    };

    return Array.from(techniqueMap.entries()).map(
      ([id, evidenceCount]) => ({
        id,
        name: names[id] || "Unknown Technique",
        confidence: Math.min(
          99,
          70 + evidenceCount * 5
        ),
        evidenceCount,
      })
    );
  }, [events]);

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

            <Pill tone="critical">
              {incident.severity.toUpperCase()}
            </Pill>

            <Pill tone="muted">
              {status.toUpperCase()}
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
            <b>{incident.ai_confidence}%</b>
          </div>

          <div>
  <span>Detected</span>
  <b>
    {formatDateTime(incident.detected_at)}
  </b>
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
          onClick={() => {
            setOwner("Jordan Davis");
            notify("Investigation ownership assigned");
          }}
        >
          <UserRound size={14} />
          Take ownership
        </button>

        <label className="select-action">
          Change status

          <select
            value={status}
            onChange={(event) => {
              setStatus(event.target.value);
              notify(
                `Status changed to ${event.target.value}`
              );
            }}
          >
            {[
              "New",
              "Investigating",
              "Contained",
              "Resolved",
              "False Positive",
            ].map((value) => (
              <option key={value}>
                {value}
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

        <button
          className="filter"
          onClick={() =>
            notify("Incident export prepared")
          }
        >
          <Download size={14} />
          Export
        </button>

        <button className="filter">
          <MoreHorizontal size={14} />
          More
        </button>

        <button
          className="primary"
          onClick={() => {
            setStatus("Resolved");
            notify("Incident marked resolved");
          }}
        >
          Mark resolved
        </button>

      </div>

      {/* SUMMARY */}

      <Section title="Investigation Summary">

        <div className="summary-grid">

          <p>{incident.summary}</p>

          <div className="summary-facts">

            {[
              [
                "Detection source",
                incident.detection_source,
              ],
              [
                "Risk",
                `${incident.risk_score}/100`,
              ],
              [
                "AI confidence",
                `${incident.ai_confidence}%`,
              ],
              [
                "Correlated events",
                String(
                  incident.correlated_event_count
                ),
              ],
              [
                "Affected entities",
                String(
                  incident.affected_entity_count
                ),
              ],
              [
                "Detection model",
                incident.detection_model,
              ],
            ].map(([label, value]) => (
              <div key={label}>
                <span>{label}</span>
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

            {events.map((event) => (

              <button
                className={`timeline-event ${
                  selectedEvent?.id === event.id
                    ? "selected"
                    : ""
                }`}
                key={event.id}
                onClick={() =>
                  setSelectedEvent(event)
                }
              >

              <time>
  {formatTime(event.timestamp)}
</time>

                <i
                  className={severityTone(
                    event.severity
                  )}
                />

                <div>

                  <b>
                    {event.event_type.replaceAll(
                      "_",
                      " "
                    )}
                  </b>

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

            {entities.map((entity) => (

              <button
                className="entity-row"
                key={entity.id}
                onClick={() =>
                  notify(
                    `${entity.type} investigation opened`
                  )
                }
              >

                {entity.type === "User" ? (
                  <UserRound size={15} />
                ) : entity.type === "Host" ? (
                  <Server size={15} />
                ) : entity.type === "Source IP" ? (
                  <Globe2 size={15} />
                ) : (
                  <Terminal size={15} />
                )}

                <span>

                  <b>{entity.name}</b>

                  {entity.metadata.map(
                    (value) => (
                      <small key={value}>
                        {value}
                      </small>
                    )
                  )}

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
            <button
              className="icon-btn"
              onClick={() =>
                setSelectedEvent(null)
              }
            >
              <X size={15} />
            </button>
          }
        >

          <div className="event-details">

            {[
              ["Event ID", selectedEvent.id],
              [
                [
  "Timestamp",
  formatDateTime(selectedEvent.timestamp),
],
              ],
              [
                "Event type",
                selectedEvent.event_type,
              ],
              ["Source", selectedEvent.source],
              [
                "User",
                selectedEvent.username || "—",
              ],
              [
                "Host",
                selectedEvent.hostname || "—",
              ],
              [
                "Source IP",
                selectedEvent.source_ip || "—",
              ],
              [
                "Destination IP",
                selectedEvent.destination_ip || "—",
              ],
              [
                "Process",
                selectedEvent.process_name || "—",
              ],
              [
                "Command",
                selectedEvent.command_line || "—",
              ],
              [
                "Detection",
                selectedEvent.detection_method ||
                  "—",
              ],
              [
                "Severity",
                selectedEvent.severity || "—",
              ],
              [
                "MITRE",
                selectedEvent.mitre_technique ||
                  "—",
              ],
            ].map(([label, value]) => (

              <div key={label}>
                <span>{label}</span>
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

            {[
              "Overview",
              "Events",
              "Authentication",
              "Processes",
              "Network",
              "DNS",
            ].map((value) => (

              <button
                className={
                  tab === value
                    ? "active"
                    : ""
                }
                key={value}
                onClick={() =>
                  setTab(value)
                }
              >
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

                {tab === "Processes"
                  ? [
                      "Time",
                      "Process",
                      "PID",
                      "Parent Process",
                      "User",
                      "Command",
                      "Risk",
                    ]
                  : tab === "Network"
                    ? [
                        "Timestamp",
                        "Source",
                        "Destination",
                        "Port",
                        "Protocol",
                        "Direction",
                        "Risk",
                      ]
                    : [
                        "Timestamp",
                        "Event Type",
                        "Source",
                        "User",
                        "Host",
                        "IP",
                        "Severity",
                      ].map((value) => (
                        <th key={value}>
                          {value}
                        </th>
                      ))}

              </tr>

            </thead>

            <tbody>

              {rows.map((event) => (

                <tr
                  key={event.id}
                  onClick={() =>
                    setSelectedEvent(event)
                  }
                >

                  {tab === "Processes" ? (

                    <>
                      <td>
                        
  {formatTime(event.timestamp)}

                      </td>

                      <td>
                        {event.process_name || "—"}
                      </td>

                      <td>—</td>

                      <td>—</td>

                      <td>
                        {event.username || "—"}
                      </td>

                      <td>
                        {event.command_line || "—"}
                      </td>

                      <td>
                        <Pill tone="high">
                          {event.severity || "—"}
                        </Pill>
                      </td>
                    </>

                  ) : tab === "Network" ? (

                    <>
                      <td>
                        
  {formatTime(event.timestamp)}

                      </td>

                      <td>
                        {event.source_ip || "—"}
                      </td>

                      <td>
                        {event.destination_ip || "—"}
                      </td>

                      <td>
                        {event.destination_port || "—"}
                      </td>

                      <td>
                        HTTPS
                      </td>

                      <td>
                        Outbound
                      </td>

                      <td>
                        <Pill tone="high">
                          {event.severity || "—"}
                        </Pill>
                      </td>
                    </>

                  ) : (

                    <>
                      <td>
  {formatTime(event.timestamp)}
                      </td>

                      <td>
                        <b>
                          {event.event_type}
                        </b>
                      </td>

                      <td>
                        {event.source}
                      </td>

                      <td>
                        {event.username || "—"}
                      </td>

                      <td>
                        {event.hostname || "—"}
                      </td>

                      <td>
                        {event.source_ip || "—"}
                      </td>

                      <td>
                        <Pill
                          tone={severityTone(
                            event.severity
                          )}
                        >
                          {event.severity || "—"}
                        </Pill>
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

          {techniques.map(
            (technique, index) => (

              <div
                className="mitre-item"
                key={technique.id}
              >

                <button
                  onClick={() =>
                    notify(
                      `${technique.id} technique details opened`
                    )
                  }
                >

                  <b>{technique.id}</b>

                  <span>
                    {technique.name}
                  </span>

                  <small>
                    Confidence{" "}
                    {technique.confidence}%
                    {" · "}
                    Evidence{" "}
                    {technique.evidenceCount} events
                  </small>

                </button>

                {index <
                  techniques.length - 1 && (
                  <span className="chain-arrow">
                    ↓
                  </span>
                )}

              </div>

            )
          )}

        </div>

      </Section>

      {/* AI */}

      <Section title="AI Investigation">

        <div className="ai-investigation">

          <div className="ai-assessment">

            <span>Assessment</span>

            <b>{ai.assessment}</b>

            <Pill tone="muted">
              Confidence {ai.confidence}%
            </Pill>

          </div>

          <div>

            <h3>Evidence considered</h3>

            <ul>

              {ai.evidence.map(
                (value) => (
                  <li key={value}>
                    <Check size={14} />
                    {value}
                  </li>
                )
              )}

            </ul>

          </div>

          <div>

            <h3>Investigation steps</h3>

            <ul className="ai-steps">

              {[
                "Search authentication events",
                "Review user history",
                "Correlate host activity",
                "Check source IP reputation",
                "Map MITRE ATT&CK techniques",
              ].map((value) => (

                <li key={value}>

                  <Check size={14} />

                  {value}

                  <span>
                    Completed
                  </span>

                </li>

              ))}

            </ul>

          </div>

          <div className="recommendations">

            <h3>Recommended Actions</h3>

            {ai.recommendations.map(
              (value) => (

                <button
                  key={value}
                  onClick={() =>
                    [
                      "Isolate host",
                      "Revoke active sessions",
                    ].includes(value)
                      ? setModal(value)
                      : notify(
                          `${value} queued for review`
                        )
                  }
                >
                  {value}
                  <ExternalLink size={12} />
                </button>

              )
            )}

          </div>

        </div>

      </Section>

      {/* NOTES */}

      <Section title="Investigation Notes">

        <div
          id="notes"
          className="note-form"
        >

          <textarea
            value={note}
            onChange={(event) =>
              setNote(event.target.value)
            }
            placeholder="Add investigation notes..."
          />

          <button
            className="primary"
            onClick={() => {

              if (!note.trim()) return;

              setNotes([
                ...notes,
                {
                  analyst: owner,
                  timestamp: "Just now",
                  note,
                },
              ]);

              setNote("");

              notify("Note saved");
            }}
          >
            Save note
          </button>

        </div>

        {notes.map(
          (savedNote, index) => (

            <div
              className="saved-note"
              key={index}
            >

              <b>
                {savedNote.analyst}
              </b>

              <span>
                {savedNote.timestamp}
              </span>

              <p>
                {savedNote.note}
              </p>

            </div>

          )
        )}

      </Section>

      {/* AUDIT */}

      <Section title="Activity / Audit History">

        <div className="audit-list">

          <div>
            <time>—</time>
            <span>
              Incident loaded from database
            </span>
          </div>

          <div>
            <time>—</time>
            <span>
              Investigation data retrieved
            </span>
          </div>

        </div>

      </Section>

      {/* CONFIRMATION MODAL */}

      {modal && (

        <div className="modal-backdrop">

          <div className="confirm-modal">

            <button
              className="icon-btn"
              onClick={() =>
                setModal("")
              }
            >
              <X size={16} />
            </button>

            <ShieldAlert size={24} />

            <h2>
              Confirm {modal}
            </h2>

            <p>
              This action may affect production
              systems. Confirm only if you have
              validated the evidence.
            </p>

            <div>

              <button
                className="filter"
                onClick={() =>
                  setModal("")
                }
              >
                Cancel
              </button>

              <button
                className="primary"
                onClick={() => {
                  const action = modal;

                  setModal("");

                  notify(
                    `${action} confirmation recorded`
                  );
                }}
              >
                Confirm action
              </button>

            </div>

          </div>

        </div>

      )}

      {toast && (
        <div className="toast">
          {toast}
        </div>
      )}

    </div>
  );
}
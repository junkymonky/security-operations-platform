export type Severity = "critical" | "high" | "medium" | "low";
export type IncidentStatus =
  | "new"
  | "investigating"
  | "contained"
  | "resolved"
  | "false_positive";

export interface Incident {
  id: string;
  title: string;
  severity: Severity;
  status: IncidentStatus;
  risk_score: number;
  ai_confidence: number | null;
  detected_at: string;
  assigned_analyst: string | null;
  summary: string | null;
  detection_source: string | null;
  detection_model: string | null;
  correlated_event_count: number;
  affected_entity_count: number;
}

export interface SecurityEvent {
  id: string;
  incident_id: string | null;
  timestamp: string;
  event_type: string;
  source: string;
  username: string | null;
  hostname: string | null;
  source_ip: string | null;
  destination_ip: string | null;
  destination_port: number | null;
  process_name: string | null;
  command_line: string | null;
  severity: string | null;
  detection_method: string | null;
  mitre_technique: string | null;
}

export interface Entity {
  id: string;
  type: string;
  name: string;
  metadata: Record<string, unknown>;
}

export interface MitreTechnique {
  id: string;
  name: string;
  tactic: string | null;
  confidence: number | null;
  evidence_count: number;
}
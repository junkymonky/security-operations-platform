import IncidentInvestigation from "@/components/incident-investigation";
import { sql } from "@/lib/db";

interface IncidentPageProps {
  params: Promise<{
    id: string;
  }>;
}

export default async function IncidentPage({
  params,
}: IncidentPageProps) {
  const { id } = await params;

  const incidents = await sql`
    SELECT
      id,
      title,
      severity,
      status,
      risk_score,
      ai_confidence,
      detected_at,
      assigned_analyst,
      summary,
      detection_source,
      detection_model,
      correlated_event_count,
      affected_entity_count
    FROM incidents
    WHERE id = ${id}
    LIMIT 1
  `;

  if (incidents.length === 0) {
    throw new Error("Incident not found");
  }

  const events = await sql`
    SELECT
      id,
      timestamp,
      event_type,
      source,
      username,
      hostname,
      source_ip,
      destination_ip,
      destination_port,
      process_name,
      command_line,
      severity,
      detection_method,
      mitre_technique
    FROM events
    WHERE incident_id = ${id}
    ORDER BY timestamp ASC
  `;

  return (
    <IncidentInvestigation
      incident={incidents[0]}
      events={events}
    />
  );
}
import { notFound } from "next/navigation";
import IncidentInvestigation from "@/components/incident-investigation";
import { sql } from "@/lib/db";
import { analystProfile } from "@/lib/profile";
import { techniqueSummary } from "@/lib/soc-data";

export const dynamic = "force-dynamic";

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
    notFound();
  }

  const [events, techniques, notes, audit] = await Promise.all([
    sql`
      SELECT
        id,
        timestamp,
        event_type,
        source,
        username,
        hostname,
        host(source_ip) AS source_ip,
        host(destination_ip) AS destination_ip,
        destination_port,
        process_name,
        command_line,
        severity,
        detection_method,
        mitre_technique,
        raw_event
      FROM events
      WHERE incident_id = ${id}
      ORDER BY timestamp ASC
    `,
    techniqueSummary(id),
    sql`SELECT id, analyst, note, created_at FROM investigation_notes WHERE incident_id = ${id} ORDER BY created_at DESC`,
    sql`SELECT id, actor, action, details, created_at FROM audit_logs WHERE incident_id = ${id} ORDER BY created_at DESC LIMIT 200`,
  ]);

  return (
    <IncidentInvestigation
      incident={incidents[0] as any}
      events={events as any}
      techniques={techniques as any}
      initialNotes={notes as any}
      initialAudit={audit as any}
      analyst={analystProfile().name}
    />
  );
}

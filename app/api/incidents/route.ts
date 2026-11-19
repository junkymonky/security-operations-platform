import { NextRequest, NextResponse } from "next/server";
import { sql } from "@/lib/db";
import { analystProfile, writeAudit } from "@/lib/soc-data";

const SEVERITIES = ["low", "medium", "high", "critical"];

export async function GET() {
  try {
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
      ORDER BY detected_at DESC
    `;

    return NextResponse.json({
      success: true,
      data: incidents,
    });
  } catch (error) {
    console.error("GET /api/incidents failed:", error);

    return NextResponse.json(
      {
        success: false,
        error: "Failed to fetch incidents",
      },
      { status: 500 }
    );
  }
}

/** Open a new, manually created investigation. */
export async function POST(request: NextRequest) {
  let body: any;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ success: false, error: "Body must be JSON" }, { status: 400 });
  }
  const title = String(body?.title ?? "").trim();
  const severity = String(body?.severity ?? "").toLowerCase();
  const summary = String(body?.summary ?? "").trim();
  if (title.length < 3 || title.length > 200) {
    return NextResponse.json({ success: false, error: "Title must be 3–200 characters" }, { status: 400 });
  }
  if (!SEVERITIES.includes(severity)) {
    return NextResponse.json({ success: false, error: "Severity must be low, medium, high or critical" }, { status: 400 });
  }
  if (summary.length > 2000) {
    return NextResponse.json({ success: false, error: "Summary must be at most 2000 characters" }, { status: 400 });
  }
  const analyst = analystProfile().name || null;
  try {
    const [row] = await sql`
      WITH next AS (
        SELECT 'INC-' || (coalesce(max(NULLIF(regexp_replace(id, '\\D', '', 'g'), '')::int), 1000) + 1) AS id
        FROM incidents
      )
      INSERT INTO incidents (id, title, severity, status, risk_score, summary, detection_source, assigned_analyst)
      SELECT next.id, ${title}, ${severity}, 'new', 0, ${summary || null}, 'Manual', ${analyst} FROM next
      RETURNING id, title, severity, status, detected_at`;
    await writeAudit(row.id, analyst || "analyst", "incident_created", { title, severity });
    return NextResponse.json({ success: true, data: row }, { status: 201 });
  } catch (error) {
    console.error("POST /api/incidents failed:", error);
    return NextResponse.json({ success: false, error: "Failed to create incident" }, { status: 500 });
  }
}

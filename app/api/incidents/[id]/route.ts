import { NextRequest, NextResponse } from "next/server";
import { sql } from "@/lib/db";
import { analystProfile, writeAudit } from "@/lib/soc-data";

export async function GET(
  _request: NextRequest,
  context: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await context.params;

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
      return NextResponse.json(
        {
          success: false,
          error: "Incident not found",
        },
        { status: 404 }
      );
    }

    return NextResponse.json({
      success: true,
      data: incidents[0],
    });
  } catch (error) {
    console.error("GET /api/incidents/[id] failed:", error);

    return NextResponse.json(
      {
        success: false,
        error: "Failed to fetch incident",
      },
      { status: 500 }
    );
  }
}
const STATUSES = ["new", "investigating", "contained", "resolved", "false_positive"];

/** Update status and/or assigned analyst; every change is written to audit_logs. */
export async function PATCH(
  request: NextRequest,
  context: { params: Promise<{ id: string }> }
) {
  const { id } = await context.params;
  let body: any;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ success: false, error: "Body must be JSON" }, { status: 400 });
  }
  const status = body?.status === undefined ? undefined : String(body.status).toLowerCase();
  const assigned =
    body?.assigned_analyst === undefined ? undefined : String(body.assigned_analyst).trim().slice(0, 120) || null;
  if (status === undefined && assigned === undefined) {
    return NextResponse.json({ success: false, error: "Nothing to update" }, { status: 400 });
  }
  if (status !== undefined && !STATUSES.includes(status)) {
    return NextResponse.json({ success: false, error: `Status must be one of ${STATUSES.join(", ")}` }, { status: 400 });
  }
  try {
    const [before] = await sql`SELECT status, assigned_analyst FROM incidents WHERE id = ${id}`;
    if (!before) {
      return NextResponse.json({ success: false, error: "Incident not found" }, { status: 404 });
    }
    const [row] = await sql`
      UPDATE incidents SET
        status = COALESCE(${status ?? null}, status),
        assigned_analyst = CASE WHEN ${assigned !== undefined} THEN ${assigned ?? null} ELSE assigned_analyst END,
        updated_at = NOW()
      WHERE id = ${id}
      RETURNING id, status, assigned_analyst, updated_at`;
    const actor = analystProfile().name || "analyst";
    if (status !== undefined && status !== before.status) {
      await writeAudit(id, actor, "status_changed", { from: before.status, to: status });
    }
    if (assigned !== undefined && assigned !== before.assigned_analyst) {
      await writeAudit(id, actor, "owner_changed", { from: before.assigned_analyst, to: assigned });
    }
    return NextResponse.json({ success: true, data: row });
  } catch (error) {
    console.error("PATCH /api/incidents/[id] failed:", error);
    return NextResponse.json({ success: false, error: "Failed to update incident" }, { status: 500 });
  }
}

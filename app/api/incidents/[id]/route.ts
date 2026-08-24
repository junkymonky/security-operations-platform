import { NextRequest, NextResponse } from "next/server";
import { sql } from "@/lib/db";

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
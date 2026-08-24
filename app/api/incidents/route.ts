import { NextResponse } from "next/server";
import { sql } from "@/lib/db";

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
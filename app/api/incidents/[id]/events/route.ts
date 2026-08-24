import { NextRequest, NextResponse } from "next/server";
import { sql } from "@/lib/db";

export async function GET(
  _request: NextRequest,
  context: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await context.params;

    const events = await sql`
      SELECT
        id,
        incident_id,
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

    return NextResponse.json({
      success: true,
      data: events,
    });
  } catch (error) {
    console.error("GET incident events failed:", error);

    return NextResponse.json(
      {
        success: false,
        error: "Failed to fetch incident events",
      },
      { status: 500 }
    );
  }
}
import { NextRequest, NextResponse } from "next/server";
import { sql } from "@/lib/db";
import { analystProfile, writeAudit } from "@/lib/soc-data";

type Ctx = { params: Promise<{ id: string }> };

// Analyst actions that can be recorded from the investigation page.
const ACTIONS = ["action_requested", "action_confirmed", "incident_exported"];

export async function GET(_request: NextRequest, context: Ctx) {
  const { id } = await context.params;
  try {
    const rows = await sql`
      SELECT id, actor, action, details, created_at FROM audit_logs
      WHERE incident_id = ${id} ORDER BY created_at DESC LIMIT 200`;
    return NextResponse.json({ success: true, data: rows });
  } catch (error) {
    console.error("GET audit failed:", error);
    return NextResponse.json({ success: false, error: "Failed to fetch audit history" }, { status: 500 });
  }
}

export async function POST(request: NextRequest, context: Ctx) {
  const { id } = await context.params;
  let body: any;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ success: false, error: "Body must be JSON" }, { status: 400 });
  }
  const action = String(body?.action ?? "");
  if (!ACTIONS.includes(action)) {
    return NextResponse.json({ success: false, error: `action must be one of ${ACTIONS.join(", ")}` }, { status: 400 });
  }
  const details = body?.details && typeof body.details === "object" ? body.details : {};
  if (JSON.stringify(details).length > 4000) {
    return NextResponse.json({ success: false, error: "details too large" }, { status: 400 });
  }
  try {
    const [exists] = await sql`SELECT 1 FROM incidents WHERE id = ${id}`;
    if (!exists) return NextResponse.json({ success: false, error: "Incident not found" }, { status: 404 });
    await writeAudit(id, analystProfile().name || "analyst", action, details);
    return NextResponse.json({ success: true }, { status: 201 });
  } catch (error) {
    console.error("POST audit failed:", error);
    return NextResponse.json({ success: false, error: "Failed to record action" }, { status: 500 });
  }
}

import { NextRequest, NextResponse } from "next/server";
import { sql } from "@/lib/db";
import { analystProfile, writeAudit } from "@/lib/soc-data";

type Ctx = { params: Promise<{ id: string }> };

export async function GET(_request: NextRequest, context: Ctx) {
  const { id } = await context.params;
  try {
    const notes = await sql`
      SELECT id, analyst, note, created_at FROM investigation_notes
      WHERE incident_id = ${id} ORDER BY created_at DESC`;
    return NextResponse.json({ success: true, data: notes });
  } catch (error) {
    console.error("GET notes failed:", error);
    return NextResponse.json({ success: false, error: "Failed to fetch notes" }, { status: 500 });
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
  const note = String(body?.note ?? "").trim();
  if (!note || note.length > 5000) {
    return NextResponse.json({ success: false, error: "Note must be 1–5000 characters" }, { status: 400 });
  }
  const analyst = String(body?.analyst ?? "").trim().slice(0, 120) || analystProfile().name || "analyst";
  try {
    const [exists] = await sql`SELECT 1 FROM incidents WHERE id = ${id}`;
    if (!exists) return NextResponse.json({ success: false, error: "Incident not found" }, { status: 404 });
    const [row] = await sql`
      INSERT INTO investigation_notes (incident_id, analyst, note) VALUES (${id}, ${analyst}, ${note})
      RETURNING id, analyst, note, created_at`;
    await writeAudit(id, analyst, "note_added", { note_id: row.id });
    return NextResponse.json({ success: true, data: row }, { status: 201 });
  } catch (error) {
    console.error("POST notes failed:", error);
    return NextResponse.json({ success: false, error: "Failed to save note" }, { status: 500 });
  }
}

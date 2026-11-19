import { NextRequest, NextResponse } from "next/server";
import { listEvents } from "@/lib/soc-data";

const SEVERITIES = ["low", "medium", "high", "critical"];

export async function GET(request: NextRequest) {
  const raw = request.nextUrl.searchParams.get("severity");
  const severities = raw ? raw.split(",").map((s) => s.trim().toLowerCase()).filter(Boolean) : undefined;
  if (severities?.some((s) => !SEVERITIES.includes(s))) {
    return NextResponse.json({ success: false, error: "Invalid severity filter" }, { status: 400 });
  }
  const limit = Number(request.nextUrl.searchParams.get("limit") || 500);
  try {
    const events = await listEvents({ severities, limit: Number.isFinite(limit) ? limit : 500 });
    return NextResponse.json({ success: true, data: events });
  } catch (error) {
    console.error("GET /api/events failed:", error);
    return NextResponse.json({ success: false, error: "Failed to fetch events" }, { status: 500 });
  }
}

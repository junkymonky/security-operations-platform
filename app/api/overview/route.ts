import { NextRequest, NextResponse } from "next/server";
import { activitySeries, databaseLatencyMs, listIncidents, overviewMetrics, techniqueSummary, type Range, type SeriesMetric } from "@/lib/soc-data";

const RANGES: Range[] = ["1h", "6h", "24h", "7d", "all"];
const METRICS: SeriesMetric[] = ["events", "detections", "alerts", "incidents", "critical", "ml"];

export async function GET(request: NextRequest) {
  const range = (request.nextUrl.searchParams.get("range") || "all") as Range;
  const metric = (request.nextUrl.searchParams.get("metric") || "events") as SeriesMetric;
  if (!RANGES.includes(range) || !METRICS.includes(metric)) {
    return NextResponse.json({ success: false, error: "Invalid range or metric" }, { status: 400 });
  }
  try {
    const latency = await databaseLatencyMs();
    const [metrics, series, incidents, techniques] = await Promise.all([
      overviewMetrics(),
      activitySeries(range, metric),
      listIncidents(),
      techniqueSummary(),
    ]);
    return NextResponse.json({
      success: true,
      data: { metrics, series, incidents, techniques, health: { database_latency_ms: latency }, generated_at: new Date().toISOString() },
    });
  } catch (error) {
    console.error("GET /api/overview failed:", error);
    return NextResponse.json({ success: false, error: "Failed to load overview data" }, { status: 500 });
  }
}

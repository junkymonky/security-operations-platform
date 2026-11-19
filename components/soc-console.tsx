'use client'

import { useCallback, useEffect, useMemo, useState } from 'react'
import { Activity, AlertOctagon, ArrowUpRight, Bell, Bot, BrainCircuit, ChevronDown, ChevronLeft, ChevronRight, CircleHelp, Command, Globe2, LayoutDashboard, Menu, MoreHorizontal, Network, Plus, RefreshCw, Search, Server, Settings, Shield, ShieldAlert, Target, Terminal, Timer, UserRound, X } from 'lucide-react'
import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import MLAnalytics from '@/components/ml/ml-analytics'
import type { AnalystProfile } from '@/lib/profile'

// ------------------------------------------------------------------ types & helpers
interface IncidentRow {
  id: string; title: string; severity: string; status: string; risk_score: number; ai_confidence: string | number | null
  detected_at: string; assigned_analyst: string | null; summary: string | null; detection_source: string | null
  detection_model: string | null; correlated_event_count: number; affected_entity_count: number
  primary_user: string | null; primary_host: string | null
}
interface EventRow {
  id: string; incident_id: string | null; timestamp: string; event_type: string; source: string; username: string | null
  hostname: string | null; source_ip: string | null; destination_ip: string | null; destination_port: number | null
  process_name: string | null; command_line: string | null; severity: string | null; detection_method: string | null; mitre_technique: string | null
}
interface Overview {
  metrics: Record<string, any>
  series: { start: string | null; end: string | null; bucketSeconds: number; points: { t: string; v: number; critical: number }[] }
  incidents: IncidentRow[]
  techniques: { id: string; name: string | null; tactic: string | null; evidence_count: number; incident_count: number; last_seen: string }[]
  health: { database_latency_ms: number }
  generated_at: string
}

const SERIES_METRICS: [string, string][] = [['events', 'Events'], ['detections', 'Detections'], ['alerts', 'Alerts'], ['incidents', 'Incidents'], ['critical', 'Critical Alerts'], ['ml', 'ML Anomalies']]
const RANGES: [string, string][] = [['1h', 'Last hour'], ['6h', 'Last 6 hours'], ['24h', 'Last 24 hours'], ['7d', 'Last 7 days'], ['all', 'All time']]
const ACTIVE = ['new', 'investigating', 'contained']

const cap = (s: string | null | undefined) => (s ? s.replaceAll('_', ' ').replace(/\b\w/g, (c) => c.toUpperCase()) : '—')
const tone = (s: string | null | undefined) => (s ? s.toLowerCase() : 'muted')
const initials = (text: string, fallback = '?') => {
  const p = text.trim().split(/\s+/).filter(Boolean)
  return p.length ? (p[0][0] + (p.length > 1 ? p[p.length - 1][0] : '')).toUpperCase() : fallback
}
function ago(iso: string | null | undefined, now: number) {
  if (!iso || !now) return '—'
  const s = Math.max(0, Math.round((now - new Date(iso).getTime()) / 1000))
  if (s < 60) return `${s}s ago`
  if (s < 3600) return `${Math.floor(s / 60)}m ago`
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`
  return `${Math.floor(s / 86400)}d ago`
}
function duration(sec: number | null | undefined) {
  if (sec === null || sec === undefined || !Number.isFinite(Number(sec))) return '—'
  const s = Math.round(Number(sec))
  if (s < 60) return `${s}s`
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`
}
const compact = (n: number) => new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 }).format(n)
const conf = (v: string | number | null) => (v === null || v === undefined ? null : Math.round(Number(v)))

async function getJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  const body = await res.json().catch(() => null)
  if (!res.ok || body?.success === false) throw new Error(body?.error || `Request failed (${res.status})`)
  return (body?.data ?? body) as T
}

// ------------------------------------------------------------------ small components
function Pill({ children, tone = 'muted' }: { children: React.ReactNode; tone?: string }) { return <span className={`pill pill-${tone}`}>{children}</span> }
function Metric({ label, value, delta, note, icon: Icon, tone }: any) {
  return <div className="metric"><div className="metric-top"><span>{label}</span><Icon size={15} /></div><strong>{value}</strong><div className="metric-bottom"><span className={`delta ${tone}`}>{delta}</span><span className="muted">{note}</span></div></div>
}

function Sidebar({ collapsed, setCollapsed, active, setActive, counts, profile }: any) {
  const nav = [
    { label: 'Overview', icon: LayoutDashboard }, { label: 'Alerts', icon: ShieldAlert, count: counts.alerts }, { label: 'Incidents', icon: AlertOctagon, count: counts.incidents },
    { label: 'Events', icon: Activity }, { label: 'AI Analyst', icon: Bot, badge: 'BETA' }, { label: 'MITRE ATT&CK', icon: Target }, { label: 'Threat Intel', icon: Globe2 }, { label: 'ML Analytics', icon: BrainCircuit },
  ]
  const org = profile.org || 'SOC workspace'
  return <aside className={`sidebar ${collapsed ? 'collapsed' : ''}`}><div className="brand"><div className="brand-mark"><Shield size={18} /></div>{!collapsed && <><span>Grep<span className="brand-accent"> Analyst</span></span><span className="env">SOC</span></>}</div><button className="collapse" onClick={() => setCollapsed(!collapsed)}>{collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}</button><div className="workspace"><div className="workspace-icon">{initials(org, 'S')}</div>{!collapsed && <div><b>{org}</b><small>Workspace</small></div>}{!collapsed && <ChevronDown size={14} className="muted" />}</div><nav>{nav.map(item => { const I = item.icon; return <button key={item.label} className={active === item.label ? 'active' : ''} onClick={() => setActive(item.label)} title={item.label}><I size={17} />{!collapsed && <><span>{item.label}</span>{item.count ? <em>{item.count}</em> : null}{item.badge && <small>{item.badge}</small>}</>}</button> })}</nav><div className="sidebar-bottom"><button onClick={() => setActive('Settings')}><Settings size={17} />{!collapsed && 'Settings'}</button><button><CircleHelp size={17} />{!collapsed && 'Help center'}</button><div className="profile"><div className="avatar">{initials(profile.name)}</div>{!collapsed && <div><b>{profile.name || 'Analyst'}</b><small>{profile.role}</small></div>}</div></div></aside>
}

function IncidentTable({ incidents, onSelect, now, title = 'Critical Incidents', subtitle = 'Incidents requiring analyst attention.', onViewAll }: any) {
  return <div className="panel incident-panel"><div className="panel-head"><div><h2>{title}</h2><p>{subtitle}</p></div>{onViewAll && <button className="ghost" onClick={onViewAll}>View all <ArrowUpRight size={14} /></button>}</div><div className="table-wrap"><table><thead><tr><th>Severity</th><th>Incident</th><th>User</th><th>Host</th><th>Risk</th><th>Status</th><th>Time</th></tr></thead><tbody>
    {incidents.length === 0 && <tr><td colSpan={7} className="muted">No incidents.</td></tr>}
    {incidents.map((x: IncidentRow) => <tr key={x.id} onClick={() => onSelect(x)}><td><Pill tone={tone(x.severity)}>{cap(x.severity)}</Pill></td><td><b>{x.title}</b><small>{x.id}{conf(x.ai_confidence) !== null ? ` · AI confidence ${conf(x.ai_confidence)}%` : ''}</small></td><td>{x.primary_user || '—'}</td><td>{x.primary_host || '—'}</td><td><span className={`risk ${x.risk_score > 80 ? 'high-risk' : ''}`}>{x.risk_score}</span></td><td><Pill tone={x.status === 'new' ? 'new' : 'muted'}>{cap(x.status)}</Pill></td><td className="muted" title={x.detected_at}>{ago(x.detected_at, now)}</td></tr>)}
  </tbody></table></div></div>
}

function ThreatActivity({ series, metric, setMetric, range, setRange, loading }: any) {
  const points = series?.points || []
  const spanDays = series?.start ? (new Date(series.end).getTime() - new Date(series.start).getTime()) / 86400000 : 1
  const fmt = (t: string) => {
    const d = new Date(t)
    return spanDays > 2 ? d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', timeZone: 'UTC' })
      : d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: spanDays < 0.01 ? '2-digit' : undefined, timeZone: 'UTC' })
  }
  const data = points.map((p: any) => ({ ...p, label: fmt(p.t) }))
  const total = points.reduce((a: number, p: any) => a + p.v, 0)
  const metricLabel = SERIES_METRICS.find(([k]) => k === metric)?.[1]
  return <div className="panel overview-chart"><div className="panel-head"><div><h2>Threat Activity</h2><p>{loading ? 'Loading…' : `${total} ${metricLabel?.toLowerCase()} in range${series?.start ? ` · ${new Date(series.start).toUTCString().slice(5, 22)} – ${new Date(series.end).toUTCString().slice(5, 22)} UTC` : ''}`}</p></div><div className="chart-controls"><select value={metric} onChange={e => setMetric(e.target.value)}>{SERIES_METRICS.map(([k, l]) => <option key={k} value={k}>{l}</option>)}</select><select value={range} onChange={e => setRange(e.target.value)}>{RANGES.map(([k, l]) => <option key={k} value={k}>{l}</option>)}</select></div></div><div className="chart-legend"><span><i className="legend-line" /> <b>{metricLabel}</b></span><span><i className="legend-red" /> Critical activity</span></div><div className="chart">{!loading && total === 0 && !points.some((p: any) => p.critical) ? <div className="empty-workspace" style={{ minHeight: 160 }}><p>No {metricLabel?.toLowerCase()} recorded in this time range.</p></div> : <ResponsiveContainer width="100%" height="100%"><AreaChart data={data}><XAxis dataKey="label" stroke="#61707d" fontSize={10} tickLine={false} axisLine={false} minTickGap={24} /><YAxis stroke="#61707d" fontSize={10} tickLine={false} axisLine={false} width={28} allowDecimals={false} /><Tooltip contentStyle={{ background: '#111a22', border: '1px solid #34414d', color: '#dce5ec', fontSize: 11 }} labelFormatter={(_, p: any) => p?.[0] ? `${new Date(p[0].payload.t).toUTCString().slice(5, 25)} UTC` : ''} /><Area type="monotone" dataKey="v" name={metricLabel} stroke="#8a99a8" fill="#8a99a8" fillOpacity={0.08} strokeWidth={1.5} isAnimationActive={false} /><Area type="monotone" dataKey="critical" name="Critical" stroke="#ff6570" fill="#ff6570" fillOpacity={0.05} strokeWidth={1.5} isAnimationActive={false} /></AreaChart></ResponsiveContainer>}</div></div>
}

function AIAnalyst({ incidents, onOpen, onSelect }: any) {
  const queue = incidents.filter((x: IncidentRow) => ACTIVE.includes(x.status)).sort((a: IncidentRow, b: IncidentRow) => b.risk_score - a.risk_score)
  return <div className="panel ai-panel"><div className="panel-head"><div><h2>AI Analyst</h2><p>Investigations requiring attention</p></div><Bot size={17} className="muted" /></div><div className="ai-count"><strong>{queue.length}</strong><span>investigation{queue.length === 1 ? '' : 's'} require{queue.length === 1 ? 's' : ''} attention</span></div>{queue.slice(0, 3).map((x: IncidentRow) => <div className="ai-item" key={x.id} onClick={() => onSelect(x)} style={{ cursor: 'pointer' }}><div><Pill tone={tone(x.severity)}>{cap(x.severity)}</Pill><b>Incident {x.id}</b><span>{x.title}</span></div><small>Model confidence <strong>{conf(x.ai_confidence) !== null ? `${conf(x.ai_confidence)}%` : '—'}</strong></small></div>)}<button className="primary ai-button" onClick={onOpen}>Open AI Analyst <ArrowUpRight size={14} /></button></div>
}

function Health({ health }: { health: { label: string; value: string; ok: boolean }[] }) {
  const down = health.filter(h => !h.ok).length
  return <div className="panel health"><div className="panel-head"><div><h2>System Health</h2><p>{health.length === 0 ? 'Checking services…' : down ? `${down} check${down > 1 ? 's' : ''} need attention` : 'All services operational'}</p></div><span className="status-dot" style={{ color: down ? '#ff727c' : undefined }}>●</span></div>{health.map(h => <div className="health-row" key={h.label}><span>{h.label}</span><b style={{ color: h.ok ? undefined : '#ff727c' }}>{h.value}</b></div>)}</div>
}

function DetailDrawer({ incident, close, now }: { incident: IncidentRow; close: () => void; now: number }) {
  const [events, setEvents] = useState<EventRow[] | null>(null)
  const [error, setError] = useState('')
  useEffect(() => { getJson<EventRow[]>(`/api/incidents/${encodeURIComponent(incident.id)}/events`).then(setEvents).catch(e => setError(e.message)) }, [incident.id])
  const count = (key: keyof EventRow) => { const m = new Map<string, number>(); (events || []).forEach(e => { const v = e[key]; if (v) m.set(String(v), (m.get(String(v)) || 0) + 1) }); return [...m.entries()] }
  const hosts = count('hostname'), users = count('username'), sources = count('source_ip')
  const commands = (events || []).filter(e => e.command_line)
  const network = (events || []).filter(e => e.destination_ip)
  const c = conf(incident.ai_confidence)
  return <div className="drawer-backdrop" onClick={close}><section className="drawer" onClick={e => e.stopPropagation()}><div className="drawer-head"><div><Pill tone={tone(incident.severity)}>{cap(incident.severity)} severity</Pill><h2>{incident.title}</h2><span className="muted">{incident.id} · detected {ago(incident.detected_at, now)}</span></div><button className="icon-btn" onClick={close}><X size={18} /></button></div><div className="drawer-actions"><button className="primary" onClick={() => { window.location.href = `/incidents/${encodeURIComponent(incident.id)}` }}>Start investigation</button><button className="filter"><MoreHorizontal size={15} /></button></div>
    <div className="drawer-section"><h3>Risk assessment <span className="score">{incident.risk_score}/100</span></h3><div className="risk-bar"><span style={{ width: `${incident.risk_score}%` }} /></div><p className="muted">{c !== null ? `AI confidence ${c}%. ` : ''}Correlated with {incident.correlated_event_count} event{incident.correlated_event_count === 1 ? '' : 's'} across {incident.affected_entity_count} entit{incident.affected_entity_count === 1 ? 'y' : 'ies'}.{incident.detection_source ? ` Source: ${incident.detection_source}.` : ''}</p></div>
    <div className="drawer-section"><h3>Entities involved</h3>{error && <p className="muted">{error}</p>}{!events && !error && <p className="muted">Loading events…</p>}{events && hosts.length + users.length + sources.length === 0 && <p className="muted">No entities recorded on this incident's events.</p>}
      {hosts.map(([h, n]) => <div className="entity-card" key={'h' + h}><Server size={16} /><div><b>{h}</b><small>Host · {n} event{n > 1 ? 's' : ''}</small></div></div>)}
      {users.map(([u, n]) => <div className="entity-card" key={'u' + u}><UserRound size={16} /><div><b>{u}</b><small>User · {n} event{n > 1 ? 's' : ''}</small></div></div>)}
      {sources.map(([ip, n]) => <div className="entity-card" key={'s' + ip}><Globe2 size={16} /><div><b>{ip}</b><small>Source IP · {n} event{n > 1 ? 's' : ''}</small></div></div>)}
    </div>
    <div className="drawer-section"><h3>Detection evidence</h3>{events && commands.length + network.length === 0 && <p className="muted">No command-line or network evidence on these events.</p>}
      {commands.map(e => <div className="evidence" key={e.id}><Terminal size={15} /><code>{e.command_line}</code></div>)}
      {network.map(e => <div className="evidence" key={e.id}><Network size={15} /><code>{e.destination_ip}{e.destination_port ? `:${e.destination_port}` : ''}</code></div>)}
    </div></section></div>
}

function EventsTable({ events, now, onIncident, title, subtitle, error, loading }: any) {
  return <div className="panel incident-panel"><div className="panel-head"><div><h2>{title}</h2><p>{subtitle}</p></div></div>{error && <p className="muted" style={{ padding: '0 17px' }}>{error}</p>}<div className="table-wrap"><table><thead><tr><th>Severity</th><th>Event</th><th>User</th><th>Host</th><th>Source IP</th><th>Detection</th><th>MITRE</th><th>Incident</th><th>Time</th></tr></thead><tbody>
    {loading && <tr><td colSpan={9} className="muted">Loading…</td></tr>}
    {!loading && events.length === 0 && <tr><td colSpan={9} className="muted">No events.</td></tr>}
    {events.map((e: EventRow) => <tr key={e.id} onClick={() => e.incident_id && onIncident(e.incident_id)}><td><Pill tone={tone(e.severity)}>{cap(e.severity)}</Pill></td><td><b>{cap(e.event_type)}</b><small>{e.id} · {e.source}{e.process_name ? ` · ${e.process_name}` : ''}</small></td><td>{e.username || '—'}</td><td>{e.hostname || '—'}</td><td>{e.source_ip || '—'}</td><td>{cap(e.detection_method)}</td><td>{e.mitre_technique || '—'}</td><td>{e.incident_id || '—'}</td><td className="muted" title={e.timestamp}>{ago(e.timestamp, now)}</td></tr>)}
  </tbody></table></div></div>
}

function NewInvestigation({ close, onCreated }: { close: () => void; onCreated: (id: string) => void }) {
  const [title, setTitle] = useState('')
  const [severity, setSeverity] = useState('medium')
  const [summary, setSummary] = useState('')
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)
  const submit = async () => {
    if (title.trim().length < 3) return setError('Title must be at least 3 characters.')
    setSaving(true); setError('')
    try {
      const row = await getJson<{ id: string }>('/api/incidents', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ title, severity, summary }) })
      onCreated(row.id)
    } catch (e: any) { setError(e.message) } finally { setSaving(false) }
  }
  return <div className="modal-backdrop"><div className="confirm-modal"><button className="icon-btn" onClick={close}><X size={16} /></button><Plus size={24} /><h2>New investigation</h2><p>Creates an incident record with status “new”.</p>
    <div className="ml-form" style={{ marginTop: 14 }}><label className="ml-field"><span>Title</span><input className="ml-input" value={title} onChange={e => setTitle(e.target.value)} maxLength={200} autoFocus /></label><label className="ml-field"><span>Severity</span><select className="ml-input" value={severity} onChange={e => setSeverity(e.target.value)}>{['low', 'medium', 'high', 'critical'].map(s => <option key={s} value={s}>{cap(s)}</option>)}</select></label><label className="ml-field"><span>Summary (optional)</span><textarea className="ml-input" rows={3} value={summary} onChange={e => setSummary(e.target.value)} maxLength={2000} /></label>{error && <div className="ml-error">{error}</div>}</div>
    <div><button className="filter" onClick={close}>Cancel</button><button className="primary" disabled={saving} onClick={submit}>{saving ? 'Creating…' : 'Create investigation'}</button></div></div></div>
}

// ------------------------------------------------------------------ console
export default function SocConsole({ profile }: { profile: AnalystProfile }) {
  const [collapsed, setCollapsed] = useState(false)
  const [active, setActive] = useState('Overview')
  const [selected, setSelected] = useState<IncidentRow | null>(null)
  const [search, setSearch] = useState(false)
  const [query, setQuery] = useState('')
  const [toast, setToast] = useState('')
  const [creating, setCreating] = useState(false)
  const [overview, setOverview] = useState<Overview | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [metric, setMetric] = useState('events')
  const [range, setRange] = useState('all')
  const [mlHealth, setMlHealth] = useState<{ ok: boolean; value: string } | null>(null)
  const [events, setEvents] = useState<EventRow[] | null>(null)
  const [eventsError, setEventsError] = useState('')
  const [now, setNow] = useState(0)

  const notify = (msg: string) => { setToast(msg); setTimeout(() => setToast(''), 2500) }

  useEffect(() => { setNow(Date.now()); const t = setInterval(() => setNow(Date.now()), 30000); return () => clearInterval(t) }, [])

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const data = await getJson<Overview>(`/api/overview?range=${range}&metric=${metric}`)
      setOverview(data); setError(''); setNow(Date.now())
      return true
    } catch (e: any) { setError(e.message); return false } finally { setLoading(false) }
  }, [range, metric])
  const checkMl = useCallback(async () => {
    try {
      const res = await fetch('/ml-api/health'); const body = await res.json().catch(() => null)
      setMlHealth(res.ok && body?.status === 'ok' ? { ok: true, value: 'Operational' } : { ok: false, value: body?.status ? cap(body.status) : 'Unreachable' })
    } catch { setMlHealth({ ok: false, value: 'Unreachable' }) }
  }, [])
  const loadEvents = useCallback(async () => {
    try { setEvents(await getJson<EventRow[]>('/api/events?limit=1000')); setEventsError('') } catch (e: any) { setEventsError(e.message) }
  }, [])

  useEffect(() => { load() }, [load])
  useEffect(() => { checkMl() }, [checkMl])
  useEffect(() => { if ((active === 'Events' || active === 'Alerts' || search) && !events) loadEvents() }, [active, search, events, loadEvents])

  const m = overview?.metrics || {}
  const incidents = overview?.incidents || []
  const counts = { alerts: m.alerts || 0, incidents: m.active_incidents || 0 }
  const health = useMemo(() => {
    const rows: { label: string; value: string; ok: boolean }[] = []
    if (overview || error) rows.push({ label: 'Database', value: overview ? `Connected · ${overview.health.database_latency_ms} ms` : 'Unavailable', ok: !!overview && !error })
    if (overview) rows.push({ label: 'Event pipeline', value: m.last_event_ingested_at ? `Last event ${ago(m.last_event_ingested_at, now)}` : 'No events ingested', ok: !!m.last_event_ingested_at })
    if (mlHealth) rows.push({ label: 'ML service', value: mlHealth.value, ok: mlHealth.ok })
    if (overview) rows.push({ label: 'Detection data', value: `${m.events_total} event${m.events_total === 1 ? "" : "s"} · ${m.total_incidents} incident${m.total_incidents === 1 ? "" : "s"}`, ok: true })
    return rows
  }, [overview, error, mlHealth, m, now])
  const degraded = health.some(h => !h.ok)

  const nowDate = now ? new Date(now) : null
  const stamp = nowDate ? `${nowDate.toLocaleDateString('en-US', { month: 'short', day: '2-digit', year: 'numeric', timeZone: 'UTC' }).toUpperCase()} ${nowDate.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', timeZone: 'UTC' })} UTC` : ''
  const hour = nowDate ? nowDate.getHours() : 12
  const greeting = `Good ${hour < 12 ? 'morning' : hour < 18 ? 'afternoon' : 'evening'}${profile.name ? `, ${profile.name.split(' ')[0]}` : ''}`

  const evDelta = (m.events_24h ?? 0) - (m.events_prev_24h ?? 0)
  const results = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return []
    const inc = incidents.filter(x => [x.id, x.title, x.primary_user, x.primary_host, x.summary].some(v => v?.toLowerCase().includes(q))).map(x => ({ key: x.id, label: `${x.id} · ${x.title}`, kind: 'Incident', incident: x.id }))
    const ev = (events || []).filter(e => [e.id, e.event_type, e.username, e.hostname, e.source_ip, e.process_name, e.command_line].some(v => v?.toLowerCase().includes(q))).slice(0, 8).map(e => ({ key: e.id, label: `${e.id} · ${cap(e.event_type)}${e.hostname ? ` on ${e.hostname}` : ''}`, kind: 'Event', incident: e.incident_id }))
    return [...inc, ...ev].slice(0, 12)
  }, [query, incidents, events])
  const openIncident = (id: string) => { const x = incidents.find(i => i.id === id); if (x) { setSearch(false); setSelected(x) } }

  const [detecting, setDetecting] = useState(false)
  const runDetection = async () => {
    setDetecting(true)
    try {
      const res = await fetch('/ml-api/soc/detect', { method: 'POST' })
      const body = await res.json().catch(() => null)
      if (!res.ok) throw new Error(body?.detail || `ML service error (${res.status})`)
      await load()
      notify(`ML detection: ${body.flows_flagged} of ${body.flows_scored} flows flagged → ${body.incidents_created.length} investigations opened`)
    } catch (e: any) { notify(`ML detection failed: ${e.message}`) } finally { setDetecting(false) }
  }
  const refresh = async () => { const ok = await load(); checkMl(); setEvents(null); notify(ok ? 'Dashboard refreshed' : 'Refresh failed — see error above') }

  const pageText: Record<string, string> = {
    Overview: 'Here’s what’s happening across your security environment.',
    'ML Analytics': 'Machine learning lab: probability, supervised models, clustering, sequence models and ensembles on your SOC datasets.',
    Incidents: 'All incidents, ordered by risk.', Alerts: 'High and critical severity events.', Events: 'Every ingested security event, newest first.',
    'MITRE ATT&CK': 'Techniques observed in ingested events.', 'AI Analyst': 'Model-scored investigations awaiting analyst review.',
  }

  return <div className="soc-app"><Sidebar {...{ collapsed, setCollapsed, active, setActive, counts, profile }} /><main className="main"><header className="topbar"><div className="crumb"><button className="mobile-menu"><Menu size={18} /></button><span>Workspace</span><ChevronRight size={14} /><b>{active}</b></div><div className="top-actions"><button className="global-search" onClick={() => setSearch(true)}><Search size={15} /><span>Search anything</span><kbd><Command size={11} /> K</kbd></button><button className="icon-btn notification" onClick={() => notify(m.new_incidents ? `${m.new_incidents} new incident${m.new_incidents > 1 ? 's' : ''} awaiting triage` : 'No new incidents')}><Bell size={17} />{m.new_incidents ? <i /> : null}</button><div className="top-avatar" title={profile.name}>{initials(profile.name)}</div></div></header><div className="content"><div className="page-heading"><div><div className="eyebrow"><span className="live-dot" style={degraded ? { background: '#ff727c', boxShadow: '0 0 7px #ff727c' } : undefined} /> {degraded ? 'SYSTEM DEGRADED' : 'SYSTEM OPERATIONAL'} <span>·</span> {stamp}</div><h1>{active === 'Overview' ? greeting : `/${active.toLowerCase()}`}</h1><p>{pageText[active] || `Operational workspace for ${active.toLowerCase()}.`}</p></div><div className="heading-actions"><button className="filter" onClick={refresh} disabled={loading}><RefreshCw size={14} /> {loading ? 'Refreshing…' : 'Refresh'}</button><button className="primary" onClick={() => setCreating(true)}><Plus size={15} /> New investigation</button></div></div>
    {error && <div className="ml-error" style={{ marginBottom: 14 }}>{error}</div>}
    {active === 'Overview' ? <><div className="metrics">
      <Metric label="Critical Alerts" value={overview ? m.critical_alerts : '—'} delta={overview ? `+${m.critical_alerts_24h} in 24h` : ''} note="critical-severity events" icon={ShieldAlert} tone={m.critical_alerts_24h ? 'red' : 'slate'} />
      <Metric label="Active Incidents" value={overview ? m.active_incidents : '—'} delta={overview ? `${m.new_incidents} new` : ''} note={overview ? `of ${m.total_incidents} total` : ''} icon={AlertOctagon} tone={m.new_incidents ? 'red' : 'slate'} />
      <Metric label="Mean Time to Detect" value={overview ? duration(m.mttd_seconds) : '—'} delta={overview ? `${m.mttd_sample} incident${Number(m.mttd_sample) === 1 ? '' : 's'}` : ''} note="first event → detection" icon={Timer} tone="slate" />
      <Metric label="Events Processed" value={overview ? compact(m.events_total) : '—'} delta={overview ? `${evDelta >= 0 ? '+' : '−'}${Math.abs(evDelta)}` : ''} note="vs previous 24h" icon={Activity} tone={evDelta > 0 ? 'green' : 'slate'} />
    </div><IncidentTable incidents={incidents.filter(x => ACTIVE.includes(x.status)).slice(0, 6)} onSelect={setSelected} now={now} onViewAll={() => setActive('Incidents')} /><div className="dashboard-grid"><ThreatActivity series={overview?.series} metric={metric} setMetric={setMetric} range={range} setRange={setRange} loading={loading && !overview} /><AIAnalyst incidents={incidents} onSelect={setSelected} onOpen={() => setActive('AI Analyst')} /></div><Health health={health} /></>
    : active === 'ML Analytics' ? <MLAnalytics notify={notify} />
    : active === 'Incidents' ? <IncidentTable incidents={incidents} onSelect={setSelected} now={now} title="All Incidents" subtitle={`${incidents.length} incident${incidents.length === 1 ? '' : 's'} in the database.`} />
    : active === 'Alerts' ? <EventsTable events={(events || []).filter(e => ['high', 'critical'].includes(tone(e.severity)))} now={now} onIncident={openIncident} title="Alerts" subtitle="Events with high or critical severity." error={eventsError} loading={!events && !eventsError} />
    : active === 'Events' ? <EventsTable events={events || []} now={now} onIncident={openIncident} title="Events" subtitle={events ? `${events.length} most recent events.` : 'Loading…'} error={eventsError} loading={!events && !eventsError} />
    : active === 'MITRE ATT&CK' ? <div className="panel incident-panel"><div className="panel-head"><div><h2>Observed Techniques</h2><p>Technique names from the mitre_techniques reference table; counts from events.</p></div></div><div className="table-wrap"><table><thead><tr><th>Technique</th><th>Name</th><th>Tactic</th><th>Events</th><th>Incidents</th><th>Last seen</th></tr></thead><tbody>{(overview?.techniques || []).length === 0 && <tr><td colSpan={6} className="muted">No techniques observed.</td></tr>}{(overview?.techniques || []).map(t => <tr key={t.id}><td><b>{t.id}</b></td><td>{t.name || 'Not in reference table'}</td><td>{t.tactic || '—'}</td><td>{t.evidence_count}</td><td>{t.incident_count}</td><td className="muted">{ago(t.last_seen, now)}</td></tr>)}</tbody></table></div></div>
    : active === 'AI Analyst' ? <div className="panel incident-panel"><div className="panel-head"><div><h2>Model-scored Investigations</h2><p>Open incidents with their detection model and confidence.</p></div><button className="primary" onClick={runDetection} disabled={detecting}><Bot size={14} /> {detecting ? 'Scoring flows…' : 'Run ML detection'}</button></div><div className="table-wrap"><table><thead><tr><th>Severity</th><th>Incident</th><th>Detection source</th><th>Model</th><th>Confidence</th><th>Status</th><th>Owner</th></tr></thead><tbody>{incidents.filter(x => ACTIVE.includes(x.status)).length === 0 && <tr><td colSpan={7} className="muted">No open investigations.</td></tr>}{incidents.filter(x => ACTIVE.includes(x.status)).map(x => <tr key={x.id} onClick={() => setSelected(x)}><td><Pill tone={tone(x.severity)}>{cap(x.severity)}</Pill></td><td><b>{x.title}</b><small>{x.id}</small></td><td>{x.detection_source || '—'}</td><td>{x.detection_model || '—'}</td><td>{conf(x.ai_confidence) !== null ? `${conf(x.ai_confidence)}%` : '—'}</td><td><Pill tone={x.status === 'new' ? 'new' : 'muted'}>{cap(x.status)}</Pill></td><td>{x.assigned_analyst || 'Unassigned'}</td></tr>)}</tbody></table></div></div>
    : <div className="panel placeholder"><h2>{active}</h2><p>{active === 'Threat Intel' ? 'No threat-intelligence feed is connected yet.' : 'Operational workspace for analyst workflows. Select Overview to return to the SOC summary.'}</p></div>}
  </div></main>
    {selected && <DetailDrawer incident={selected} close={() => setSelected(null)} now={now} />}
    {search && <div className="command-backdrop" onClick={() => setSearch(false)}><div className="command" onClick={e => e.stopPropagation()}><div className="command-input"><Search size={16} /><input autoFocus value={query} onChange={e => setQuery(e.target.value)} placeholder="Search alerts, incidents, entities..." onKeyDown={e => { if (e.key === 'Escape') setSearch(false); if (e.key === 'Enter' && results[0]?.incident) openIncident(results[0].incident) }} /></div>{query.trim() ? <div className="command-results"><small>{results.length ? 'RESULTS' : 'NO MATCHES'}</small>{results.map(r => <button key={r.kind + r.key} onClick={() => r.incident && openIncident(r.incident)} disabled={!r.incident}>{r.kind === 'Incident' ? <AlertOctagon size={14} /> : <Activity size={14} />}<span>{r.label}</span><ArrowUpRight size={13} /></button>)}</div> : <p>Try searching for an incident ID, hostname, or user.</p>}</div></div>}
    {creating && <NewInvestigation close={() => setCreating(false)} onCreated={(id) => { setCreating(false); window.location.href = `/incidents/${encodeURIComponent(id)}` }} />}
    {toast && <div className="toast">{toast}</div>}</div>
}

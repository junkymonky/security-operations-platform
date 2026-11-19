'use client'

import { useCallback, useRef, useState } from 'react'
import { AlertTriangle, Loader2 } from 'lucide-react'
import type { ColumnInfo } from './api'

// ---------------------------------------------------------------- palette
// Categorical slots validated on the panel surface #0e171e (dataviz validator: all checks pass).
export const SERIES = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767']
// Secondary encoding for scatter plots with more than three groups.
export const SHAPES = ['circle', 'square', 'triangle', 'diamond', 'cross', 'star', 'wye', 'circle'] as const
export const INK = { axis: '#61707d', grid: '#1e2d38', text: '#b4c1c9', muted: '#7d8d9c' }
export const tooltipStyle = { background: '#111a22', border: '1px solid #34414d', color: '#dce5ec', fontSize: 11 }
export const axisProps = { stroke: INK.axis, fontSize: 10, tickLine: false, axisLine: false } as const

export const color = (i: number) => SERIES[((i % SERIES.length) + SERIES.length) % SERIES.length]

/** Sensible default target: fewest-class categorical (classification) or last numeric column (regression). */
export function defaultTarget(cols: ColumnInfo[], task: 'regression' | 'classification'): string {
  if (task === 'regression') return cols.filter((c) => c.kind === 'numeric' && c.unique > 15).at(-1)?.name || ''
  const cats = cols.filter((c) => c.kind === 'categorical' && c.unique >= 2 && c.unique <= 30).sort((a, b) => a.unique - b.unique)
  return cats[0]?.name || cols.find((c) => c.unique >= 2 && c.unique <= 30)?.name || ''
}

// ---------------------------------------------------------------- formatting
export function fmt(v: unknown, digits = 3): string {
  if (v === null || v === undefined || v === '') return '—'
  if (typeof v === 'number') {
    if (!Number.isFinite(v)) return '—'
    if (Number.isInteger(v) && Math.abs(v) < 1e9) return v.toLocaleString('en-US')
    const a = Math.abs(v)
    if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(2)
    return v.toFixed(digits).replace(/\.?0+$/, '')
  }
  if (typeof v === 'boolean') return v ? 'yes' : 'no'
  return String(v)
}
export const pct = (v: number | null | undefined, digits = 1) =>
  v === null || v === undefined ? '—' : `${(v * 100).toFixed(digits)}%`
export const label = (s: string) => s.replaceAll('_', ' ')

// ---------------------------------------------------------------- async state
export function useRequest<T>() {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const seq = useRef(0)
  const run = useCallback(async (fn: () => Promise<T>) => {
    const id = ++seq.current
    setLoading(true)
    setError('')
    try {
      const result = await fn()
      if (id === seq.current) setData(result)
      return result
    } catch (e: any) {
      if (id === seq.current) setError(e?.message || 'Request failed')
      return null
    } finally {
      if (id === seq.current) setLoading(false)
    }
  }, [])
  return { data, error, loading, run, setData, setError }
}

// ---------------------------------------------------------------- layout
export function Panel({ title, subtitle, actions, children, className = '' }: {
  title: string; subtitle?: string; actions?: React.ReactNode; children: React.ReactNode; className?: string
}) {
  return (
    <section className={`panel ml-panel ${className}`}>
      <div className="panel-head">
        <div>
          <h2>{title}</h2>
          {subtitle && <p>{subtitle}</p>}
        </div>
        {actions}
      </div>
      <div className="ml-body">{children}</div>
    </section>
  )
}

export function Field({ label: text, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <label className="ml-field">
      <span>{text}</span>
      {children}
      {hint && <small>{hint}</small>}
    </label>
  )
}

export function Select({ value, onChange, options, placeholder }: {
  value: string; onChange: (v: string) => void; options: (string | [string, string])[]; placeholder?: string
}) {
  return (
    <select className="ml-input" value={value} onChange={(e) => onChange(e.target.value)}>
      {placeholder !== undefined && <option value="">{placeholder}</option>}
      {options.map((o) => {
        const [v, t] = Array.isArray(o) ? o : [o, o]
        return <option key={v} value={v}>{t}</option>
      })}
    </select>
  )
}

export function NumberInput({ value, onChange, min, max, step = 1 }: {
  value: number | string; onChange: (v: string) => void; min?: number; max?: number; step?: number | 'any'
}) {
  return <input className="ml-input" type="number" value={value} min={min} max={max} step={step} onChange={(e) => onChange(e.target.value)} />
}

/** Parse a numeric form value; returns an error string when out of range. */
export function checkNumber(raw: string | number, name: string, min: number, max: number, integer = false): string | null {
  const v = Number(raw)
  if (raw === '' || Number.isNaN(v)) return `${name} must be a number.`
  if (integer && !Number.isInteger(v)) return `${name} must be a whole number.`
  if (v < min || v > max) return `${name} must be between ${min} and ${max}.`
  return null
}

export function ColumnPicker({ columns, selected, onChange, kind }: {
  columns: ColumnInfo[]; selected: string[]; onChange: (v: string[]) => void; kind?: 'numeric' | 'categorical'
}) {
  const list = kind ? columns.filter((c) => c.kind === kind) : columns
  const toggle = (name: string) =>
    onChange(selected.includes(name) ? selected.filter((x) => x !== name) : [...selected, name])
  return (
    <div className="ml-chips">
      {list.map((c) => (
        <button type="button" key={c.name} className={selected.includes(c.name) ? 'on' : ''} onClick={() => toggle(c.name)}
          title={`${c.kind} · ${c.unique} unique · ${c.missing} missing`}>
          {c.name}
        </button>
      ))}
      <button type="button" className="ml-chip-link" onClick={() => onChange(list.map((c) => c.name))}>all</button>
      <button type="button" className="ml-chip-link" onClick={() => onChange([])}>none</button>
    </div>
  )
}

export function RunButton({ loading, children, onClick, disabled }: {
  loading: boolean; children: React.ReactNode; onClick: () => void; disabled?: boolean
}) {
  return (
    <button type="button" className="primary ml-run" onClick={onClick} disabled={loading || disabled}>
      {loading && <Loader2 size={14} className="ml-spin" />}
      {loading ? 'Running…' : children}
    </button>
  )
}

export function ErrorNote({ error }: { error: string }) {
  if (!error) return null
  return (
    <div className="ml-error" role="alert">
      <AlertTriangle size={14} />
      <span>{error}</span>
    </div>
  )
}

export function Empty({ children }: { children: React.ReactNode }) {
  return <div className="ml-empty">{children}</div>
}

export function Kpis({ items }: { items: [string, React.ReactNode, string?][] }) {
  return (
    <div className="ml-kpis">
      {items.map(([k, v, hint]) => (
        <div key={k} className="metric">
          <div className="metric-top"><span>{k}</span></div>
          <strong>{v}</strong>
          {hint && <small>{hint}</small>}
        </div>
      ))}
    </div>
  )
}

export function Table({ columns, rows, max = 500, onRow }: {
  columns: string[]; rows: React.ReactNode[][]; max?: number; onRow?: (i: number) => void
}) {
  return (
    <div className="table-wrap ml-table">
      <table>
        <thead><tr>{columns.map((c) => <th key={c}>{c}</th>)}</tr></thead>
        <tbody>
          {rows.slice(0, max).map((r, i) => (
            <tr key={i} onClick={onRow ? () => onRow(i) : undefined} className={onRow ? '' : 'ml-static'}>
              {r.map((c, j) => <td key={j}>{c}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function Legend({ items }: { items: { name: string; color: string; shape?: string }[] }) {
  return (
    <div className="ml-legend">
      {items.map((it) => (
        <span key={it.name}><i style={{ background: it.color }} className={it.shape && it.shape !== 'circle' ? 'sq' : ''} />{it.name}</span>
      ))}
    </div>
  )
}

/** Matrix heatmap: sequential (single hue, magnitude) or diverging (blue +, orange −, gray 0). */
export function Heatmap({ rows, cols, matrix, mode = 'sequential', digits = 2, title }: {
  rows: string[]; cols: string[]; matrix: (number | null)[][]; mode?: 'sequential' | 'diverging'; digits?: number; title?: string
}) {
  const flat = matrix.flat().filter((v): v is number => v !== null)
  const maxAbs = Math.max(...flat.map(Math.abs), 1e-12)
  const max = Math.max(...flat, 1e-12)
  const bg = (v: number | null) => {
    if (v === null) return 'transparent'
    if (mode === 'diverging') {
      const a = Math.min(Math.abs(v) / maxAbs, 1)
      return v >= 0 ? `rgba(57,135,229,${0.08 + a * 0.75})` : `rgba(217,89,38,${0.08 + a * 0.75})`
    }
    return `rgba(57,135,229,${0.06 + (v / max) * 0.8})`
  }
  return (
    <div className="ml-heatmap-wrap">
      {title && <h3>{title}</h3>}
      <table className="ml-heatmap">
        <thead><tr><th />{cols.map((c) => <th key={c} title={c}>{c}</th>)}</tr></thead>
        <tbody>
          {matrix.map((r, i) => (
            <tr key={i}>
              <th title={rows[i]}>{rows[i]}</th>
              {r.map((v, j) => (
                <td key={j} style={{ background: bg(v) }} title={`${rows[i]} × ${cols[j]}: ${fmt(v, 4)}`}>{fmt(v, digits)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function Tabs({ tabs, value, onChange }: { tabs: string[]; value: string; onChange: (v: string) => void }) {
  return (
    <div className="evidence-tabs ml-subtabs">
      {tabs.map((t) => (
        <button type="button" key={t} className={t === value ? 'active' : ''} onClick={() => onChange(t)}>{t}</button>
      ))}
    </div>
  )
}

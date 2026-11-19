'use client'

import { useEffect, useState } from 'react'
import { Area, Bar, BarChart, CartesianGrid, ComposedChart, Line, LineChart, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from 'recharts'
import { api, type Dataset } from './api'
import {
  axisProps, checkNumber, color, Empty, ErrorNote, Field, fmt, Heatmap, INK, Kpis, Legend, NumberInput, Panel, pct,
  RunButton, Select, Table, Tabs, tooltipStyle, useRequest,
} from './ui'

const SECTIONS = ['Markov chain', 'HMM — learn from data', 'HMM — inference playground', 'Linear dynamical system']

export default function SequenceTab({ dataset }: { dataset: Dataset | null }) {
  const [section, setSection] = useState(SECTIONS[0])
  const needs = section !== 'HMM — inference playground'
  return (
    <div className="ml-stack">
      <div className="ml-intro"><Tabs tabs={SECTIONS} value={section} onChange={setSection} /></div>
      {needs && !dataset ? (
        <Empty>Select a dataset (e.g. the auth_sequences sample) or use the inference playground, which needs no dataset.</Empty>
      ) : (
        <>
          {section === 'Markov chain' && <Markov key={dataset!.id} dataset={dataset!} />}
          {section === 'HMM — learn from data' && <HmmFit key={dataset!.id} dataset={dataset!} />}
          {section === 'HMM — inference playground' && <HmmPlayground />}
          {section === 'Linear dynamical system' && <Lds key={dataset!.id} dataset={dataset!} />}
        </>
      )}
    </div>
  )
}

const catCols = (d: Dataset) => (d.columns || []).filter((c) => c.unique >= 2 && c.unique <= 30).map((c) => c.name)
const allCols = (d: Dataset) => (d.columns || []).map((c) => c.name)
const pick = (cols: string[], ...names: string[]) => names.find((n) => cols.includes(n)) || ''

// ------------------------------------------------------------------ visual pieces
export function TransitionGraph({ names, A, title }: { names: string[]; A: number[][]; title: string }) {
  const W = 360, H = 280, cx = W / 2, cy = H / 2 + 6, R = Math.min(105, 40 + names.length * 12)
  const pos = names.map((_, i) => {
    const t = (2 * Math.PI * i) / names.length - Math.PI / 2
    return { x: cx + R * Math.cos(t), y: cy + R * Math.sin(t) }
  })
  const show = names.length <= 8
  return (
    <div className="ml-chart-card">
      <h3>{title}</h3>
      <svg viewBox={`0 0 ${W} ${H}`} className="ml-svg" role="img" aria-label="State transition graph">
        <defs>
          <marker id="ml-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="#8a99a8" />
          </marker>
        </defs>
        {A.map((row, i) => row.map((p, j) => {
          if (p < 0.03 || !show) return null
          const a = pos[i], b = pos[j]
          if (i === j) {
            const dx = a.x - cx, dy = a.y - cy, n = Math.hypot(dx, dy) || 1
            const lx = a.x + (dx / n) * 26, ly = a.y + (dy / n) * 26
            return (
              <g key={`${i}-${j}`}>
                <circle cx={lx} cy={ly} r={11} fill="none" stroke="#8a99a8" strokeOpacity={0.3 + p * 0.7} strokeWidth={0.6 + p * 3} />
                <text x={a.x + (dx / n) * 46} y={a.y + (dy / n) * 46 + 3} fill={INK.text} fontSize={9} textAnchor="middle">{p.toFixed(2)}</text>
              </g>
            )
          }
          const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2
          const nx = -(b.y - a.y), ny = b.x - a.x, len = Math.hypot(nx, ny) || 1
          const qx = mx + (nx / len) * 22, qy = my + (ny / len) * 22
          const ux = b.x - qx, uy = b.y - qy, ul = Math.hypot(ux, uy) || 1
          const ex = b.x - (ux / ul) * 20, ey = b.y - (uy / ul) * 20
          return (
            <g key={`${i}-${j}`}>
              <path d={`M${a.x},${a.y} Q${qx},${qy} ${ex},${ey}`} fill="none" stroke="#8a99a8" strokeOpacity={0.25 + p * 0.75} strokeWidth={0.6 + p * 3} markerEnd="url(#ml-arrow)" />
              <text x={qx} y={qy} fill={INK.text} fontSize={9} textAnchor="middle">{p.toFixed(2)}</text>
            </g>
          )
        }))}
        {pos.map((p, i) => (
          <g key={names[i]}>
            <circle cx={p.x} cy={p.y} r={18} fill="#111d25" stroke={color(i)} strokeWidth={2} />
            <text x={p.x} y={p.y + 3} fill="#dce5ec" fontSize={8.5} textAnchor="middle">{names[i].length > 9 ? names[i].slice(0, 8) + '…' : names[i]}</text>
          </g>
        ))}
      </svg>
      {!show && <small>Too many states to draw; see the matrix.</small>}
    </div>
  )
}

export function SequenceStrips({ rows, max = 160 }: { rows: { name: string; seq: string[]; palette: string[] }[]; max?: number }) {
  const T = Math.min(max, Math.max(...rows.map((r) => r.seq.length)))
  return (
    <div className="ml-chart-card">
      <h3>Sequences (first {T} steps)</h3>
      <div className="ml-strips">
        {rows.map((r) => (
          <div key={r.name} className="ml-strip">
            <span>{r.name}</span>
            <svg viewBox={`0 0 ${T} 1`} preserveAspectRatio="none" className="ml-strip-svg">
              {r.seq.slice(0, T).map((s, t) => <rect key={t} x={t} y={0} width={1.02} height={1} fill={color(r.palette.indexOf(s))}><title>{`t=${t}: ${s}`}</title></rect>)}
            </svg>
          </div>
        ))}
      </div>
      {rows.map((r) => <Legend key={r.name} items={r.palette.map((s, i) => ({ name: `${r.name}: ${s}`, color: color(i) }))} />).slice(0, 2)}
    </div>
  )
}

// ------------------------------------------------------------------ Markov
function Markov({ dataset }: { dataset: Dataset }) {
  const cols = catCols(dataset)
  const [column, setColumn] = useState(pick(cols, 'event') || cols[0] || '')
  const [group, setGroup] = useState(pick(allCols(dataset), 'session_id'))
  const [order, setOrder] = useState(pick(allCols(dataset), 't', 'step'))
  const [smoothing, setSmoothing] = useState('0')
  const [steps, setSteps] = useState('3')
  const req = useRequest<any>()
  const run = () => {
    const err = !column ? 'Choose a categorical sequence column.' : checkNumber(smoothing, 'Smoothing', 0, 10) || checkNumber(steps, 'n', 1, 50, true)
    if (err) return req.setError(err)
    req.run(() => api.markov(dataset.id, { column, group_column: group || null, order_column: order || null, smoothing: Number(smoothing), n_steps: Number(steps) }))
  }
  const r = req.data
  return (
    <Panel title="First-order Markov chain" subtitle="Transition probabilities are maximum-likelihood estimates Aᵢⱼ = N(i→j) / N(i→·). The stationary distribution solves π = πA.">
      <div className="ml-form-row">
        <Field label="State column"><Select value={column} onChange={setColumn} options={cols} /></Field>
        <Field label="Sequence id (optional)"><Select value={group} onChange={setGroup} options={allCols(dataset)} placeholder="Single sequence" /></Field>
        <Field label="Order by (optional)"><Select value={order} onChange={setOrder} options={allCols(dataset)} placeholder="Row order" /></Field>
        <Field label="Laplace smoothing"><NumberInput value={smoothing} onChange={setSmoothing} step="any" min={0} max={10} /></Field>
        <Field label="n for Aⁿ"><NumberInput value={steps} onChange={setSteps} min={1} max={50} /></Field>
      </div>
      <ErrorNote error={req.error} />
      <RunButton loading={req.loading} onClick={run}>Estimate chain</RunButton>
      {r && (
        <>
          <Kpis items={[['States', r.symbols.length], ['Sequences', r.n_sequences], ['Transitions observed', fmt(r.n_transitions)], ['Log-likelihood', fmt(r.log_likelihood, 1)]]} />
          <div className="ml-grid-2">
            <TransitionGraph names={r.symbols} A={r.transition_matrix} title="State transition graph" />
            <Heatmap title="Transition matrix A (row = from, column = to)" rows={r.symbols} cols={r.symbols} matrix={r.transition_matrix} />
          </div>
          <div className="ml-grid-2">
            <Heatmap title={`${r.n_steps}-step transition matrix Aⁿ`} rows={r.symbols} cols={r.symbols} matrix={r.n_step_matrix} />
            <div className="ml-chart-card">
              <h3>Initial vs stationary distribution</h3>
              <Legend items={[{ name: 'initial π₀', color: color(0) }, { name: 'stationary π', color: color(1) }]} />
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={r.symbols.map((s: string, i: number) => ({ s, initial: r.initial[i], stationary: r.stationary[i] }))} margin={{ left: -6, right: 10 }}>
                  <CartesianGrid stroke={INK.grid} vertical={false} />
                  <XAxis dataKey="s" {...axisProps} interval={0} angle={-20} textAnchor="end" height={45} />
                  <YAxis {...axisProps} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                  <Tooltip contentStyle={tooltipStyle} formatter={(v: any, n: any) => [pct(v), n]} cursor={{ fill: '#ffffff08' }} />
                  <Bar dataKey="initial" fill={color(0)} radius={[4, 4, 0, 0]} maxBarSize={18} isAnimationActive={false} />
                  <Bar dataKey="stationary" fill={color(1)} radius={[4, 4, 0, 0]} maxBarSize={18} isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          <SequenceStrips rows={[{ name: 'observed', seq: r.observed, palette: r.symbols }, { name: 'simulated', seq: r.simulated, palette: r.symbols }]} />
        </>
      )}
    </Panel>
  )
}

// ------------------------------------------------------------------ HMM results
function HmmResults({ r, trueStates }: { r: any; trueStates?: string[] }) {
  const states: string[] = r.states
  // Colour follows the entity: a true state takes the colour of the learned state it maps to,
  // so "normal" looks the same in the strips, the posterior chart and the transition graph.
  const mapping: Record<string, string> = r.evaluation?.state_mapping || {}
  const hiddenPalette: string[] = r.evaluation
    ? [...states.map((s) => mapping[s] ?? `(${s})`), ...r.evaluation.true_states.filter((t: string) => !Object.values(mapping).includes(t))]
    : trueStates || states
  const stripRows = [
    { name: 'observed', seq: r.observations, palette: r.symbols },
    ...(r.true_path ? [{ name: 'true hidden', seq: r.true_path, palette: hiddenPalette }] : []),
    ...(r.evaluation ? [{ name: 'true hidden', seq: r.evaluation.true_path, palette: hiddenPalette }] : []),
    { name: 'Viterbi', seq: r.evaluation ? r.viterbi_path.map((s: string) => mapping[s] ?? `(${s})`) : r.viterbi_path, palette: hiddenPalette },
  ]
  const scaling = r.scaling_factors.map((c: number, t: number) => ({ t, lnc: Math.log(c) }))
  return (
    <>
      <Kpis items={[
        ['ln p(X) via scaling', fmt(r.log_likelihood, 2), `length T = ${r.length}`],
        ['Unscaled p(X)', r.unscaled_underflow ? 'underflow → 0' : fmt(r.unscaled_likelihood), r.unscaled_underflow ? 'why scaling factors are needed' : 'matches exp(ln p(X))'],
        ['Viterbi ln p(X, Z*)', fmt(r.viterbi_log_prob, 2)],
        r.evaluation ? ['Viterbi accuracy', pct(r.evaluation.viterbi_accuracy), `posterior decoding ${pct(r.evaluation.posterior_decoding_accuracy)}`]
          : r.viterbi_accuracy !== undefined ? ['Viterbi accuracy', pct(r.viterbi_accuracy), 'vs sampled hidden path'] : ['Sum-product check', fmt(r.sum_product_max_abs_diff), 'max |γ − marginal|'],
      ]} />
      <div className="ml-grid-2">
        <TransitionGraph names={states} A={r.transition_matrix} title="Hidden-state transition graph" />
        <Heatmap title="Emission matrix B = p(xₜ | zₜ)" rows={states} cols={r.symbols} matrix={r.emission_matrix} />
      </div>
      <SequenceStrips rows={stripRows} />
      <div className="ml-chart-card">
        <h3>Posterior marginals γ(zₜ) = p(zₜ | X) — forward–backward / sum-product (max difference {fmt(r.sum_product_max_abs_diff)})</h3>
        <Legend items={states.map((s, i) => ({ name: r.evaluation?.state_mapping?.[s] ? `${s} (${r.evaluation.state_mapping[s]})` : s, color: color(i) }))} />
        <ResponsiveContainer width="100%" height={200}>
          <ComposedChart data={r.posterior.slice(0, 200)} margin={{ left: -6, right: 10 }}>
            <CartesianGrid stroke={INK.grid} vertical={false} />
            <XAxis dataKey="t" {...axisProps} />
            <YAxis domain={[0, 1]} {...axisProps} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
            <Tooltip contentStyle={tooltipStyle} formatter={(v: any, n: any) => [pct(v), n]} labelFormatter={(v) => `t = ${v}`} />
            {states.map((s, i) => <Area key={s} dataKey={s} stackId="1" type="stepAfter" stroke={color(i)} fill={color(i)} fillOpacity={0.55} isAnimationActive={false} />)}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div className="ml-grid-2">
        <div className="ml-chart-card">
          <h3>Scaling factors ln cₜ (ln p(X) = Σ ln cₜ)</h3>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={scaling.slice(0, 200)} margin={{ left: -6, right: 10 }}>
              <CartesianGrid stroke={INK.grid} vertical={false} />
              <XAxis dataKey="t" {...axisProps} />
              <YAxis {...axisProps} tickFormatter={(v) => fmt(v, 2)} />
              <Tooltip contentStyle={tooltipStyle} formatter={(v: any) => [fmt(v, 4), 'ln cₜ']} labelFormatter={(v) => `t = ${v}`} />
              <Line dataKey="lnc" stroke={color(0)} strokeWidth={1.5} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <Table columns={['t', 'obs', ...states.map((s) => `α̂ ${s}`), ...states.map((s) => `β̂ ${s}`)]}
          rows={r.alpha_hat.slice(0, 10).map((a: number[], t: number) => [t, r.observations[t], ...a.map((v) => fmt(v, 3)), ...r.beta_hat[t].map((v: number) => fmt(v, 3))])} />
      </div>
    </>
  )
}

function HmmFit({ dataset }: { dataset: Dataset }) {
  const cols = catCols(dataset), all = allCols(dataset)
  const [column, setColumn] = useState(pick(cols, 'event') || cols[0] || '')
  const [group, setGroup] = useState(pick(all, 'session_id'))
  const [order, setOrder] = useState(pick(all, 't', 'step'))
  const [hidden, setHidden] = useState(pick(cols, 'phase'))
  const [k, setK] = useState('3')
  const [iters, setIters] = useState('100')
  const [inits, setInits] = useState('3')
  const req = useRequest<any>()
  const run = () => {
    const err = !column ? 'Choose an observation column.' : checkNumber(k, 'Hidden states', 2, 10, true) || checkNumber(iters, 'Iterations', 1, 500, true) || checkNumber(inits, 'Restarts', 1, 10, true)
    if (err) return req.setError(err)
    req.run(() => api.hmmFit(dataset.id, { column, group_column: group || null, order_column: order || null, hidden_column: hidden || null, n_states: Number(k), n_iter: Number(iters), n_init: Number(inits) }))
  }
  const r = req.data
  return (
    <Panel title="Hidden Markov model — maximum likelihood by Baum–Welch (EM)"
      subtitle="E-step: scaled forward–backward gives γ and ξ; M-step: re-estimate π, A, B. Decoding uses Viterbi. If a hidden-state column exists it is only used to score the learned states.">
      <div className="ml-form-row">
        <Field label="Observation column"><Select value={column} onChange={setColumn} options={cols} /></Field>
        <Field label="Sequence id"><Select value={group} onChange={setGroup} options={all} placeholder="Single sequence" /></Field>
        <Field label="Order by"><Select value={order} onChange={setOrder} options={all} placeholder="Row order" /></Field>
        <Field label="True hidden states (eval only)"><Select value={hidden} onChange={setHidden} options={cols} placeholder="None" /></Field>
      </div>
      <div className="ml-form-row">
        <Field label="Hidden states K"><NumberInput value={k} onChange={setK} min={2} max={10} /></Field>
        <Field label="Max EM iterations"><NumberInput value={iters} onChange={setIters} min={1} max={500} /></Field>
        <Field label="Random restarts"><NumberInput value={inits} onChange={setInits} min={1} max={10} /></Field>
      </div>
      <ErrorNote error={req.error} />
      <RunButton loading={req.loading} onClick={run}>Learn HMM</RunButton>
      {r && (
        <>
          <p className="ml-note">{r.n_sequences} sequences · {fmt(r.total_length)} observations · visualisations show the first sequence.{r.evaluation ? ` Learned→true state mapping: ${Object.entries(r.evaluation.state_mapping).map(([a, b]) => `${a}→${b}`).join(', ')}.` : ''}</p>
          <div className="ml-chart-card">
            <h3>Baum–Welch log-likelihood by iteration (monotone non-decreasing)</h3>
            <ResponsiveContainer width="100%" height={170}>
              <LineChart data={r.em_trace} margin={{ left: 0, right: 10 }}>
                <CartesianGrid stroke={INK.grid} vertical={false} />
                <XAxis dataKey="iteration" {...axisProps} />
                <YAxis {...axisProps} domain={['auto', 'auto']} tickFormatter={(v) => fmt(v, 0)} width={55} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v: any) => [fmt(v, 2), 'ln p(X|θ)']} labelFormatter={(v) => `iteration ${v}`} />
                <Line dataKey="log_likelihood" stroke={color(0)} strokeWidth={2} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <HmmResults r={r} />
          {r.evaluation && <Heatmap title="Learned state (row) vs true state (column) counts — Viterbi path" rows={r.states} cols={r.evaluation.true_states} matrix={r.evaluation.confusion} digits={0} />}
        </>
      )}
    </Panel>
  )
}

// ------------------------------------------------------------------ playground
const parseMatrix = (text: string) => text.trim().split('\n').filter(Boolean).map((row) => row.split(/[\s,]+/).filter(Boolean).map(Number))
const parseList = (text: string) => text.split(/[\s,]+/).map((s) => s.trim()).filter(Boolean)

function HmmPlayground() {
  const [states, setStates] = useState('normal, attack')
  const [symbols, setSymbols] = useState('login_ok, login_fail, exfil')
  const [pi, setPi] = useState('0.9 0.1')
  const [A, setA] = useState('0.95 0.05\n0.10 0.90')
  const [B, setB] = useState('0.85 0.13 0.02\n0.15 0.55 0.30')
  const [mode, setMode] = useState<'sample' | 'observe'>('sample')
  const [length, setLength] = useState('300')
  const [obs, setObs] = useState('login_ok login_ok login_fail login_fail exfil login_fail login_ok')
  const req = useRequest<any>()
  const run = () => {
    const s = parseList(states), sym = parseList(symbols)
    const p = parseList(pi).map(Number), a = parseMatrix(A), b = parseMatrix(B)
    if ([p, ...a, ...b].flat().some((v) => Number.isNaN(v))) return req.setError('Matrices must contain only numbers.')
    if (mode === 'sample') {
      const err = checkNumber(length, 'Sequence length', 2, 5000, true)
      if (err) return req.setError(err)
    }
    req.run(() => api.hmmDecode({
      states: s, symbols: sym, initial: p, transition: a, emission: b,
      ...(mode === 'sample' ? { sample_length: Number(length) } : { observations: parseList(obs) }),
    }))
  }
  return (
    <Panel title="HMM inference playground" subtitle="Specify π, A and B directly, then either sample (X, Z) from the model or type an observation sequence. Runs forward, backward, sum-product and Viterbi.">
      <div className="ml-grid-2">
        <div className="ml-form">
          <Field label="Hidden states (comma separated)"><input className="ml-input" value={states} onChange={(e) => setStates(e.target.value)} /></Field>
          <Field label="Observation symbols"><input className="ml-input" value={symbols} onChange={(e) => setSymbols(e.target.value)} /></Field>
          <Field label="Initial distribution π"><input className="ml-input" value={pi} onChange={(e) => setPi(e.target.value)} /></Field>
        </div>
        <div className="ml-form">
          <Field label="Transition matrix A (one row per line)"><textarea className="ml-input ml-mono" rows={3} value={A} onChange={(e) => setA(e.target.value)} /></Field>
          <Field label="Emission matrix B (rows = states)"><textarea className="ml-input ml-mono" rows={3} value={B} onChange={(e) => setB(e.target.value)} /></Field>
        </div>
      </div>
      <div className="ml-form-row">
        <Field label="Input"><Select value={mode} onChange={(v) => setMode(v as any)} options={[['sample', 'Sample from model'], ['observe', 'Enter observations']]} /></Field>
        {mode === 'sample'
          ? <Field label="Length T" hint="try 1500 to see unscaled underflow"><NumberInput value={length} onChange={setLength} min={2} max={5000} /></Field>
          : <Field label="Observations"><input className="ml-input" value={obs} onChange={(e) => setObs(e.target.value)} /></Field>}
      </div>
      <ErrorNote error={req.error} />
      <RunButton loading={req.loading} onClick={run}>Run inference</RunButton>
      {req.data && <HmmResults r={req.data} trueStates={req.data.states} />}
    </Panel>
  )
}

// ------------------------------------------------------------------ LDS
function Lds({ dataset }: { dataset: Dataset }) {
  const nums = (dataset.columns || []).filter((c) => c.kind === 'numeric').map((c) => c.name)
  const [column, setColumn] = useState(pick(nums, 'bytes_out_kb') || nums[0] || '')
  const [order, setOrder] = useState(pick(allCols(dataset), 't'))
  const [model, setModel] = useState('random_walk')
  const [estimate, setEstimate] = useState(true)
  const [q, setQ] = useState('')
  const [r, setR] = useState('')
  const [points, setPoints] = useState('300')
  const req = useRequest<any>()
  useEffect(() => { req.setData(null) }, [model])
  const run = () => {
    if (!column) return req.setError('Choose a numeric column.')
    if (!estimate && (!(Number(q) > 0) || !(Number(r) > 0))) return req.setError('Enter positive q and r, or enable estimation.')
    const err = checkNumber(points, 'Points', 5, 5000, true)
    if (err) return req.setError(err)
    req.run(() => api.lds(dataset.id, { column, order_column: order || null, model, estimate_noise: estimate, q: q ? Number(q) : null, r: r ? Number(r) : null, max_points: Number(points), forecast: 20 }))
  }
  const d = req.data
  const series = d ? [
    ...d.series.map((s: any) => ({ ...s, band: [s.smoothed - 2 * s.smoothed_std, s.smoothed + 2 * s.smoothed_std] })),
    ...d.forecast.map((f: any) => ({ t: f.t, forecast: f.mean, fband: [f.mean - 2 * f.std, f.mean + 2 * f.std] })),
  ] : []
  return (
    <Panel title="Linear dynamical system — Kalman filter and RTS smoother"
      subtitle="zₜ = A zₜ₋₁ + w, xₜ = C zₜ + v with Gaussian noise Q = q, R = r. Noise variances can be learned by maximising the innovations log-likelihood.">
      <div className="ml-form-row">
        <Field label="Observed signal"><Select value={column} onChange={setColumn} options={nums} /></Field>
        <Field label="Order by"><Select value={order} onChange={setOrder} options={allCols(dataset)} placeholder="Row order" /></Field>
        <Field label="State model"><Select value={model} onChange={setModel} options={[['random_walk', 'Random walk (local level)'], ['constant_velocity', 'Constant velocity (level + trend)']]} /></Field>
        <Field label="Points"><NumberInput value={points} onChange={setPoints} min={5} max={5000} /></Field>
      </div>
      <div className="ml-form-row">
        <label className="ml-check"><input type="checkbox" checked={estimate} onChange={(e) => setEstimate(e.target.checked)} /> Estimate q, r by maximum likelihood</label>
        <Field label="Process noise q"><NumberInput value={q} onChange={setQ} step="any" min={0} /></Field>
        <Field label="Measurement noise r"><NumberInput value={r} onChange={setR} step="any" min={0} /></Field>
      </div>
      <ErrorNote error={req.error} />
      <RunButton loading={req.loading} onClick={run}>Run Kalman filter</RunButton>
      {d && (
        <>
          <Kpis items={[
            ['q (process)', fmt(d.q), d.noise_estimated ? 'MLE' : 'given'],
            ['r (measurement)', fmt(d.r), d.noise_estimated ? 'MLE' : 'given'],
            ['Log-likelihood', fmt(d.log_likelihood, 1)],
            ['Filter residual RMSE', fmt(d.filter_rmse)],
          ]} />
          <div className="ml-chart-card">
            <h3>Observed signal, smoothed state (±2σ) and 20-step forecast</h3>
            <Legend items={[{ name: 'observed', color: '#5b6b78' }, { name: 'filtered', color: color(1) }, { name: 'smoothed ±2σ', color: color(0) }, { name: 'forecast ±2σ', color: color(2) }]} />
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={series} margin={{ left: -6, right: 10 }}>
                <CartesianGrid stroke={INK.grid} vertical={false} />
                <XAxis dataKey="t" {...axisProps} type="number" domain={['dataMin', 'dataMax']} />
                <YAxis {...axisProps} domain={['auto', 'auto']} tickFormatter={(v) => fmt(v, 0)} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v: any, n: any) => [Array.isArray(v) ? `${fmt(v[0], 1)} – ${fmt(v[1], 1)}` : fmt(v, 2), n]} labelFormatter={(v) => `t = ${v}`} />
                <Area dataKey="band" stroke="none" fill={color(0)} fillOpacity={0.18} isAnimationActive={false} name="smoothed ±2σ" />
                <Area dataKey="fband" stroke="none" fill={color(2)} fillOpacity={0.2} isAnimationActive={false} name="forecast ±2σ" />
                <Scatter dataKey="observed" fill="#5b6b78" isAnimationActive={false} name="observed" shape={(p: any) => <circle cx={p.cx} cy={p.cy} r={1.6} fill="#5b6b78" />} />
                <Line dataKey="filtered" stroke={color(1)} strokeWidth={1.2} dot={false} isAnimationActive={false} />
                <Line dataKey="smoothed" stroke={color(0)} strokeWidth={2} dot={false} isAnimationActive={false} />
                <Line dataKey="forecast" stroke={color(2)} strokeWidth={2} dot={false} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <p className="ml-note">A = {JSON.stringify(d.A)}, C = {JSON.stringify(d.C)}. Innovation std {fmt(d.innovation_std)}.</p>
        </>
      )}
    </Panel>
  )
}

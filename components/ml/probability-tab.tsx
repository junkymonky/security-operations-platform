'use client'

import { useEffect, useMemo, useState } from 'react'
import { Area, Bar, BarChart, CartesianGrid, ComposedChart, Line, LineChart, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from 'recharts'
import { api, type Dataset } from './api'
import {
  axisProps, checkNumber, color, ColumnPicker, Empty, ErrorNote, Field, fmt, Heatmap, INK, Kpis, label, Legend,
  NumberInput, Panel, pct, RunButton, Select, Table, Tabs, tooltipStyle, useRequest,
} from './ui'

const SECTIONS = ['Random variables', 'Covariance', 'Probability density', "Bayes' rule", 'Polynomial curve fitting']

export default function ProbabilityTab({ dataset }: { dataset: Dataset }) {
  const [section, setSection] = useState(SECTIONS[0])
  return (
    <div className="ml-stack">
      <div className="ml-intro">
        <Tabs tabs={SECTIONS} value={section} onChange={setSection} />
      </div>
      {section === 'Random variables' && <Describe dataset={dataset} />}
      {section === 'Covariance' && <Covariance dataset={dataset} />}
      {section === 'Probability density' && <Density dataset={dataset} />}
      {section === "Bayes' rule" && <Bayes dataset={dataset} />}
      {section === 'Polynomial curve fitting' && <Polynomial dataset={dataset} />}
    </div>
  )
}

const numericCols = (d: Dataset) => (d.columns || []).filter((c) => c.kind === 'numeric').map((c) => c.name)

// ------------------------------------------------------------------ describe
function Describe({ dataset }: { dataset: Dataset }) {
  const req = useRequest<any>()
  const [cols, setCols] = useState<string[]>([])
  useEffect(() => { setCols((dataset.columns || []).slice(0, 6).map((c) => c.name)) }, [dataset.id])
  const run = () => cols.length ? req.run(() => api.describe(dataset.id, cols)) : req.setError('Select at least one column.')
  return (
    <Panel title="Discrete and continuous random variables"
      subtitle="Columns with ≤15 distinct values are treated as discrete (empirical PMF); others as continuous (moments and quantiles).">
      <Field label="Columns"><ColumnPicker columns={dataset.columns || []} selected={cols} onChange={setCols} /></Field>
      <RunButton loading={req.loading} onClick={run}>Compute statistics</RunButton>
      <ErrorNote error={req.error} />
      {req.data && (
        <>
          {req.data.continuous.length > 0 && (
            <Table columns={['Variable', 'n', 'Mean E[X]', 'Median', 'Mode', 'Var (unbiased)', 'Var (MLE)', 'Std', 'IQR', 'Skew', 'Kurtosis', 'q05', 'q95']}
              rows={req.data.continuous.map((c: any) => [<b key="c">{c.column}</b>, fmt(c.count), fmt(c.mean), fmt(c.median), fmt(c.mode), fmt(c.variance), fmt(c.variance_population), fmt(c.std), fmt(c.iqr), fmt(c.skewness), fmt(c.kurtosis), fmt(c.quantiles.q5), fmt(c.quantiles.q95)])} />
          )}
          <div className="ml-grid-3">
            {req.data.discrete.map((d: any) => (
              <div key={d.column} className="ml-chart-card">
                <h3>{d.column} <span className="muted">· H = {fmt(d.entropy_bits)} bits{d.expectation !== undefined ? ` · E[X] = ${fmt(d.expectation)}` : ''}</span></h3>
                <ResponsiveContainer width="100%" height={180}>
                  <BarChart data={d.pmf} margin={{ left: -10, right: 8 }}>
                    <CartesianGrid stroke={INK.grid} vertical={false} />
                    <XAxis dataKey="value" {...axisProps} interval={0} angle={d.pmf.length > 4 ? -25 : 0} textAnchor={d.pmf.length > 4 ? 'end' : 'middle'} height={d.pmf.length > 4 ? 50 : 20} />
                    <YAxis {...axisProps} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                    <Tooltip contentStyle={tooltipStyle} formatter={(v: any) => [pct(v), 'P(X = x)']} cursor={{ fill: '#ffffff08' }} />
                    <Bar dataKey="p" fill={color(0)} radius={[4, 4, 0, 0]} maxBarSize={36} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ))}
          </div>
        </>
      )}
    </Panel>
  )
}

// ------------------------------------------------------------------ covariance
function Covariance({ dataset }: { dataset: Dataset }) {
  const req = useRequest<any>()
  const [cols, setCols] = useState<string[]>([])
  const [method, setMethod] = useState('pearson')
  useEffect(() => { setCols(numericCols(dataset).slice(0, 6)) }, [dataset.id])
  const run = () => cols.length < 2 ? req.setError('Select at least two numeric columns.') : req.run(() => api.covariance(dataset.id, { columns: cols, method }))
  return (
    <Panel title="Expectation, covariance and correlation" subtitle="Cov(X,Y) = E[(X−E[X])(Y−E[Y])]; correlation normalises covariance to [−1, 1].">
      <div className="ml-form-row">
        <Field label="Numeric columns"><ColumnPicker columns={dataset.columns || []} kind="numeric" selected={cols} onChange={setCols} /></Field>
        <Field label="Correlation"><Select value={method} onChange={setMethod} options={['pearson', 'spearman', 'kendall']} /></Field>
      </div>
      <RunButton loading={req.loading} onClick={run}>Compute matrices</RunButton>
      <ErrorNote error={req.error} />
      {req.data && (
        <>
          <div className="ml-grid-2">
            <Heatmap title={`${label(req.data.method)} correlation`} rows={req.data.columns} cols={req.data.columns} matrix={req.data.correlation} mode="diverging" />
            <Heatmap title="Sample covariance" rows={req.data.columns} cols={req.data.columns} matrix={req.data.covariance} mode="diverging" digits={2} />
          </div>
          <p className="ml-note">
            Identity check on <b>{req.data.identity_check.x}</b>, <b>{req.data.identity_check.y}</b>: E[XY] − E[X]E[Y] = {fmt(req.data.identity_check['E[XY]'])} − {fmt(req.data.identity_check['E[X]E[Y]'])} = <b>{fmt(req.data.identity_check.cov_population)}</b>; NumPy population covariance = <b>{fmt(req.data.identity_check.cov_population_numpy)}</b> (n = {fmt(req.data.n)}).
          </p>
        </>
      )}
    </Panel>
  )
}

// ------------------------------------------------------------------ density
function Density({ dataset }: { dataset: Dataset }) {
  const req = useRequest<any>()
  const cols = numericCols(dataset)
  const [column, setColumn] = useState(cols[0] || '')
  const [bins, setBins] = useState('30')
  const [lower, setLower] = useState('')
  const [upper, setUpper] = useState('')
  useEffect(() => { setColumn(numericCols(dataset)[0] || ''); req.setData(null) }, [dataset.id])
  const run = () => {
    const err = !column ? 'Choose a numeric column.' : checkNumber(bins, 'Bins', 5, 100, true)
    if (err) return req.setError(err)
    req.run(() => api.distribution(dataset.id, { column, bins: Number(bins), lower: lower === '' ? null : Number(lower), upper: upper === '' ? null : Number(upper) }))
  }
  const d = req.data
  const steps = useMemo(() => d ? d.histogram.flatMap((h: any) => [{ x: h.x0, density: h.density }, { x: h.x1, density: h.density }]) : [], [d])
  return (
    <Panel title="Continuous random variables and probability densities"
      subtitle="Histogram density, Gaussian kernel density estimate and maximum-likelihood parametric fits ranked by AIC.">
      <div className="ml-form-row">
        <Field label="Column"><Select value={column} onChange={setColumn} options={cols} /></Field>
        <Field label="Bins"><NumberInput value={bins} onChange={setBins} min={5} max={100} /></Field>
        <Field label="P(a ≤ X ≤ b): a" hint="blank = min"><NumberInput value={lower} onChange={setLower} step="any" /></Field>
        <Field label="b" hint="blank = max"><NumberInput value={upper} onChange={setUpper} step="any" /></Field>
      </div>
      <RunButton loading={req.loading} onClick={run}>Estimate density</RunButton>
      <ErrorNote error={req.error} />
      {d && (
        <>
          <Kpis items={[
            ['E[X]', fmt(d.moments.expectation)],
            ['Var[X] (MLE)', fmt(d.moments.variance_mle), `unbiased ${fmt(d.moments.variance_unbiased)}`],
            ['Median', fmt(d.moments.median)],
            [`P(${fmt(d.interval_probability.lower)} ≤ X ≤ ${fmt(d.interval_probability.upper)})`, pct(d.interval_probability.empirical), `normal model ${pct(d.interval_probability.normal_model)}`],
          ]} />
          <div className="ml-grid-2">
            <div className="ml-chart-card">
              <h3>Density p(x)</h3>
              <Legend items={[{ name: 'histogram', color: '#4a5a67' }, ...d.curve_series.map((s: string, i: number) => ({ name: s === 'kde' ? 'KDE' : `${s} (MLE)`, color: color(i) }))]} />
              <ResponsiveContainer width="100%" height={260}>
                <ComposedChart margin={{ left: -6, right: 10 }}>
                  <CartesianGrid stroke={INK.grid} vertical={false} />
                  <XAxis type="number" dataKey="x" domain={['dataMin', 'dataMax']} {...axisProps} tickFormatter={(v) => fmt(v, 2)} allowDuplicatedCategory={false} />
                  <YAxis {...axisProps} tickFormatter={(v) => fmt(v, 3)} />
                  <Tooltip contentStyle={tooltipStyle} formatter={(v: any, n: any) => [fmt(v, 4), n]} labelFormatter={(v) => `x = ${fmt(v, 3)}`} />
                  <Area data={steps} dataKey="density" type="linear" stroke="#4a5a67" fill="#4a5a67" fillOpacity={0.35} isAnimationActive={false} name="histogram" />
                  {d.curve_series.map((s: string, i: number) => (
                    <Line key={s} data={d.curve} dataKey={s} stroke={color(i)} strokeWidth={2} dot={false} isAnimationActive={false} name={s} />
                  ))}
                </ComposedChart>
              </ResponsiveContainer>
            </div>
            <div className="ml-chart-card">
              <h3>Empirical CDF F(x) = P(X ≤ x)</h3>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={d.ecdf} margin={{ left: -6, right: 10 }}>
                  <CartesianGrid stroke={INK.grid} vertical={false} />
                  <XAxis type="number" dataKey="x" domain={['dataMin', 'dataMax']} {...axisProps} tickFormatter={(v) => fmt(v, 2)} />
                  <YAxis {...axisProps} domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                  <Tooltip contentStyle={tooltipStyle} formatter={(v: any) => [pct(v), 'F(x)']} labelFormatter={(v) => `x = ${fmt(v, 3)}`} />
                  <Line dataKey="p" type="stepAfter" stroke={color(0)} strokeWidth={2} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="ml-grid-2">
            <Table columns={['Family', 'Parameters', 'Log-likelihood', 'AIC']}
              rows={d.fits.map((f: any, i: number) => [<b key="f">{i === 0 ? `${f.family} (best)` : f.family}</b>, f.params.map((p: number) => fmt(p)).join(', '), fmt(f.log_likelihood), fmt(f.aic)])} />
            <Table columns={['Quantile', 'Empirical', 'Normal model']}
              rows={Object.keys(d.quantiles).map((q) => [q, fmt(d.quantiles[q]), fmt(d.normal_quantiles[q])])} />
          </div>
          <p className="ml-note">{d.normality_test.test}: W = {fmt(d.normality_test.statistic)}, p = {fmt(d.normality_test.p_value)} — {d.normality_test.p_value < 0.05 ? 'normality rejected at 5%' : 'no evidence against normality at 5%'}.</p>
        </>
      )}
    </Panel>
  )
}

// ------------------------------------------------------------------ Bayes
function EventPicker({ dataset, title, state, set }: { dataset: Dataset; title: string; state: any; set: (s: any) => void }) {
  const col = (dataset.columns || []).find((c) => c.name === state.column)
  const values = dataset.summary?.categorical?.[state.column]?.map((v) => v.value) || []
  return (
    <div className="ml-event">
      <h3>{title}</h3>
      <div className="ml-form-row">
        <Field label="Column"><Select value={state.column} onChange={(v) => set({ column: v, value: '', op: '>=', threshold: '' })} options={(dataset.columns || []).map((c) => c.name)} placeholder="Select…" /></Field>
        {col?.kind === 'numeric' ? (
          <>
            <Field label="Operator"><Select value={state.op} onChange={(v) => set({ ...state, op: v })} options={['>=', '>', '<=', '<']} /></Field>
            <Field label="Threshold"><NumberInput value={state.threshold} onChange={(v) => set({ ...state, threshold: v })} step="any" /></Field>
          </>
        ) : col ? (
          <Field label="Equals"><Select value={state.value} onChange={(v) => set({ ...state, value: v })} options={values} placeholder="Select…" /></Field>
        ) : null}
      </div>
    </div>
  )
}

function Bayes({ dataset }: { dataset: Dataset }) {
  const req = useRequest<any>()
  const blank = { column: '', value: '', op: '>=', threshold: '' }
  const [A, setA] = useState<any>(blank)
  const [B, setB] = useState<any>(blank)
  const [cond, setCond] = useState('')
  useEffect(() => { setA(blank); setB(blank); setCond(''); req.setData(null) }, [dataset.id])

  const toEvent = (e: any, prefix: string) => {
    const kind = (dataset.columns || []).find((c) => c.name === e.column)?.kind
    if (!e.column) return `Choose a column for ${prefix}.`
    if (kind === 'numeric') {
      if (e.threshold === '' || Number.isNaN(Number(e.threshold))) return `Enter a numeric threshold for ${prefix}.`
      return { [`${prefix}_column`]: e.column, [`${prefix}_threshold`]: Number(e.threshold), [`${prefix}_op`]: e.op }
    }
    if (!e.value) return `Choose a value for ${prefix}.`
    return { [`${prefix}_column`]: e.column, [`${prefix}_value`]: e.value }
  }
  const run = () => {
    const a = toEvent(A, 'hypothesis'), b = toEvent(B, 'evidence')
    if (typeof a === 'string') return req.setError(a)
    if (typeof b === 'string') return req.setError(b)
    req.run(() => api.bayes(dataset.id, { ...a, ...b, condition_column: cond || null }))
  }
  const r = req.data
  return (
    <Panel title="Bayes' rule, sum and product rules, independence"
      subtitle="P(A|B) = P(B|A)·P(A) / P(B), with P(B) = P(B|A)P(A) + P(B|¬A)P(¬A). All probabilities are estimated from row frequencies.">
      <div className="ml-grid-2">
        <EventPicker dataset={dataset} title="Hypothesis A" state={A} set={setA} />
        <EventPicker dataset={dataset} title="Evidence B" state={B} set={setB} />
      </div>
      <Field label="Conditional independence given C (optional)">
        <Select value={cond} onChange={setCond} options={(dataset.columns || []).map((c) => c.name)} placeholder="None" />
      </Field>
      <RunButton loading={req.loading} onClick={run}>Apply Bayes' rule</RunButton>
      <ErrorNote error={req.error} />
      {r && (
        <>
          <p className="ml-note"><b>A</b>: {r.hypothesis} · <b>B</b>: {r.evidence} · n = {fmt(r.n)}</p>
          <Kpis items={[
            ['Prior P(A)', pct(r.prior, 2)],
            ['Likelihood P(B|A)', pct(r.likelihood, 2), `P(B|¬A) = ${pct(r.likelihood_not, 2)}`],
            ['Evidence P(B)', pct(r.evidence_sum_rule, 2), `direct count ${pct(r.evidence_prob, 2)}`],
            ['Posterior P(A|B)', pct(r.posterior, 2), `direct count ${pct(r.posterior_direct, 2)}`],
          ]} />
          <div className="ml-grid-2">
            <Heatmap title="Joint distribution P(A, B)" rows={r.joint_table.rows} cols={r.joint_table.cols} matrix={r.joint_table.probabilities} digits={3} />
            <Table columns={['Independence test', 'Value']} rows={[
              ['P(A∩B)', fmt(r.independence.p_joint, 4)],
              ['P(A)·P(B)', fmt(r.independence.p_product, 4)],
              ['Mutual information I(A;B)', `${fmt(r.independence.mutual_information_bits, 4)} bits`],
              ['χ² statistic', fmt(r.independence.chi2)],
              ['p-value', fmt(r.independence.p_value)],
              ['Independent at 5%?', r.independence.independent_at_5pct === null ? '—' : r.independence.independent_at_5pct ? 'yes' : 'no'],
            ]} />
          </div>
          {r.conditional_independence && (
            <>
              <h3 className="ml-h3">Conditional independence of A and B given {r.conditional_independence.condition_column} — I(A;B|C) = {fmt(r.conditional_independence.conditional_mutual_information_bits, 4)} bits</h3>
              <Table columns={['C = c', 'P(C=c)', 'P(A,B|c)', 'P(A|c)·P(B|c)', 'I(A;B|c) bits']}
                rows={r.conditional_independence.by_condition.map((c: any) => [c.condition, pct(c.p_c), fmt(c.p_ab_given_c, 4), fmt(c.p_a_given_c_times_p_b_given_c, 4), fmt(c.mutual_information_bits, 4)])} />
            </>
          )}
        </>
      )}
    </Panel>
  )
}

// ------------------------------------------------------------------ polynomial
function Polynomial({ dataset }: { dataset: Dataset }) {
  const req = useRequest<any>()
  const cols = numericCols(dataset)
  const [x, setX] = useState(cols[0] || '')
  const [y, setY] = useState(cols[1] || '')
  const [degree, setDegree] = useState(3)
  const [lam, setLam] = useState('0')
  useEffect(() => { const c = numericCols(dataset); setX(c[0] || ''); setY(c[1] || ''); req.setData(null) }, [dataset.id])
  const run = () => {
    if (!x || !y || x === y) return req.setError('Choose two different numeric columns.')
    if (Number(lam) < 0 || Number.isNaN(Number(lam))) return req.setError('λ must be a non-negative number.')
    req.run(() => api.polynomial(dataset.id, { x_column: x, y_column: y, degree, lam: Number(lam) }))
  }
  const r = req.data
  const train = r?.points.filter((p: any) => p.split === 'train') || []
  const test = r?.points.filter((p: any) => p.split === 'test') || []
  return (
    <Panel title="Polynomial curve fitting" subtitle="y(x, w) = Σ wⱼ xʲ fitted by regularised least squares on a 70/30 split. Compare train vs test error as the degree grows (over-fitting).">
      <div className="ml-form-row">
        <Field label="Input x"><Select value={x} onChange={setX} options={cols} /></Field>
        <Field label="Target y"><Select value={y} onChange={setY} options={cols} /></Field>
        <Field label={`Degree M = ${degree}`}><input type="range" min={0} max={15} value={degree} onChange={(e) => setDegree(Number(e.target.value))} className="ml-range" /></Field>
        <Field label="λ (L2 penalty)"><NumberInput value={lam} onChange={setLam} step="any" min={0} /></Field>
      </div>
      <RunButton loading={req.loading} onClick={run}>Fit polynomial</RunButton>
      <ErrorNote error={req.error} />
      {r && (
        <>
          <Kpis items={[
            ['Train RMSE', fmt(r.train.rmse)], ['Test RMSE', fmt(r.test.rmse)],
            ['Test MAE', fmt(r.test.mae)], ['Test R²', fmt(r.test.r2)],
          ]} />
          <div className="ml-grid-2">
            <div className="ml-chart-card">
              <h3>Fitted curve (M = {r.degree}, λ = {fmt(r.lambda)})</h3>
              <Legend items={[{ name: 'train', color: color(0) }, { name: 'test', color: color(1), shape: 'square' }, { name: 'fit', color: '#dce5ec' }]} />
              <ResponsiveContainer width="100%" height={280}>
                <ComposedChart margin={{ left: -6, right: 10 }}>
                  <CartesianGrid stroke={INK.grid} vertical={false} />
                  <XAxis type="number" dataKey="x" domain={['dataMin', 'dataMax']} {...axisProps} name={r.x_column} tickFormatter={(v) => fmt(v, 2)} />
                  <YAxis type="number" dataKey="y" {...axisProps} name={r.y_column} tickFormatter={(v) => fmt(v, 2)} />
                  <Tooltip contentStyle={tooltipStyle} formatter={(v: any) => fmt(v, 3)} />
                  <Scatter data={train} fill={color(0)} fillOpacity={0.55} shape="circle" isAnimationActive={false} name="train" />
                  <Scatter data={test} fill={color(1)} fillOpacity={0.7} shape="square" isAnimationActive={false} name="test" />
                  <Line data={r.curve} dataKey="y" stroke="#dce5ec" strokeWidth={2} dot={false} isAnimationActive={false} name="fit" />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
            <div className="ml-chart-card">
              <h3>RMSE vs polynomial degree</h3>
              <Legend items={[{ name: 'train', color: color(0) }, { name: 'test', color: color(1) }]} />
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={r.degree_scan} margin={{ left: -6, right: 10 }}>
                  <CartesianGrid stroke={INK.grid} vertical={false} />
                  <XAxis dataKey="degree" {...axisProps} />
                  <YAxis {...axisProps} tickFormatter={(v) => fmt(v, 2)} />
                  <Tooltip contentStyle={tooltipStyle} formatter={(v: any, n: any) => [fmt(v, 4), n]} labelFormatter={(v) => `M = ${v}`} />
                  <Line dataKey="train_rmse" name="train" stroke={color(0)} strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
                  <Line dataKey="test_rmse" name="test" stroke={color(1)} strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="ml-chart-card">
            <h3>Effect of regularisation at M = {r.degree}</h3>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={r.lambda_scan} margin={{ left: -6, right: 10 }}>
                <CartesianGrid stroke={INK.grid} vertical={false} />
                <XAxis dataKey="ln_lambda" {...axisProps} tickFormatter={(v) => `ln λ=${v}`} />
                <YAxis {...axisProps} tickFormatter={(v) => fmt(v, 2)} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v: any, n: any) => [fmt(v, 4), n]} labelFormatter={(v) => `ln λ = ${v}`} />
                <Line dataKey="train_rmse" name="train" stroke={color(0)} strokeWidth={2} dot={false} isAnimationActive={false} />
                <Line dataKey="test_rmse" name="test" stroke={color(1)} strokeWidth={2} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="ml-note">Coefficients (on standardised x): {r.coefficients_standardized_x.map((w: number, j: number) => `w${j} = ${fmt(w)}`).join(', ')}</p>
        </>
      )}
      {!r && !req.loading && <Empty>Pick an input and target, then fit. Try degree 1, 3 and 12 to see under- and over-fitting.</Empty>}
    </Panel>
  )
}

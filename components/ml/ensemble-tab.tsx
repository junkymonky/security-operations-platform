'use client'

import { useEffect, useMemo, useState } from 'react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api, type Dataset } from './api'
import SupervisedTab from './supervised-tab'
import { axisProps, color, ColumnPicker, defaultTarget, ErrorNote, Field, fmt, INK, Panel, pct, RunButton, Select, Table, Tabs, tooltipStyle, useRequest } from './ui'

export default function EnsembleTab({ dataset, notify }: { dataset: Dataset; notify: (m: string) => void }) {
  const [section, setSection] = useState('Trees & ensembles')
  return (
    <div className="ml-stack">
      <div className="ml-intro"><Tabs tabs={['Trees & ensembles', 'Bayesian model averaging']} value={section} onChange={setSection} /></div>
      {section === 'Trees & ensembles' ? <SupervisedTab dataset={dataset} unit={5} notify={notify} /> : <BMA dataset={dataset} />}
    </div>
  )
}

function BMA({ dataset }: { dataset: Dataset }) {
  const cols = dataset.columns || []
  const [task, setTask] = useState<'regression' | 'classification'>('regression')
  const targets = useMemo(() => cols.filter((c) => task === 'regression' ? c.kind === 'numeric' && c.unique > 15 : c.unique >= 2 && c.unique <= 30), [cols, task])
  const [target, setTarget] = useState('')
  const [features, setFeatures] = useState<string[]>([])
  const [maxF, setMaxF] = useState('6')
  const req = useRequest<any>()
  useEffect(() => { setTarget(defaultTarget(cols, task)); req.setData(null) }, [dataset.id, task])
  useEffect(() => { setFeatures(cols.filter((c) => c.kind === 'numeric' && c.name !== target).slice(0, 6).map((c) => c.name)) }, [dataset.id, target])

  const run = () => {
    if (!target) return req.setError('Choose a target.')
    if (!features.length) return req.setError('Select at least one feature.')
    req.run(() => api.bma(dataset.id, { task, target, features: features.filter((f) => f !== target), max_features: Number(maxF) }))
  }
  const r = req.data
  const clf = task === 'classification'
  return (
    <Panel title="Bayesian model averaging"
      subtitle="Every subset of candidate features defines a linear (or logistic) model M. Posterior weights p(M|D) ∝ exp(−BIC/2) approximate the marginal likelihood; predictions are averaged under those weights.">
      <div className="ml-form-row">
        <Field label="Task"><Select value={task} onChange={(v) => setTask(v as any)} options={[['regression', 'Regression'], ['classification', 'Classification']]} /></Field>
        <Field label="Target"><Select value={target} onChange={setTarget} options={targets.map((c) => c.name)} /></Field>
        <Field label="Max candidate features" hint="2^k models"><Select value={maxF} onChange={setMaxF} options={['3', '4', '5', '6', '7', '8']} /></Field>
      </div>
      <Field label="Features"><ColumnPicker columns={cols.filter((c) => c.name !== target)} selected={features} onChange={setFeatures} /></Field>
      <ErrorNote error={req.error} />
      <RunButton loading={req.loading} onClick={run}>Average models</RunButton>
      {r && (
        <>
          <p className="ml-note">{r.n_models} models over candidate features: {r.candidate_features.join(', ')}.</p>
          <Table columns={clf ? ['Predictor', 'Accuracy', 'F1', 'Log loss'] : ['Predictor', 'RMSE', 'MAE', 'R²']}
            rows={r.comparison.map((c: any) => clf ? [<b key="m">{c.model}</b>, pct(c.accuracy), fmt(c.f1), fmt(c.log_loss)] : [<b key="m">{c.model}</b>, fmt(c.rmse), fmt(c.mae), fmt(c.r2)])} />
          <div className="ml-grid-2">
            <div className="ml-chart-card">
              <h3>Posterior inclusion probability p(feature ∈ M | D)</h3>
              <ResponsiveContainer width="100%" height={Math.max(150, r.inclusion_probability.length * 26)}>
                <BarChart data={r.inclusion_probability} layout="vertical" margin={{ left: 30, right: 16 }}>
                  <CartesianGrid stroke={INK.grid} horizontal={false} />
                  <XAxis type="number" domain={[0, 1]} {...axisProps} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                  <YAxis type="category" dataKey="feature" {...axisProps} width={120} interval={0} />
                  <Tooltip contentStyle={tooltipStyle} formatter={(v: any) => [pct(v), 'inclusion']} cursor={{ fill: '#ffffff08' }} />
                  <Bar dataKey="probability" fill={color(0)} radius={[0, 4, 4, 0]} maxBarSize={12} isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <Table columns={['Rank', 'Model features', 'BIC', 'p(M|D)']}
              rows={r.top_models.map((m: any) => [m.rank, m.features.length ? m.features.join(' + ') : '(intercept only)', fmt(m.bic, 1), pct(m.posterior, 2)])} />
          </div>
        </>
      )}
    </Panel>
  )
}

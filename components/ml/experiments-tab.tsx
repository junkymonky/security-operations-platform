'use client'

import { useCallback, useEffect, useState } from 'react'
import { Trash2 } from 'lucide-react'
import { api, type Dataset, type Experiment } from './api'
import { Empty, ErrorNote, Field, fmt, label, Panel, Select, Table, useRequest } from './ui'

const UNIT_NAMES: Record<number, string> = { 1: 'Probability', 2: 'Regression & SVM', 3: 'Clustering & PCA', 4: 'Sequences & HMM', 5: 'Ensembles' }

function metricSummary(m: Record<string, unknown>) {
  const keys = ['accuracy', 'f1', 'rmse', 'r2', 'test_rmse', 'silhouette', 'adjusted_rand', 'log_likelihood', 'viterbi_accuracy', 'posterior', 'retained_ratio']
  return keys.filter((k) => m[k] !== undefined && m[k] !== null).slice(0, 3).map((k) => `${label(k)} ${fmt(m[k], 3)}`).join(' · ') || '—'
}

export default function ExperimentsTab({ datasets, notify }: { datasets: Dataset[]; notify: (m: string) => void }) {
  const [unit, setUnit] = useState('')
  const [ds, setDs] = useState('')
  const list = useRequest<Experiment[]>()
  const detail = useRequest<{ exp: Experiment; predictions: any[] }>()

  const load = useCallback(() => list.run(() => api.experiments({ unit: unit ? Number(unit) : undefined, dataset_id: ds || undefined })), [unit, ds, list.run])
  useEffect(() => { load() }, [load])

  const open = (e: Experiment) => detail.run(async () => {
    const [exp, predictions] = await Promise.all([api.experiment(e.id), e.has_model ? api.predictions(e.id) : Promise.resolve([])])
    return { exp, predictions }
  })

  const remove = async (e: Experiment) => {
    if (!window.confirm(`Delete experiment ${e.algorithm} (${e.id.slice(0, 8)})?`)) return
    try {
      await api.deleteExperiment(e.id)
      notify('Experiment deleted')
      if (detail.data?.exp.id === e.id) detail.setData(null)
      load()
    } catch (err: any) { list.setError(err.message) }
  }

  const rows = list.data || []
  const exp = detail.data?.exp
  return (
    <div className="ml-stack">
      <Panel title="Experiment history" subtitle="Every training run, analysis and prediction is persisted in Postgres (ml_experiments, ml_predictions)."
        actions={
          <div className="ml-form-row ml-compact">
            <Field label="Unit"><Select value={unit} onChange={setUnit} options={Object.entries(UNIT_NAMES).map(([k, v]) => [k, `Unit ${k} · ${v}`] as [string, string])} placeholder="All units" /></Field>
            <Field label="Dataset"><Select value={ds} onChange={setDs} options={datasets.map((d) => [d.id, d.name] as [string, string])} placeholder="All datasets" /></Field>
          </div>
        }>
        <ErrorNote error={list.error} />
        {list.loading && !rows.length && <Empty>Loading…</Empty>}
        {!list.loading && !rows.length && !list.error && <Empty>No experiments yet. Run an analysis in any unit tab.</Empty>}
        {rows.length > 0 && (
          <Table columns={['When', 'Unit', 'Algorithm', 'Task', 'Dataset', 'Key metrics', 'Model', 'Predictions', '']}
            rows={rows.map((e) => [
              new Date(e.created_at).toLocaleString('en-GB', { hour12: false }),
              <span key="u" className="pill pill-muted">U{e.unit}</span>,
              <b key="a">{label(e.algorithm)}</b>,
              label(e.task),
              e.dataset_name || '—',
              metricSummary(e.metrics),
              e.has_model ? 'saved' : '—',
              e.prediction_count ?? 0,
              <button key="d" className="icon-btn" title="Delete" onClick={(ev) => { ev.stopPropagation(); remove(e) }}><Trash2 size={13} /></button>,
            ])}
            onRow={(i) => open(rows[i])}
          />
        )}
      </Panel>

      <ErrorNote error={detail.error} />
      {exp && (
        <Panel title={`${label(exp.algorithm)} · Unit ${exp.unit}`} subtitle={`${exp.id} · ${exp.dataset_name || 'no dataset'} · ${exp.duration_ms ?? '—'} ms${exp.group_id ? ` · comparison group ${exp.group_id.slice(0, 8)}` : ''}`}>
          <div className="ml-grid-2">
            <Table columns={['Parameter', 'Value']} rows={[
              ...(exp.target ? [['target', exp.target]] : []),
              ...(exp.features ? [['features', exp.features.join(', ')]] : []),
              ...Object.entries(exp.params || {}).map(([k, v]) => [k, typeof v === 'object' ? JSON.stringify(v) : fmt(v)]),
            ]} />
            <Table columns={['Metric', 'Value']} rows={Object.entries(exp.metrics || {}).map(([k, v]) => [label(k), typeof v === 'object' ? JSON.stringify(v) : fmt(v, 4)])} />
          </div>
          {detail.data!.predictions.length > 0 && (
            <>
              <h3 className="ml-h3">Prediction history</h3>
              <Table columns={['When', 'Inputs', 'Output']} rows={detail.data!.predictions.map((p) => [
                new Date(p.created_at).toLocaleString('en-GB', { hour12: false }),
                <code key="i">{JSON.stringify(p.inputs).slice(0, 120)}</code>,
                <b key="o">{fmt(p.outputs.prediction ?? p.outputs.cluster)}</b>,
              ])} />
            </>
          )}
        </Panel>
      )}
    </div>
  )
}

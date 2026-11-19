'use client'

import { useEffect, useRef, useState } from 'react'
import { ChevronLeft, ChevronRight, Trash2, Upload } from 'lucide-react'
import { api, type Dataset } from './api'
import { ColumnPicker, Empty, ErrorNote, Field, fmt, Kpis, NumberInput, Panel, RunButton, Select, Table, useRequest } from './ui'

const MAX_MB = 10

export default function DataTab({ datasets, dataset, onSelect, onChanged, notify }: {
  datasets: Dataset[]; dataset: Dataset | null; onSelect: (id: string) => void
  onChanged: (selectId?: string) => void; notify: (m: string) => void
}) {
  const upload = useRequest<Dataset>()
  const samples = useRequest<{ key: string; filename: string; description: string }[]>()
  const [drag, setDrag] = useState(false)
  const input = useRef<HTMLInputElement>(null)

  useEffect(() => { samples.run(api.samples) }, [samples.run])

  const send = async (file?: File | null) => {
    if (!file) return
    if (!/\.(csv|txt)$/i.test(file.name)) return upload.setError('Only .csv files are accepted.')
    if (file.size > MAX_MB * 1024 * 1024) return upload.setError(`File exceeds the ${MAX_MB} MB limit.`)
    const ds = await upload.run(() => api.upload(file))
    if (ds) { notify(`Uploaded ${ds.name} (${ds.n_rows} rows)`); onChanged(ds.id) }
    if (input.current) input.current.value = ''
  }

  const importSample = async (key: string) => {
    const ds = await upload.run(() => api.importSample(key))
    if (ds) { notify(`Imported sample ${ds.name}`); onChanged(ds.id) }
  }

  const remove = async (d: Dataset) => {
    if (!window.confirm(`Delete dataset "${d.name}"? Experiments keep their metrics but lose the dataset link.`)) return
    try {
      await api.deleteDataset(d.id)
      notify(`Deleted ${d.name}`)
      onChanged()
    } catch (e: any) { upload.setError(e.message) }
  }

  return (
    <div className="ml-stack">
      <div className="ml-grid-2">
        <Panel title="Import dataset" subtitle="Upload a CSV (max 10 MB). Delimiter and column types are detected automatically.">
          <div
            className={`ml-drop ${drag ? 'drag' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => { e.preventDefault(); setDrag(false); send(e.dataTransfer.files?.[0]) }}
            onClick={() => input.current?.click()}
            role="button" tabIndex={0}
          >
            <Upload size={18} />
            <b>{upload.loading ? 'Uploading…' : 'Drop a CSV here or click to browse'}</b>
            <small>Header row required · .csv / .txt</small>
            <input ref={input} type="file" accept=".csv,.txt,text/csv" hidden onChange={(e) => send(e.target.files?.[0])} />
          </div>
          <ErrorNote error={upload.error} />
          <h3 className="ml-h3">Sample SOC datasets</h3>
          {samples.error && <ErrorNote error={samples.error} />}
          {samples.data?.map((s) => (
            <div className="entity-card" key={s.key}>
              <div><b>{s.filename}</b><small>{s.description}</small></div>
              <button className="filter" disabled={upload.loading} onClick={() => importSample(s.key)}>Import</button>
            </div>
          ))}
        </Panel>

        <Panel title="Datasets" subtitle="Stored on the ML service; metadata persisted in Postgres.">
          {datasets.length ? (
            <Table
              columns={['Name', 'Source', 'Rows', 'Cols', 'Size', '']}
              rows={datasets.map((d) => [
                <b key="n" className={d.id === dataset?.id ? 'ml-active' : ''}>{d.name}</b>,
                d.source,
                fmt(d.n_rows),
                d.n_columns,
                `${(d.size_bytes / 1024).toFixed(1)} KB`,
                <button key="x" className="icon-btn" title="Delete" onClick={(e) => { e.stopPropagation(); remove(d) }}><Trash2 size={13} /></button>,
              ])}
              onRow={(i) => onSelect(datasets[i].id)}
            />
          ) : <Empty>No datasets yet. Upload one or import a sample.</Empty>}
        </Panel>
      </div>

      {dataset && <DatasetDetail dataset={dataset} onChanged={onChanged} notify={notify} />}
    </div>
  )
}

function DatasetDetail({ dataset, onChanged, notify }: { dataset: Dataset; onChanged: (id?: string) => void; notify: (m: string) => void }) {
  const preview = useRequest<{ columns: string[]; rows: unknown[][]; total: number; offset: number }>()
  const [offset, setOffset] = useState(0)
  const pageSize = 25
  useEffect(() => { setOffset(0) }, [dataset.id])
  useEffect(() => { preview.run(() => api.preview(dataset.id, offset, pageSize)) }, [dataset.id, offset, preview.run])

  const cols = dataset.columns || []
  const missing = cols.reduce((a, c) => a + c.missing, 0)
  const numeric = Object.entries(dataset.summary?.numeric || {})

  return (
    <>
      <Kpis items={[
        ['Rows', fmt(dataset.n_rows)],
        ['Columns', dataset.n_columns, `${cols.filter((c) => c.kind === 'numeric').length} numeric · ${cols.filter((c) => c.kind === 'categorical').length} categorical`],
        ['Missing values', fmt(missing), missing ? 'use preprocessing to impute or drop' : 'complete'],
        ['File', `${(dataset.size_bytes / 1024).toFixed(1)} KB`, dataset.original_filename],
      ]} />

      {dataset.preprocessing && (
        <Panel title="Preprocessing lineage" subtitle={`Derived from a parent dataset · ${dataset.preprocessing.rows_before} → ${dataset.preprocessing.rows_after} rows`}>
          <ul className="ml-list">{dataset.preprocessing.steps.map((s) => <li key={s}>{s}</li>)}</ul>
        </Panel>
      )}

      <Panel title="Preview" subtitle={preview.data ? `Rows ${offset + 1}–${Math.min(offset + pageSize, preview.data.total)} of ${fmt(preview.data.total)}` : 'Loading…'}
        actions={
          <div className="ml-pager">
            <button className="filter" disabled={offset === 0 || preview.loading} onClick={() => setOffset(Math.max(0, offset - pageSize))}><ChevronLeft size={13} /></button>
            <button className="filter" disabled={!preview.data || offset + pageSize >= preview.data.total || preview.loading} onClick={() => setOffset(offset + pageSize)}><ChevronRight size={13} /></button>
          </div>
        }>
        <ErrorNote error={preview.error} />
        {preview.data && <Table columns={preview.data.columns} rows={preview.data.rows.map((r) => r.map((v) => fmt(v)))} />}
      </Panel>

      <div className="ml-grid-2">
        <Panel title="Column information" subtitle="Inferred types, cardinality and missing values.">
          <Table columns={['Column', 'Type', 'Kind', 'Unique', 'Missing']}
            rows={cols.map((c) => [<b key="c">{c.name}</b>, c.dtype, <span key="k" className={`pill pill-${c.kind === 'numeric' ? 'new' : 'muted'}`}>{c.kind}</span>, fmt(c.unique), c.missing ? <span key="m" className="red">{c.missing}</span> : 0])} />
        </Panel>
        <PreprocessPanel dataset={dataset} onChanged={onChanged} notify={notify} />
      </div>

      <Panel title="Summary statistics" subtitle="Numeric columns: count, central tendency, dispersion and quartiles.">
        <Table columns={['Column', 'Count', 'Mean', 'Median', 'Mode', 'Std', 'Variance', 'Min', 'Q1', 'Q3', 'Max']}
          rows={numeric.map(([name, s]) => [<b key="n">{name}</b>, fmt(s.count), fmt(s.mean), fmt(s.median), fmt(s.mode), fmt(s.std), fmt(s.variance), fmt(s.min), fmt(s.q25), fmt(s.q75), fmt(s.max)])} />
        {Object.keys(dataset.summary?.categorical || {}).length > 0 && (
          <div className="ml-cat-summary">
            {Object.entries(dataset.summary!.categorical).map(([col, vals]) => (
              <div key={col}><h3>{col}</h3>{vals.slice(0, 6).map((v) => <div className="health-row" key={v.value}><span>{v.value}</span><b>{fmt(v.count)}</b></div>)}</div>
            ))}
          </div>
        )}
      </Panel>
    </>
  )
}

function PreprocessPanel({ dataset, onChanged, notify }: { dataset: Dataset; onChanged: (id?: string) => void; notify: (m: string) => void }) {
  const req = useRequest<{ dataset: Dataset; steps: string[] }>()
  const cols = dataset.columns || []
  const [drop, setDrop] = useState<string[]>([])
  const [dedupe, setDedupe] = useState(true)
  const [missing, setMissing] = useState('mean')
  const [z, setZ] = useState('')
  const [scale, setScale] = useState('none')
  const [oneHot, setOneHot] = useState<string[]>([])
  useEffect(() => { setDrop([]); setOneHot([]); req.setData(null); req.setError('') }, [dataset.id])

  const submit = async () => {
    if (z !== '' && (Number(z) <= 0 || Number(z) > 10)) return req.setError('Outlier z-score must be between 0 and 10.')
    const out = await req.run(() => api.preprocess(dataset.id, {
      drop_columns: drop, drop_duplicates: dedupe, missing, scale,
      outlier_zscore: z === '' ? null : Number(z), one_hot: oneHot,
    }))
    if (out) { notify(`Created ${out.dataset.name}`); onChanged(out.dataset.id) }
  }

  return (
    <Panel title="Preprocessing" subtitle="Creates a new derived dataset; the original is kept unchanged.">
      <div className="ml-form">
        <Field label="Drop columns"><ColumnPicker columns={cols} selected={drop} onChange={setDrop} /></Field>
        <div className="ml-form-row">
          <Field label="Missing values"><Select value={missing} onChange={setMissing} options={[['none', 'Leave'], ['drop', 'Drop rows'], ['mean', 'Impute mean'], ['median', 'Impute median']]} /></Field>
          <Field label="Scaling"><Select value={scale} onChange={setScale} options={[['none', 'None'], ['standard', 'Standardize (z)'], ['minmax', 'Min–max [0,1]']]} /></Field>
          <Field label="Outlier |z| >" hint="blank = keep all"><NumberInput value={z} onChange={setZ} step="any" min={0} max={10} /></Field>
        </div>
        <Field label="One-hot encode"><ColumnPicker columns={cols} kind="categorical" selected={oneHot} onChange={setOneHot} /></Field>
        <label className="ml-check"><input type="checkbox" checked={dedupe} onChange={(e) => setDedupe(e.target.checked)} /> Remove duplicate rows</label>
        <ErrorNote error={req.error} />
        <RunButton loading={req.loading} onClick={submit}>Apply preprocessing</RunButton>
        {req.data && <ul className="ml-list">{req.data.steps.map((s) => <li key={s}>{s}</li>)}</ul>}
      </div>
    </Panel>
  )
}

'use client'

import { useEffect, useMemo, useState } from 'react'
import { api, type Dataset } from './api'
import { Comparison, ModelDetail, type TrainResponse } from './model-results'
import { ColumnPicker, defaultTarget, Empty, ErrorNote, Field, fmt, NumberInput, Panel, pct, RunButton, Select, Table, Tabs, useRequest } from './ui'

type Param = { key: string; label: string; type: 'number' | 'select' | 'bool'; default: any; min?: number; max?: number; int?: boolean; options?: string[]; optional?: boolean }
type Algo = { name: string; label: string; params: Param[]; tasks: ('regression' | 'classification')[]; note?: string }

const SVM_PARAMS: Param[] = [
  { key: 'kernel', label: 'Kernel', type: 'select', default: 'rbf', options: ['linear', 'poly', 'rbf', 'sigmoid'] },
  { key: 'C', label: 'C', type: 'number', default: 1, min: 0.0001, max: 10000 },
  { key: 'gamma', label: 'γ (scale/auto/number)', type: 'select', default: 'scale', options: ['scale', 'auto', '0.01', '0.1', '1', '10'] },
  { key: 'degree', label: 'Poly degree', type: 'number', default: 3, min: 1, max: 10, int: true },
  { key: 'coef0', label: 'coef0', type: 'number', default: 0, min: -100, max: 100 },
]

export const UNIT2: Algo[] = [
  { name: 'least_squares', label: 'Least squares (MLE)', params: [], tasks: ['regression'], note: 'Normal equations; reports σ² MLE and log-likelihood' },
  { name: 'linear', label: 'Linear regression', params: [], tasks: ['regression'] },
  { name: 'ridge', label: 'Ridge regression', params: [{ key: 'alpha', label: 'α (L2)', type: 'number', default: 1, min: 0, max: 1e6 }], tasks: ['regression'] },
  { name: 'bayesian_linear', label: 'Bayesian linear regression', tasks: ['regression'], note: 'Posterior over weights; α, β by evidence maximisation',
    params: [{ key: 'prior_alpha', label: 'Prior α', type: 'number', default: 1, min: 1e-6, max: 1e6 }, { key: 'noise_beta', label: 'Noise β', type: 'number', default: 1, min: 1e-6, max: 1e6 }, { key: 'fit_hyper', label: 'Optimise α, β', type: 'bool', default: true }] },
  { name: 'huber', label: 'Robust regression (Huber)', params: [{ key: 'epsilon', label: 'ε', type: 'number', default: 1.35, min: 1, max: 10 }], tasks: ['regression'] },
  { name: 'logistic', label: 'Logistic regression', params: [{ key: 'C', label: 'C (1/λ)', type: 'number', default: 1, min: 0.0001, max: 10000 }], tasks: ['classification'] },
  { name: 'bayesian_logistic', label: 'Bayesian logistic (Laplace)', params: [{ key: 'prior_alpha', label: 'Prior α', type: 'number', default: 1, min: 1e-6, max: 1e6 }], tasks: ['classification'] },
  { name: 'lda', label: 'Linear discriminant (LDA)', params: [], tasks: ['classification'], note: 'Generative, shared covariance' },
  { name: 'qda', label: 'Quadratic discriminant (QDA)', params: [{ key: 'reg_param', label: 'Regularisation', type: 'number', default: 0.01, min: 0, max: 1 }], tasks: ['classification'], note: 'Generative, per-class covariance' },
  { name: 'naive_bayes', label: 'Gaussian naive Bayes', params: [], tasks: ['classification'], note: 'Generative' },
  { name: 'perceptron', label: 'Perceptron', params: [], tasks: ['classification'], note: 'Discriminant function' },
  { name: 'svm', label: 'Support vector machine', params: SVM_PARAMS, tasks: ['regression', 'classification'] },
]

export const UNIT5: Algo[] = [
  { name: 'cart', label: 'Decision tree (CART)', tasks: ['regression', 'classification'], params: [
    { key: 'criterion', label: 'Criterion', type: 'select', default: '', options: ['', 'gini', 'entropy', 'squared_error', 'absolute_error'] },
    { key: 'max_depth', label: 'Max depth', type: 'number', default: 5, min: 1, max: 30, int: true },
    { key: 'min_samples_leaf', label: 'Min samples leaf', type: 'number', default: 1, min: 1, max: 1000, int: true },
    { key: 'ccp_alpha', label: 'Pruning α', type: 'number', default: 0, min: 0, max: 100 },
  ] },
  { name: 'random_forest', label: 'Random forest', tasks: ['regression', 'classification'], params: [
    { key: 'n_estimators', label: 'Trees', type: 'number', default: 100, min: 1, max: 1000, int: true },
    { key: 'max_depth', label: 'Max depth', type: 'number', default: '', min: 1, max: 50, int: true, optional: true },
    { key: 'max_features', label: 'Max features', type: 'select', default: 'sqrt', options: ['sqrt', 'log2', 'all'] },
  ] },
  { name: 'bagging', label: 'Bagging (trees)', tasks: ['regression', 'classification'], params: [{ key: 'n_estimators', label: 'Estimators', type: 'number', default: 50, min: 1, max: 1000, int: true }] },
  { name: 'adaboost', label: 'AdaBoost', tasks: ['regression', 'classification'], params: [
    { key: 'n_estimators', label: 'Weak learners', type: 'number', default: 50, min: 1, max: 1000, int: true },
    { key: 'learning_rate', label: 'Learning rate', type: 'number', default: 1, min: 0.001, max: 10 },
    { key: 'base_depth', label: 'Learner depth', type: 'number', default: 1, min: 1, max: 10, int: true },
  ] },
  { name: 'gradient_boosting', label: 'Gradient boosting', tasks: ['regression', 'classification'], params: [
    { key: 'n_estimators', label: 'Stages', type: 'number', default: 100, min: 1, max: 1000, int: true },
    { key: 'learning_rate', label: 'Learning rate', type: 'number', default: 0.1, min: 0.001, max: 1 },
    { key: 'max_depth', label: 'Depth', type: 'number', default: 3, min: 1, max: 20, int: true },
  ] },
  { name: 'voting', label: 'Voting ensemble', tasks: ['regression', 'classification'], params: [], note: 'Linear/logistic + tree + NB/ridge' },
  { name: 'gam', label: 'Generalized additive model', tasks: ['regression', 'classification'], params: [
    { key: 'n_knots', label: 'Spline knots', type: 'number', default: 6, min: 3, max: 20, int: true },
    { key: 'lam', label: 'Penalty λ', type: 'number', default: 1, min: 0.0001, max: 1000 },
  ] },
]

const defaults = (a: Algo) => Object.fromEntries(a.params.map((p) => [p.key, p.default]))

export default function SupervisedTab({ dataset, unit, notify, onResult }: {
  dataset: Dataset; unit: 2 | 5; notify: (m: string) => void; onResult?: (r: TrainResponse) => void
}) {
  const catalog = unit === 2 ? UNIT2 : UNIT5
  const cols = dataset.columns || []
  const [task, setTask] = useState<'regression' | 'classification'>('classification')
  const [target, setTarget] = useState('')
  const [features, setFeatures] = useState<string[]>([])
  const [chosen, setChosen] = useState<string[]>([])
  const [params, setParams] = useState<Record<string, Record<string, any>>>({})
  const [testSize, setTestSize] = useState('0.25')
  const [scale, setScale] = useState('standard')
  const [cv, setCv] = useState('0')
  const req = useRequest<TrainResponse>()
  const [view, setView] = useState('Comparison')

  const targets = useMemo(() => cols.filter((c) => task === 'regression' ? c.kind === 'numeric' && c.unique > 15 : c.unique >= 2 && c.unique <= 30), [cols, task])

  useEffect(() => {
    setTarget(defaultTarget(cols, task))
    req.setData(null)
  }, [dataset.id, task])
  useEffect(() => {
    // Numeric columns only by default: categorical columns are often labels in disguise (e.g. attack_type
    // determines verdict), which would leak the target. Users can add categoricals explicitly.
    setFeatures(cols.filter((c) => c.name !== target && c.kind === 'numeric').map((c) => c.name))
  }, [dataset.id, target])
  useEffect(() => {
    const avail = catalog.filter((a) => a.tasks.includes(task))
    setChosen(avail.slice(0, unit === 2 ? 3 : 4).map((a) => a.name))
    setParams(Object.fromEntries(catalog.map((a) => [a.name, defaults(a)])))
  }, [task, unit])

  const available = catalog.filter((a) => a.tasks.includes(task))

  const validate = (): string | null => {
    if (!target) return 'Choose a target column.'
    if (!features.filter((f) => f !== target).length) return 'Select at least one feature.'
    if (!chosen.length) return 'Select at least one algorithm.'
    const ts = Number(testSize)
    if (!(ts >= 0.05 && ts <= 0.5)) return 'Test size must be between 0.05 and 0.5.'
    for (const name of chosen) {
      const a = catalog.find((x) => x.name === name)!
      for (const p of a.params) {
        const v = params[name]?.[p.key]
        if (p.type !== 'number' || (p.optional && v === '')) continue
        const n = Number(v)
        if (v === '' || Number.isNaN(n) || n < (p.min ?? -Infinity) || n > (p.max ?? Infinity) || (p.int && !Number.isInteger(n)))
          return `${a.label}: ${p.label} must be ${p.int ? 'a whole number ' : ''}between ${p.min} and ${p.max}.`
      }
    }
    return null
  }

  const train = async () => {
    const err = validate()
    if (err) return req.setError(err)
    const algorithms = chosen.map((name) => {
      const a = catalog.find((x) => x.name === name)!
      const p: Record<string, any> = {}
      for (const spec of a.params) {
        const v = params[name]?.[spec.key]
        if (v === '' || v === undefined) continue
        p[spec.key] = spec.type === 'number' ? Number(v) : spec.key === 'gamma' && !Number.isNaN(Number(v)) ? Number(v) : v
      }
      return { name, params: p }
    })
    const out = await req.run(() => api.train(dataset.id, {
      task, target, features: features.filter((f) => f !== target), algorithms,
      test_size: Number(testSize), scale, cv_folds: Number(cv),
    }))
    if (out) {
      setView('Comparison')
      notify(`Trained ${out.results.length} model${out.results.length > 1 ? 's' : ''} · saved to experiment history`)
      onResult?.(out)
    }
  }

  const data = req.data
  return (
    <div className="ml-stack">
      <Panel title={unit === 2 ? 'Regression, classification and SVM' : 'Trees, boosting and ensembles'}
        subtitle="Configure the experiment. Features are imputed, scaled and one-hot encoded inside a pipeline that is saved with the model.">
        <div className="ml-form-row">
          <Field label="Task"><Select value={task} onChange={(v) => setTask(v as any)} options={[['classification', 'Classification'], ['regression', 'Regression']]} /></Field>
          <Field label="Target" hint={task === 'regression' ? 'numeric columns' : '2–30 classes'}>
            <Select value={target} onChange={setTarget} options={targets.map((c) => c.name)} placeholder={targets.length ? undefined : 'No suitable column'} />
          </Field>
          <Field label="Test split"><NumberInput value={testSize} onChange={setTestSize} min={0.05} max={0.5} step={0.05} /></Field>
          <Field label="Scaling"><Select value={scale} onChange={setScale} options={[['standard', 'Standardize'], ['minmax', 'Min–max'], ['none', 'None']]} /></Field>
          <Field label="K-fold CV"><Select value={cv} onChange={setCv} options={[['0', 'Off'], ['3', '3 folds'], ['5', '5 folds'], ['10', '10 folds']]} /></Field>
        </div>
        <Field label="Features"><ColumnPicker columns={cols.filter((c) => c.name !== target)} selected={features} onChange={setFeatures} /></Field>
        <h3 className="ml-h3">Algorithms</h3>
        <div className="ml-algos">
          {available.map((a) => {
            const on = chosen.includes(a.name)
            return (
              <div key={a.name} className={`ml-algo ${on ? 'on' : ''}`}>
                <label className="ml-check">
                  <input type="checkbox" checked={on} onChange={() => setChosen(on ? chosen.filter((x) => x !== a.name) : [...chosen, a.name])} />
                  <b>{a.label}</b>
                </label>
                {a.note && <small>{a.note}</small>}
                {on && a.params.length > 0 && (
                  <div className="ml-params">
                    {a.params.map((p) => (
                      <Field key={p.key} label={p.label}>
                        {p.type === 'select' ? (
                          <Select value={String(params[a.name]?.[p.key] ?? '')} onChange={(v) => setParams({ ...params, [a.name]: { ...params[a.name], [p.key]: v } })}
                            options={p.options!.map((o) => [o, o || 'default'] as [string, string])} />
                        ) : p.type === 'bool' ? (
                          <input type="checkbox" checked={!!params[a.name]?.[p.key]} onChange={(e) => setParams({ ...params, [a.name]: { ...params[a.name], [p.key]: e.target.checked } })} />
                        ) : (
                          <NumberInput value={params[a.name]?.[p.key] ?? ''} min={p.min} max={p.max} step={p.int ? 1 : 'any'}
                            onChange={(v) => setParams({ ...params, [a.name]: { ...params[a.name], [p.key]: v } })} />
                        )}
                      </Field>
                    ))}
                  </div>
                )}
              </div>
            )
          })}
        </div>
        <ErrorNote error={req.error} />
        <RunButton loading={req.loading} onClick={train}>Train {chosen.length > 1 ? `& compare ${chosen.length} models` : 'model'}</RunButton>
      </Panel>

      {data && (
        <Panel title="Results" subtitle={`${data.task} · target ${data.target} · ${data.results[0].n_train} train / ${data.results[0].n_test} test rows${data.notes.length ? ' · ' + data.notes.join('; ') : ''}`}
          actions={<Tabs tabs={['Comparison', ...data.results.map((r) => r.label), 'Predict']} value={view} onChange={setView} />}>
          {view === 'Comparison' && <Comparison data={data} />}
          {data.results.filter((r) => r.label === view).map((r) => <ModelDetail key={r.experiment_id} result={r} data={data} />)}
          {view === 'Predict' && <PredictForm data={data} notify={notify} />}
        </Panel>
      )}
      {!data && !req.loading && <Empty>Results, metrics and visualisations appear here after training.</Empty>}
    </div>
  )
}

function PredictForm({ data, notify }: { data: TrainResponse; notify: (m: string) => void }) {
  const [model, setModel] = useState(data.results[0].experiment_id)
  const [record, setRecord] = useState<Record<string, string>>(() =>
    Object.fromEntries(Object.entries(data.sample_record).map(([k, v]) => [k, v === null ? '' : String(v)])))
  const req = useRequest<{ predictions: any[] }>()
  const [history, setHistory] = useState<{ model: string; input: Record<string, string>; out: any }[]>([])
  const numeric = (k: string) => typeof data.sample_record[k] === 'number'

  const predict = async () => {
    for (const k of data.features) {
      if (numeric(k) && (record[k] === '' || Number.isNaN(Number(record[k])))) return req.setError(`${k} must be a number.`)
    }
    const payload = Object.fromEntries(data.features.map((k) => [k, numeric(k) ? Number(record[k]) : record[k]]))
    const out = await req.run(() => api.predict(model, [payload]))
    if (out) {
      const name = data.results.find((r) => r.experiment_id === model)!.label
      setHistory([{ model: name, input: record, out: out.predictions[0] }, ...history].slice(0, 10))
      notify('Prediction stored in history')
    }
  }
  const last = req.data?.predictions[0]
  return (
    <div className="ml-stack">
      <p className="ml-note">Pre-filled with a held-out test row. Edit values and run a saved model; each prediction is logged to the database.</p>
      <Field label="Model"><Select value={model} onChange={setModel} options={data.results.map((r) => [r.experiment_id, r.label] as [string, string])} /></Field>
      <div className="ml-predict-grid">
        {data.features.map((k) => (
          <Field key={k} label={k}>
            <input className="ml-input" value={record[k] ?? ''} inputMode={numeric(k) ? 'decimal' : 'text'} onChange={(e) => setRecord({ ...record, [k]: e.target.value })} />
          </Field>
        ))}
      </div>
      <ErrorNote error={req.error} />
      <RunButton loading={req.loading} onClick={predict}>Predict</RunButton>
      {last && (
        <div className="entity-card ml-prediction">
          <div>
            <b>Prediction: {fmt(last.prediction)}</b>
            {last.probabilities && <small>{Object.entries(last.probabilities).sort((a: any, b: any) => b[1] - a[1]).map(([k, v]) => `${k} ${pct(v as number)}`).join(' · ')}</small>}
          </div>
        </div>
      )}
      {history.length > 0 && (
        <Table columns={['Model', 'Prediction', 'Top probability']}
          rows={history.map((h) => [h.model, fmt(h.out.prediction), h.out.probabilities ? pct(Math.max(...(Object.values(h.out.probabilities) as number[]))) : '—'])} />
      )}
    </div>
  )
}

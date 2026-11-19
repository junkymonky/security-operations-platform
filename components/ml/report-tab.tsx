'use client'

import { useCallback, useEffect, useState } from 'react'
import { Download, ExternalLink, FolderOpen, Play } from 'lucide-react'
import { api } from './api'
import { Empty, ErrorNote, fmt, Kpis, Panel, pct, RunButton, Select, Table, Tabs } from './ui'

const STUDIES: [string, string][] = [['attacks', 'Network attacks — KDD Cup 1999'], ['flight-fare', 'Flight fares — Kaggle airfare']]
const rs = (v: number) => `₹${fmt(v, 2)}`
const CATS = ['normal', 'DoS', 'Probe', 'R2L', 'U2R']

export default function ReportTab({ notify }: { notify: (m: string) => void }) {
  const [study, setStudy] = useState('attacks')
  const [state, setState] = useState<any>(null)
  const [error, setError] = useState('')
  const [extended, setExtended] = useState(true)
  const [starting, setStarting] = useState(false)

  const load = useCallback(async () => {
    try { setState(await api.studyStatus(study)); setError('') } catch (e: any) { setError(e.message) }
  }, [study])
  useEffect(() => { setState(null); load() }, [load])

  const running = state?.job?.status === 'running'
  useEffect(() => {
    if (!running) return
    const t = setInterval(load, 2500)
    return () => clearInterval(t)
  }, [running, load])
  useEffect(() => { if (state?.job?.status === 'done' && state?.job?.finished_at) notify('Study finished — figures saved') }, [state?.job?.finished_at])

  const start = async () => {
    setStarting(true)
    try { await api.studyRun(study, extended); await load() } catch (e: any) { setError(e.message) } finally { setStarting(false) }
  }

  const r = state?.results
  const img = (kind: 'figures' | 'code', file: string) => api.studyImage(study, kind, file, r?.generated_at || '')

  return (
    <div className="ml-stack">
      <Panel title="Project report"
        subtitle="Runs every experiment of the five units on the chosen dataset and saves each figure and code screenshot as a PNG on this PC."
        actions={<div className="ml-form-row ml-compact">
          <Select value={study} onChange={setStudy} options={STUDIES} />
          <RunButton loading={running || starting} onClick={start} disabled={!state?.dataset?.found}>
            <Play size={14} /> {r ? 'Re-run study' : 'Run study'}</RunButton>
        </div>}>
        <ErrorNote error={error} />
        {state && !state.dataset.found && (
          <div className="ml-error"><span><b>Dataset not found.</b> {state.dataset.hint}</span></div>
        )}
        {state?.dataset.found && <p className="ml-note">Dataset: <code>{state.dataset.path}</code> ({state.dataset.size_mb} MB)</p>}
        <label className="ml-check"><input type="checkbox" checked={extended} onChange={(e) => setExtended(e.target.checked)} disabled={running} />
          Include extended analysis (cross-validation, clustering comparison, feature importance, train vs test)</label>
        {state && (
          <p className="ml-note"><FolderOpen size={12} style={{ verticalAlign: '-2px' }} /> Saved on your PC: <code>{state.assets_folder}</code> → <code>figures\</code>, <code>code\</code>, <code>results.json</code>, <code>summary.md</code></p>
        )}
        {state?.job?.status && state.job.status !== 'idle' && (
          <div className="ml-joblog">
            <b className={state.job.status === 'error' ? 'red' : ''}>Job: {state.job.status}{state.job.error ? ` — ${state.job.error}` : ''}</b>
            {state.job.log.slice(-8).map((l: any, i: number) => <small key={i}>{new Date(l.t).toLocaleTimeString('en-GB')} · {l.message.split('\n')[0]}</small>)}
          </div>
        )}
      </Panel>

      {!state && !error && <Empty>Loading…</Empty>}
      {state && !r && !running && <Empty>No results yet for this study. Press Run study.</Empty>}
      {r && study === 'attacks' && <AttackResults r={r} img={img} />}
      {r && study === 'flight-fare' && <FlightFareResults r={r} img={img} />}
    </div>
  )
}

// ------------------------------------------------------------------ shared pieces
type Img = (kind: 'figures' | 'code', file: string) => string

function FigureLinks({ href, file }: { href: string; file: string }) {
  return (
    <div className="ml-figure-links">
      <a href={href} target="_blank" rel="noreferrer"><ExternalLink size={11} /> Open</a>
      <a href={href} download={file}><Download size={11} /> {file}</a>
    </div>
  )
}

function Figures({ figs, img }: { figs: any[]; img: Img }) {
  if (!figs.length) return null
  return (
    <div className="ml-figure-grid">
      {figs.map((f) => (
        <figure key={f.file} className="ml-figure panel">
          <figcaption><b>Figure {f.number}</b> — {f.title}</figcaption>
          <a href={img('figures', f.file)} target="_blank" rel="noreferrer"><img src={img('figures', f.file)} alt={f.title} loading="lazy" /></a>
          <FigureLinks href={img('figures', f.file)} file={f.file} />
        </figure>
      ))}
    </div>
  )
}

function CodeShots({ r, img }: { r: any; img: Img }) {
  return (
    <Panel title="Appendix A – Code screenshots" subtitle="Rendered from the exact source that produced these results.">
      <div className="ml-code-grid">
        {r.code_images.map((c: any) => (
          <figure key={c.file} className="ml-figure">
            <figcaption><b>{c.chapter}</b> — {c.caption}<small>{c.functions.join(', ')} · {c.lines} lines</small></figcaption>
            <a href={img('code', c.file)} target="_blank" rel="noreferrer"><img src={img('code', c.file)} alt={c.caption} loading="lazy" /></a>
            <FigureLinks href={img('code', c.file)} file={c.file} />
          </figure>
        ))}
      </div>
    </Panel>
  )
}

function Summary({ r, kpis }: { r: any; kpis: [string, React.ReactNode, string?][] }) {
  return (
    <>
      <Kpis items={kpis} />
      <Panel title="What is happening" subtitle={`Generated ${new Date(r.generated_at).toLocaleString('en-GB')} in ${fmt(r.duration_seconds, 0)} s`}>
        {r.summary.map((s: any) => (
          <div key={s.title} className="ml-summary-section">
            <h3>{s.title}</h3>
            <ul className="ml-list">{s.points.map((p: string) => <li key={p}>{p}</li>)}</ul>
          </div>
        ))}
      </Panel>
    </>
  )
}

// ------------------------------------------------------------------ network-attack study
function ClassTable({ rows }: { rows: any[] }) {
  return <Table columns={['Model', 'Accuracy', 'Macro-F1', 'Weighted-F1', ...CATS.map((c) => `Recall ${c}`), 'ROC AUC', 'Train acc.', 'Time (s)']}
    rows={rows.map((m) => [<b key="m">{m.model}{m.note ? <small>{m.note}</small> : null}</b>, pct(m.accuracy, 2), fmt(m.macro_f1, 4), fmt(m.weighted_f1, 4),
      ...CATS.map((c) => fmt(m.recall[c], 2)), fmt(m.roc_auc, 4), pct(m.train_accuracy, 2), fmt(m.train_seconds, 2)])} />
}

function AttackResults({ r, img }: { r: any; img: Img }) {
  const [chapter, setChapter] = useState('Summary')
  const figs = (ch: string) => (r.figures || []).filter((f: any) => f.chapter === ch)
  const d = r.unit1.describe
  const best = r.comparison[0]
  return (
    <>
      <div className="ml-intro"><Tabs tabs={['Summary', 'Units 1 and 2', 'Unit 3', 'Unit 4', 'Unit 5', 'Extended analysis', 'Code screenshots']} value={chapter} onChange={setChapter} /></div>
      {chapter === 'Summary' && (
        <>
          <Summary r={r} kpis={[
            ['Connections', fmt(d.rows), `${fmt(d.duplicates_removed)} duplicates removed`],
            ['Best model', best.model, `accuracy ${pct(best.accuracy, 2)} · macro-F1 ${fmt(best.macro_f1, 3)}`],
            ['K-means (no labels)', pct(r.unit3.test_accuracy, 2), `attack vs normal, K = ${r.unit3.best_k}`],
            ['HMM from TCP flags', pct(r.unit4.hmm.viterbi_accuracy, 1), 'hidden attack phase recovered'],
          ]} />
          <Panel title="Comparison of all classifiers (test set)" subtitle={`${fmt(r.split.train)} training / ${fmt(r.split.test)} test connections`}>
            <ClassTable rows={r.comparison} />
          </Panel>
        </>
      )}
      {chapter === 'Units 1 and 2' && (
        <>
          <div className="ml-grid-2">
            <Panel title="Dataset summary">
              <Table columns={['Property', 'Value']} rows={[
                ['Raw connections', fmt(d.raw_rows)], ['Exact duplicates removed', fmt(d.duplicates_removed)],
                ['Unique connections', fmt(d.rows)], ['Features', `${d.features} (${d.categorical.join(', ')} categorical)`],
                ['Missing values', d.missing_values],
                ...CATS.map((c) => [`Category: ${c}`, `${fmt(d.category_counts[c])} (${pct(d.category_counts[c] / d.rows, 2)})`]),
              ]} />
            </Panel>
            <Panel title="Descriptive statistics of key features">
              <Table columns={['Feature', 'Mean', 'Std', 'Median', 'Max']}
                rows={Object.entries(d.key_stats).map(([k, s]: any) => [k, fmt(s.mean, 3), fmt(s.std, 3), fmt(s.median, 3), fmt(s.max, 3)])} />
            </Panel>
          </div>
          <Panel title="Bayes' rule on connection evidence" subtitle="P(attack | E) = P(E | attack) · P(attack) / P(E), checked against the direct count">
            <Table columns={['Evidence E', 'Prior P(attack)', 'P(E | attack)', 'P(E)', 'Posterior', 'Direct count', 'Lift', 'χ² p-value']}
              rows={r.unit1.bayes.map((b: any) => [b.evidence, fmt(b.prior, 4), fmt(b.likelihood, 4), fmt(b.evidence_prob, 4), <b key="p">{fmt(b.posterior, 4)}</b>, fmt(b.posterior_direct, 4), `×${fmt(b.lift, 2)}`, fmt(b.p_value)])} />
          </Panel>
          <Panel title="Unit 2 classifiers (test set)"><ClassTable rows={r.unit2} /></Panel>
          <Figures figs={figs('Units 1 and 2')} img={img} />
        </>
      )}
      {chapter === 'Unit 3' && (
        <>
          <div className="ml-grid-2">
            <Panel title="K-means results for different K (validation set)" subtitle={`${r.unit3.n_features} input features`}>
              <Table columns={['K', 'Validation accuracy', 'Inertia', 'Silhouette']}
                rows={r.unit3.sweep.map((s: any) => [s.k === r.unit3.best_k ? <b key="k">{s.k} ✓</b> : s.k, pct(s.validation_accuracy, 2), fmt(s.inertia, 1), fmt(s.silhouette, 4)])} />
            </Panel>
            <Panel title={`Test set (K = ${r.unit3.best_k})`} subtitle={`Accuracy ${pct(r.unit3.test_accuracy, 2)} · silhouette ${fmt(r.unit3.test_silhouette, 4)} · ARI vs category ${fmt(r.unit3.adjusted_rand_category, 4)}`}>
              <Table columns={['Actual \\ Predicted', 'normal', 'attack']}
                rows={r.unit3.confusion_matrix.map((row: number[], i: number) => [['normal', 'attack'][i], fmt(row[0]), fmt(row[1])])} />
            </Panel>
          </div>
          <Panel title="Cluster profiles (training set)">
            <Table columns={['Cluster', 'Connections', 'Attack share', 'Dominant category', 'Top service', 'Mean count', 'Mean SYN-error rate', 'Mapped to']}
              rows={r.unit3.profile.map((c: any) => [c.cluster, fmt(c.connections), pct(c.attack_share, 1), c.dominant_category, c.top_service, fmt(c.mean_count, 1), fmt(c.mean_serror_rate, 3), c.mapped])} />
          </Panel>
          <Figures figs={figs('Unit 3')} img={img} />
        </>
      )}
      {chapter === 'Unit 4' && (
        <>
          <div className="ml-grid-2">
            <Panel title="Markov chain over consecutive connections" subtitle={`${fmt(r.unit4.markov.transitions)} transitions · mean run ${fmt(r.unit4.markov.mean_run_length, 1)} · longest ${fmt(r.unit4.markov.longest_run)}`}>
              <Table columns={['From \\ To', ...r.unit4.markov.states]}
                rows={r.unit4.markov.matrix.map((row: number[], i: number) => [r.unit4.markov.states[i], ...row.map((x) => fmt(x, 4))])} />
              <p className="ml-note">Stationary distribution: {r.unit4.markov.states.map((s: string, i: number) => `${s} ${fmt(r.unit4.markov.stationary[i], 3)}`).join(' · ')}</p>
            </Panel>
            <Panel title="HMM: hidden attack phase → observed protocol / flag" subtitle={`Baum–Welch on ${r.unit4.hmm.train_sequences} windows of ${r.unit4.hmm.window} connections`}>
              <Table columns={['State', 'Meaning', ...r.unit4.hmm.symbols.map((s: string) => `P(${s})`)]}
                rows={r.unit4.hmm.emission.map((row: number[], i: number) => [i, r.unit4.hmm.state_meaning[String(i)] ?? '—', ...row.map((x) => fmt(x, 3))])} />
              <Table columns={['Quantity', 'Value']} rows={[
                ['Viterbi accuracy (held-out windows)', pct(r.unit4.hmm.viterbi_accuracy, 2)],
                ['Majority-class baseline', pct(r.unit4.hmm.majority_baseline, 2)],
                ['Transition matrix A', r.unit4.hmm.transition.map((row: number[]) => `[${row.map((x) => fmt(x, 3)).join(', ')}]`).join(' ')],
                ['EM iterations', r.unit4.hmm.trace.length],
              ]} />
            </Panel>
          </div>
          <Figures figs={figs('Unit 4')} img={img} />
        </>
      )}
      {chapter === 'Unit 5' && (
        <>
          <Panel title="Trees and ensembles (test set)"><ClassTable rows={r.unit5.results} /></Panel>
          <Figures figs={figs('Unit 5')} img={img} />
        </>
      )}
      {chapter === 'Extended analysis' && (r.extended ? (
        <>
          <div className="ml-grid-2">
            <Panel title="5-fold cross-validation (macro-F1)">
              <Table columns={['Model', 'Mean', 'Std', 'Sample']} rows={r.extended.cross_validation.map((m: any) => [m.model, fmt(m.mean_macro_f1, 4), fmt(m.std, 4), fmt(m.sample_size)])} />
            </Panel>
            <Panel title="Clustering comparison (K = 5)">
              <Table columns={['Method', 'Silhouette', 'Adjusted Rand', 'Time (s)']} rows={r.extended.clustering.map((m: any) => [m.method, fmt(m.silhouette, 4), fmt(m.adjusted_rand, 4), fmt(m.seconds, 2)])} />
            </Panel>
          </div>
          <Figures figs={figs('Extended analysis')} img={img} />
        </>
      ) : <Empty>The extended analysis was skipped in the last run.</Empty>)}
      {chapter === 'Code screenshots' && <CodeShots r={r} img={img} />}
    </>
  )
}

// ------------------------------------------------------------------ flight-fare study
function RegTable({ rows }: { rows: any[] }) {
  return <Table columns={['Model', 'R²', 'MAE', 'RMSE', 'Training time (s)']}
    rows={rows.map((m) => [<b key="m">{m.model}</b>, fmt(m.r2, 4), rs(m.mae), rs(m.rmse), m.train_seconds !== undefined ? fmt(m.train_seconds, 2) : '—'])} />
}

function FlightFareResults({ r, img }: { r: any; img: Img }) {
  const [chapter, setChapter] = useState('Summary')
  const figs = (ch: string) => (r.figures || []).filter((f: any) => f.chapter === ch)
  return (
    <>
      <div className="ml-intro"><Tabs tabs={['Summary', 'Units 1 and 2', 'Unit 3', 'Unit 4', 'Unit 5', 'Extended analysis', 'Code screenshots']} value={chapter} onChange={setChapter} /></div>
      {chapter === 'Summary' && (
        <>
          <Summary r={r} kpis={[
            ['Records', fmt(r.unit1.describe.rows), `${r.unit1.describe.duplicate_rows.toLocaleString()} duplicates removed`],
            ['Best model', r.comparison[0].model, `R² ${fmt(r.comparison[0].r2, 4)}`],
            ['K-means test accuracy', pct(r.unit3.pipeline.test_accuracy, 2), `K = ${r.unit3.pipeline.best_k}`],
            ['P(High → High)', fmt(r.unit4.transition.matrix[2][2], 4), 'fare-state persistence'],
          ]} />
          <Panel title="Table 4.3 – Comparison of all regression models on the common test set">
            <Table columns={['Rank', 'Algorithm', 'R²', 'MAE', 'RMSE', 'Training R²', 'Training time (s)']}
              rows={r.comparison.map((m: any, i: number) => [i + 1, <b key="m">{m.model}</b>, fmt(m.r2, 4), rs(m.mae), rs(m.rmse), fmt(m.train_r2, 4), fmt(m.train_seconds, 2)])} />
          </Panel>
        </>
      )}
      {chapter === 'Units 1 and 2' && (
        <>
          <Panel title="Table 2.1 – Attributes of the flight-fare dataset">
            <Table columns={['Attribute', 'Type', 'Distinct values', 'Most frequent / range']}
              rows={r.unit1.describe.attributes.map((a: any) => [<b key="a">{a.attribute}</b>, a.type, fmt(a.distinct), a.most_frequent])} />
          </Panel>
          <div className="ml-grid-2">
            <Panel title="Table 2.3 – Label-encoded features"><RegTable rows={r.unit2.label_encoded.results} /></Panel>
            <Panel title="Table 2.4 – Common preprocessing pipeline"><RegTable rows={r.unit2.one_hot.results} /></Panel>
          </div>
          <Figures figs={figs('Units 1 and 2')} img={img} />
        </>
      )}
      {chapter === 'Unit 3' && (
        <>
          <Panel title="Table 3.3 – K-means results for different K (validation set)">
            <Table columns={['K', 'Validation accuracy', 'Inertia', 'Silhouette']}
              rows={r.unit3.pipeline.sweep.map((s: any) => [s.k === r.unit3.pipeline.best_k ? <b key="k">{s.k} ✓</b> : s.k, pct(s.validation_accuracy, 2), fmt(s.inertia, 2), fmt(s.silhouette, 4)])} />
          </Panel>
          <Panel title={`Table 3.6 – Final K-means clusters (K = ${r.unit3.pipeline.best_k})`}>
            <Table columns={['Cluster', 'Flights', 'Mean fare', 'Mean duration (h)', 'Mean days left', 'Mapped class']}
              rows={r.unit3.pipeline.clusters.map((c: any) => [c.cluster, fmt(c.flights), rs(c.mean_fare), fmt(c.mean_duration, 2), fmt(c.mean_days_left, 2), c.mapped_class])} />
          </Panel>
          <Figures figs={figs('Unit 3')} img={img} />
        </>
      )}
      {chapter === 'Unit 4' && (
        <>
          <Panel title="Table 3.11 – Results of the HMM inference algorithms">
            <Table columns={['Algorithm', 'Output']} rows={[
              ['Forward: P(O | λ)', fmt(r.unit4.hmm.forward_probability, 7)], ['Backward: P(O | λ)', fmt(r.unit4.hmm.backward_probability, 7)],
              ['Scaled forward: log P(O | λ)', fmt(r.unit4.hmm.scaled_log_likelihood, 4)], ['Viterbi path', `[${r.unit4.hmm.viterbi_path.join(', ')}]`],
            ]} />
          </Panel>
          <Figures figs={figs('Unit 4')} img={img} />
        </>
      )}
      {chapter === 'Unit 5' && (
        <>
          <Panel title="Table 4.2 – Unit 5 models on the test set"><RegTable rows={r.unit5.results} /></Panel>
          <Figures figs={figs('Unit 5')} img={img} />
        </>
      )}
      {chapter === 'Extended analysis' && <Figures figs={figs('Extended analysis')} img={img} />}
      {chapter === 'Code screenshots' && <CodeShots r={r} img={img} />}
    </>
  )
}

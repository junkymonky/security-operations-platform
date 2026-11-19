'use client'

import { useState } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import {
  Bar, BarChart, CartesianGrid, ComposedChart, ErrorBar, Line, LineChart, ReferenceLine, ResponsiveContainer, Scatter,
  ScatterChart, Tooltip, XAxis, YAxis,
} from 'recharts'
import { axisProps, color, fmt, Heatmap, INK, Kpis, label, Legend, pct, Table, tooltipStyle } from './ui'

export type TrainResponse = {
  task: 'regression' | 'classification'
  target: string
  features: string[]
  classes: string[] | null
  sample_record: Record<string, unknown>
  notes: string[]
  results: any[]
}

// ------------------------------------------------------------------ comparison
export function Comparison({ data }: { data: TrainResponse }) {
  const clf = data.task === 'classification'
  const rows = data.results
  const best = clf
    ? rows.reduce((a, b) => (b.metrics.f1 > a.metrics.f1 ? b : a))
    : rows.reduce((a, b) => (b.metrics.rmse < a.metrics.rmse ? b : a))
  const hasCv = rows.some((r) => r.metrics.cv)
  const cols = clf
    ? ['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC', 'Train acc.', ...(hasCv ? ['CV acc. (±sd)'] : []), 'Time']
    : ['Model', 'MAE', 'MSE', 'RMSE', 'R²', 'Train RMSE', ...(hasCv ? ['CV RMSE (±sd)'] : []), 'Time']
  const chart = rows.map((r) => clf
    ? { name: r.label, Accuracy: r.metrics.accuracy, F1: r.metrics.f1 }
    : { name: r.label, RMSE: r.metrics.rmse, MAE: r.metrics.mae })
  const keys = clf ? ['Accuracy', 'F1'] : ['RMSE', 'MAE']
  return (
    <>
      <Table columns={cols} rows={rows.map((r) => {
        const m = r.metrics
        const cv = m.cv ? `${fmt(m.cv.mean)} ± ${fmt(m.cv.std)}` : null
        const name = <b key="n">{r.label}{r === best && rows.length > 1 ? <span className="pill pill-low ml-best">best</span> : null}</b>
        return clf
          ? [name, pct(m.accuracy), fmt(m.precision), fmt(m.recall), fmt(m.f1), fmt(m.roc_auc), pct(m.train_accuracy), ...(hasCv ? [cv] : []), `${r.duration_ms} ms`]
          : [name, fmt(m.mae), fmt(m.mse), fmt(m.rmse), fmt(m.r2), fmt(m.train_rmse), ...(hasCv ? [cv] : []), `${r.duration_ms} ms`]
      })} />
      {rows.length > 1 && (
        <div className="ml-chart-card">
          <h3>Model comparison {clf ? '(higher is better)' : '(lower is better)'}</h3>
          <Legend items={keys.map((k, i) => ({ name: k, color: color(i) }))} />
          <ResponsiveContainer width="100%" height={Math.max(160, rows.length * 34)}>
            <BarChart data={chart} layout="vertical" margin={{ left: 40, right: 16 }} barGap={2}>
              <CartesianGrid stroke={INK.grid} horizontal={false} />
              <XAxis type="number" {...axisProps} domain={clf ? [0, 1] : [0, 'auto']} tickFormatter={(v) => (clf ? `${Math.round(v * 100)}%` : fmt(v, 1))} />
              <YAxis type="category" dataKey="name" {...axisProps} width={150} interval={0} />
              <Tooltip contentStyle={tooltipStyle} formatter={(v: any, n: any) => [clf ? pct(v) : fmt(v), n]} cursor={{ fill: '#ffffff08' }} />
              {keys.map((k, i) => <Bar key={k} dataKey={k} fill={color(i)} radius={[0, 4, 4, 0]} maxBarSize={12} isAnimationActive={false} />)}
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </>
  )
}

// ------------------------------------------------------------------ per-model detail
export function ModelDetail({ result, data }: { result: any; data: TrainResponse }) {
  const clf = data.task === 'classification'
  const m = result.metrics
  const d = result.details || {}
  return (
    <div className="ml-stack">
      <Kpis items={clf ? [
        ['Accuracy', pct(m.accuracy)], ['Precision', fmt(m.precision), m.average], ['Recall', fmt(m.recall), m.average], ['F1 score', fmt(m.f1), m.average],
      ] : [
        ['MAE', fmt(m.mae)], ['MSE', fmt(m.mse)], ['RMSE', fmt(m.rmse)], ['R²', fmt(m.r2)],
      ]} />

      {clf && (
        <div className="ml-grid-2">
          <Heatmap title="Confusion matrix (rows = actual, columns = predicted)" rows={m.labels} cols={m.labels} matrix={m.confusion_matrix} digits={0} />
          <Table columns={['Class', 'Precision', 'Recall', 'F1', 'Support']}
            rows={m.per_class.map((c: any) => [<b key="c">{c.class}</b>, fmt(c.precision), fmt(c.recall), fmt(c.f1), c.support])} />
        </div>
      )}

      <div className="ml-grid-2">
        {result.decision_surface && <DecisionSurface surface={result.decision_surface} classes={data.classes!} title={result.label} />}
        {m.roc_curve && <RocCurve points={m.roc_curve} auc={m.roc_auc} />}
        {result.predicted_vs_actual && <PredVsActual points={result.predicted_vs_actual} />}
        {d.coefficients?.length > 0 && <Coefficients rows={d.coefficients} title="Coefficients (standardised features)" />}
        {d.coefficients_by_class?.map((c: any) => (
          <Coefficients key={c.class} rows={c.coefficients} title={`Weights for class “${c.class}”${c.log_evidence !== undefined ? ` · ln evidence ${fmt(c.log_evidence)}` : ''}`} />
        ))}
      </div>

      {d.mle && (
        <p className="ml-note">Maximum likelihood: σ²<sub>ML</sub> = {fmt(d.mle.noise_variance)} (β<sub>ML</sub> = {fmt(d.mle.noise_precision_beta)}), ln p(t|w<sub>ML</sub>, β<sub>ML</sub>) = {fmt(d.mle.log_likelihood)}. Least squares and Gaussian MLE give the same weights.</p>
      )}
      {d.posterior && (
        <p className="ml-note">Evidence approximation: α = {fmt(d.posterior.alpha)}, β = {fmt(d.posterior.beta)}, ln p(t|α,β) = {fmt(d.posterior.log_evidence)} after {d.posterior.evidence_trace.length} re-estimation steps. Error bars show ±1 predictive standard deviation.</p>
      )}
      {d.laplace && <p className="ml-note">Laplace approximation: posterior N(w<sub>MAP</sub>, S<sub>N</sub>) found in {d.laplace.newton_iterations} Newton (IRLS) iterations; ln evidence ≈ {fmt(d.laplace.log_evidence)}. ± values are posterior standard deviations.</p>}
      {d.family && <p className="ml-note">Model family: <b>{d.family}</b>.</p>}
      {d.class_priors && <p className="ml-note">Class priors p(Cₖ): {Object.entries(d.class_priors).map(([k, v]) => `${k} ${pct(v as number)}`).join(' · ')}</p>}
      {d.outliers_detected !== undefined && <p className="ml-note">Huber loss down-weighted <b>{d.outliers_detected}</b> training points as outliers (scale {fmt(d.scale)}).</p>}
      {d.ridge_path && <RidgePath path={d.ridge_path} features={result.feature_names} />}
      {d.kernel && <SvmInfo d={d} />}
      <EnsembleDetail d={d} task={data.task} />
    </div>
  )
}

function DecisionSurface({ surface, classes, title }: { surface: any; classes: string[]; title: string }) {
  const W = 360, H = 260
  const nx = surface.x.length, ny = surface.y.length
  const [x0, x1] = [surface.x[0], surface.x[nx - 1]]
  const [y0, y1] = [surface.y[0], surface.y[ny - 1]]
  const sx = (v: number) => ((v - x0) / (x1 - x0)) * W
  const sy = (v: number) => H - ((v - y0) / (y1 - y0)) * H
  const cw = W / nx + 0.5, ch = H / ny + 0.5
  return (
    <div className="ml-chart-card">
      <h3>Decision regions in 2-D PCA space · {title} <span className="muted">(2-D accuracy {pct(surface.accuracy_2d)})</span></h3>
      <Legend items={classes.map((c, i) => ({ name: c, color: color(i) }))} />
      <svg viewBox={`0 0 ${W} ${H}`} className="ml-svg" role="img" aria-label="Decision regions">
        {surface.grid.map((row: number[], j: number) => row.map((k: number, i: number) => (
          <rect key={`${i}-${j}`} x={(i / nx) * W} y={H - ((j + 1) / ny) * H} width={cw} height={ch} fill={color(k)} opacity={0.2} />
        )))}
        {surface.points.map((p: any, i: number) => (
          <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={2.6} fill={color(p.label)} stroke="#0e171e" strokeWidth={0.8}>
            <title>{classes[p.label]}</title>
          </circle>
        ))}
      </svg>
      <small>Background = class predicted by the same model refit on the first two principal components; dots = test points coloured by true class.</small>
    </div>
  )
}

function RocCurve({ points, auc }: { points: any[]; auc: number }) {
  return (
    <div className="ml-chart-card">
      <h3>ROC curve · AUC {fmt(auc)}</h3>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={points} margin={{ left: -6, right: 10 }}>
          <CartesianGrid stroke={INK.grid} vertical={false} />
          <XAxis type="number" dataKey="fpr" domain={[0, 1]} {...axisProps} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
          <YAxis domain={[0, 1]} {...axisProps} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
          <Tooltip contentStyle={tooltipStyle} formatter={(v: any) => pct(v)} labelFormatter={(v) => `FPR ${pct(v as number)}`} />
          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={INK.axis} strokeDasharray="3 3" />
          <Line dataKey="tpr" name="TPR" stroke={color(0)} strokeWidth={2} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

function PredVsActual({ points }: { points: any[] }) {
  const lo = Math.min(...points.map((p) => Math.min(p.actual, p.predicted)))
  const hi = Math.max(...points.map((p) => Math.max(p.actual, p.predicted)))
  const withStd = points[0]?.std !== undefined
  return (
    <div className="ml-chart-card">
      <h3>Predicted vs actual (test set){withStd ? ' ± predictive σ' : ''}</h3>
      <ResponsiveContainer width="100%" height={260}>
        <ScatterChart margin={{ left: -6, right: 10 }}>
          <CartesianGrid stroke={INK.grid} vertical={false} />
          <XAxis type="number" dataKey="actual" name="actual" domain={[lo, hi]} {...axisProps} tickFormatter={(v) => fmt(v, 1)} />
          <YAxis type="number" dataKey="predicted" name="predicted" domain={[lo, hi]} {...axisProps} tickFormatter={(v) => fmt(v, 1)} />
          <Tooltip contentStyle={tooltipStyle} formatter={(v: any, n: any) => [fmt(v, 3), n]} />
          <ReferenceLine segment={[{ x: lo, y: lo }, { x: hi, y: hi }]} stroke={INK.axis} strokeDasharray="3 3" />
          <Scatter data={points} fill={color(0)} fillOpacity={0.6} isAnimationActive={false}>
            {withStd && <ErrorBar dataKey="std" direction="y" stroke={color(0)} strokeOpacity={0.35} width={0} />}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  )
}

export function Coefficients({ rows, title }: { rows: { feature: string; value: number; std?: number }[]; title: string }) {
  const data = [...rows].sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 15)
    .map((r) => ({ ...r, err: r.std !== undefined ? 2 * r.std : undefined }))
  const hasStd = data.some((r) => r.err !== undefined)
  return (
    <div className="ml-chart-card">
      <h3>{title}{hasStd ? ' · ±2σ' : ''}</h3>
      <ResponsiveContainer width="100%" height={Math.max(160, data.length * 22)}>
        <BarChart data={data} layout="vertical" margin={{ left: 30, right: 16 }}>
          <CartesianGrid stroke={INK.grid} horizontal={false} />
          <XAxis type="number" {...axisProps} tickFormatter={(v) => fmt(v, 2)} />
          <YAxis type="category" dataKey="feature" {...axisProps} width={120} interval={0} />
          <ReferenceLine x={0} stroke={INK.axis} />
          <Tooltip contentStyle={tooltipStyle} formatter={(v: any) => fmt(v, 4)} cursor={{ fill: '#ffffff08' }} />
          <Bar dataKey="value" fill={color(0)} maxBarSize={12} isAnimationActive={false}>
            {hasStd && <ErrorBar dataKey="err" width={4} stroke="#dce5ec" strokeOpacity={0.6} direction="x" />}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

function RidgePath({ path, features }: { path: any[]; features: string[] }) {
  const shown = features.slice(0, 8)
  return (
    <div className="ml-chart-card">
      <h3>Ridge coefficient path — shrinkage as α grows</h3>
      <Legend items={shown.map((f, i) => ({ name: f, color: color(i) }))} />
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={path.map((p) => ({ ...p, log_alpha: Math.log10(p.alpha) }))} margin={{ left: -6, right: 10 }}>
          <CartesianGrid stroke={INK.grid} vertical={false} />
          <XAxis type="number" dataKey="log_alpha" domain={['dataMin', 'dataMax']} {...axisProps} tickFormatter={(v) => `1e${Math.round(v)}`} />
          <YAxis {...axisProps} tickFormatter={(v) => fmt(v, 2)} />
          <ReferenceLine y={0} stroke={INK.axis} />
          <Tooltip contentStyle={tooltipStyle} formatter={(v: any, n: any) => [fmt(v, 3), n]} labelFormatter={(v) => `α = ${fmt(10 ** (v as number), 3)}`} />
          {shown.map((f, i) => <Line key={f} dataKey={f} stroke={color(i)} strokeWidth={2} dot={false} isAnimationActive={false} />)}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

function SvmInfo({ d }: { d: any }) {
  const K = d.kernel_matrix
  const idx = K ? K.matrix.map((_: any, i: number) => `x${i + 1}`) : []
  return (
    <div className="ml-grid-2">
      <Table columns={['SVM property', 'Value']} rows={[
        ['Kernel', d.kernel],
        ['Support vectors', `${d.n_support_vectors} (${pct(d.support_fraction)} of training set)`],
        ...(d.n_support_per_class ? [['Per class', d.n_support_per_class.join(' / ')]] : []),
        ...(K ? [['γ used', fmt(K.gamma)], ['Gram matrix PSD', K.positive_semidefinite ? `yes (λmin = ${fmt(K.min_eigenvalue)})` : `no (λmin = ${fmt(K.min_eigenvalue)})`]] : []),
      ]} />
      {K && <Heatmap title={`Kernel trick: Gram matrix k(xᵢ, xⱼ) for ${idx.length} training points`} rows={idx} cols={idx} matrix={K.matrix} digits={1} />}
    </div>
  )
}

// ------------------------------------------------------------------ ensembles
function EnsembleDetail({ d, task }: { d: any; task: string }) {
  const scoreName = d.score_name === 'accuracy' ? 'accuracy' : 'RMSE'
  const fmtScore = (v: number) => (d.score_name === 'accuracy' ? pct(v) : fmt(v))
  return (
    <>
      {(d.feature_importance || d.permutation_importance) && (
        <div className="ml-grid-2">
          {d.feature_importance && <Importance rows={d.feature_importance} title="Impurity-based feature importance" />}
          {d.permutation_importance && <Importance rows={d.permutation_importance} title={`Permutation importance (${d.permutation_sample ? `${d.permutation_sample.toLocaleString()}-row test sample` : "test set"})`} />}
        </div>
      )}
      {d.oob_score !== undefined && <p className="ml-note">Out-of-bag score: <b>{task === 'classification' ? pct(d.oob_score) : fmt(d.oob_score)}</b> {task === 'regression' ? '(R²)' : ''}</p>}
      {d.tree && (
        <div className="ml-chart-card">
          <h3>Tree structure · depth {d.depth} · {d.n_leaves} leaves <span className="muted">(thresholds apply to the preprocessed, i.e. scaled, features)</span></h3>
          <div className="ml-tree"><TreeNode node={d.tree} depth={0} /></div>
        </div>
      )}
      {d.pruning_path && (
        <div className="ml-chart-card">
          <h3>Cost-complexity pruning: {scoreName} vs α</h3>
          <Legend items={[{ name: 'train', color: color(0) }, { name: 'test', color: color(1) }]} />
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={d.pruning_path} margin={{ left: -6, right: 10 }}>
              <CartesianGrid stroke={INK.grid} vertical={false} />
              <XAxis dataKey="ccp_alpha" {...axisProps} tickFormatter={(v) => fmt(v, 3)} />
              <YAxis {...axisProps} tickFormatter={(v) => fmtScore(v)} />
              <Tooltip contentStyle={tooltipStyle} formatter={(v: any, n: any) => [fmtScore(v), n]} labelFormatter={(v) => `α = ${fmt(v, 4)}`} />
              <Line dataKey="train" stroke={color(0)} strokeWidth={2} dot={false} isAnimationActive={false} />
              <Line dataKey="test" stroke={color(1)} strokeWidth={2} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
      {d.staged && (
        <div className="ml-chart-card">
          <h3>Boosting: test {scoreName} as weak learners are added</h3>
          <ResponsiveContainer width="100%" height={220}>
            <ComposedChart data={d.staged} margin={{ left: -6, right: 10 }}>
              <CartesianGrid stroke={INK.grid} vertical={false} />
              <XAxis dataKey="n_estimators" {...axisProps} />
              <YAxis {...axisProps} tickFormatter={(v) => fmtScore(v)} domain={['auto', 'auto']} />
              <Tooltip contentStyle={tooltipStyle} formatter={(v: any) => [fmtScore(v), `test ${scoreName}`]} labelFormatter={(v) => `${v} learners`} />
              <Line dataKey="test" stroke={color(0)} strokeWidth={2} dot={false} isAnimationActive={false} />
              {d.weak_learners && <ReferenceLine y={d.weak_learners[0]?.test_score_alone} stroke={color(1)} strokeDasharray="4 3" label={{ value: 'single weak learner', fill: INK.muted, fontSize: 10, position: 'insideBottomRight' }} />}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}
      {d.weak_learners && (
        <Table columns={['Learner', 'Weight α_m', 'Weighted error', `Alone: test ${scoreName}`]}
          rows={d.weak_learners.slice(0, 12).map((w: any) => [`#${w.index}`, fmt(w.weight), fmt(w.weighted_error), fmtScore(w.test_score_alone)])} />
      )}
      {d.member_scores && (
        <p className="ml-note">Individual trees: mean test {scoreName} {fmtScore(d.member_scores.mean)} (min {fmtScore(d.member_scores.min)}, max {fmtScore(d.member_scores.max)}). Compare with the ensemble's score above.</p>
      )}
      {d.members && (
        <Table columns={['Voting member', `Test ${scoreName}`]} rows={d.members.map((m: any) => [label(m.name), fmtScore(m.test)])} />
      )}
      {d.shape_functions && (
        <div className="ml-chart-card">
          <h3>GAM shape functions fⱼ(xⱼ) (centred, standardised inputs){task === 'classification' ? ' — log-odds of first class' : ''}</h3>
          <div className="ml-grid-3">
            {d.shape_functions.slice(0, 6).map((s: any, i: number) => (
              <div key={s.feature}>
                <small>{s.feature}</small>
                <ResponsiveContainer width="100%" height={130}>
                  <LineChart data={s.points} margin={{ left: -20, right: 6, top: 6 }}>
                    <CartesianGrid stroke={INK.grid} vertical={false} />
                    <XAxis type="number" dataKey="x" domain={['dataMin', 'dataMax']} {...axisProps} tickFormatter={(v) => fmt(v, 1)} />
                    <YAxis {...axisProps} tickFormatter={(v) => fmt(v, 1)} />
                    <Tooltip contentStyle={tooltipStyle} formatter={(v: any) => [fmt(v, 3), 'f(x)']} labelFormatter={(v) => `x = ${fmt(v, 2)}`} />
                    <Line dataKey="f" stroke={color(i)} strokeWidth={2} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ))}
          </div>
        </div>
      )}
    </>
  )
}

export function Importance({ rows, title }: { rows: { feature: string; importance: number; std?: number }[]; title: string }) {
  const data = rows.slice(0, 12)
  return (
    <div className="ml-chart-card">
      <h3>{title}</h3>
      <ResponsiveContainer width="100%" height={Math.max(150, data.length * 22)}>
        <BarChart data={data} layout="vertical" margin={{ left: 30, right: 16 }}>
          <CartesianGrid stroke={INK.grid} horizontal={false} />
          <XAxis type="number" {...axisProps} tickFormatter={(v) => fmt(v, 2)} />
          <YAxis type="category" dataKey="feature" {...axisProps} width={120} interval={0} />
          <Tooltip contentStyle={tooltipStyle} formatter={(v: any) => fmt(v, 4)} cursor={{ fill: '#ffffff08' }} />
          <Bar dataKey="importance" fill={color(2)} radius={[0, 4, 4, 0]} maxBarSize={12} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

function TreeNode({ node, depth }: { node: any; depth: number }) {
  const [open, setOpen] = useState(depth < 3)
  const leaf = !node.children
  const value = node.class !== undefined ? `class ${node.class}` : `value ${fmt(node.value)}`
  return (
    <div className="ml-tree-node">
      <button type="button" className={leaf ? 'leaf' : ''} onClick={() => !leaf && setOpen(!open)}>
        {!leaf ? (open ? <ChevronDown size={12} /> : <ChevronRight size={12} />) : <i />}
        {node.feature ? <b>{node.feature} ≤ {fmt(node.threshold)}</b> : <b>leaf</b>}
        <span>{value} · n={node.samples} · impurity {fmt(node.impurity)}{node.truncated ? ' · (deeper levels hidden)' : ''}</span>
      </button>
      {open && node.children && (
        <div className="ml-tree-children">
          <div className="ml-tree-branch"><small>true</small><TreeNode node={node.children[0]} depth={depth + 1} /></div>
          <div className="ml-tree-branch"><small>false</small><TreeNode node={node.children[1]} depth={depth + 1} /></div>
        </div>
      )}
    </div>
  )
}

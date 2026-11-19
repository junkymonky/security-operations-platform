'use client'

import { useEffect, useMemo, useState } from 'react'
import { Bar, CartesianGrid, ComposedChart, Line, LineChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts'
import { api, type Dataset } from './api'
import {
  axisProps, color, ColumnPicker, Empty, ErrorNote, Field, fmt, Heatmap, INK, Kpis, Legend, NumberInput, Panel, pct,
  RunButton, Select, SHAPES, Table, Tabs, tooltipStyle, useRequest, checkNumber,
} from './ui'

const SECTIONS = ['K-means', 'Gaussian mixture (EM)', 'Hierarchical', 'Compare algorithms', 'PCA', 'Factor analysis']

export default function ClusteringTab({ dataset }: { dataset: Dataset }) {
  const [section, setSection] = useState(SECTIONS[0])
  const cols = dataset.columns || []
  const [features, setFeatures] = useState<string[]>([])
  const [labelCol, setLabelCol] = useState('')
  const [scale, setScale] = useState('standard')
  useEffect(() => {
    setFeatures(cols.filter((c) => c.kind === 'numeric').map((c) => c.name))
    // Default reference labelling: the finest categorical column with at most 12 classes.
    setLabelCol(cols.filter((c) => c.kind === 'categorical' && c.unique <= 12).sort((a, b) => b.unique - a.unique)[0]?.name || '')
  }, [dataset.id])
  const common = { features, label_column: labelCol || null, scale }

  return (
    <div className="ml-stack">
      <div className="ml-intro"><Tabs tabs={SECTIONS} value={section} onChange={setSection} /></div>
      <Panel title="Feature space" subtitle="Shared by all Unit 3 methods. Features are standardised before computing dissimilarities.">
        <Field label="Features"><ColumnPicker columns={cols} selected={features} onChange={setFeatures} /></Field>
        <div className="ml-form-row">
          <Field label="Reference labels (evaluation / colouring only)" hint="never used for fitting">
            <Select value={labelCol} onChange={setLabelCol} options={cols.filter((c) => c.unique <= 30).map((c) => c.name)} placeholder="None" />
          </Field>
          <Field label="Scaling"><Select value={scale} onChange={setScale} options={[['standard', 'Standardize'], ['minmax', 'Min–max'], ['none', 'None']]} /></Field>
        </div>
      </Panel>
      {section === 'K-means' && <Cluster key={dataset.id + 'k'} dataset={dataset} method="kmeans" common={common} />}
      {section === 'Gaussian mixture (EM)' && <Cluster key={dataset.id + 'g'} dataset={dataset} method="gmm" common={common} />}
      {section === 'Hierarchical' && <Cluster key={dataset.id + 'h'} dataset={dataset} method="hierarchical" common={common} />}
      {section === 'Compare algorithms' && <Compare key={dataset.id} dataset={dataset} common={common} />}
      {section === 'PCA' && <Decompose key={dataset.id + 'p'} dataset={dataset} method="pca" common={common} />}
      {section === 'Factor analysis' && <Decompose key={dataset.id + 'f'} dataset={dataset} method="factor_analysis" common={common} />}
    </div>
  )
}

function GroupScatter({ points, groupKey, xKey = 'x', yKey = 'y', overlay, title, xLabel = 'PC1', yLabel = 'PC2' }: {
  points: any[]; groupKey: string; xKey?: string; yKey?: string; overlay?: any[]; title: string; xLabel?: string; yLabel?: string
}) {
  const groups = useMemo(() => Array.from(new Set(points.map((p) => String(p[groupKey])))).sort((a, b) => a.localeCompare(b, undefined, { numeric: true })), [points, groupKey])
  const many = groups.length > 3
  return (
    <div className="ml-chart-card">
      <h3>{title}</h3>
      <Legend items={groups.map((g, i) => ({ name: groupKey === 'cluster' ? `cluster ${g}` : g, color: color(i), shape: many ? SHAPES[i % SHAPES.length] : 'circle' }))} />
      <ResponsiveContainer width="100%" height={300}>
        <ScatterChart margin={{ left: -6, right: 10 }}>
          <CartesianGrid stroke={INK.grid} vertical={false} />
          <XAxis type="number" dataKey={xKey} name={xLabel} {...axisProps} tickFormatter={(v) => fmt(v, 1)} />
          <YAxis type="number" dataKey={yKey} name={yLabel} {...axisProps} tickFormatter={(v) => fmt(v, 1)} />
          <Tooltip contentStyle={tooltipStyle} formatter={(v: any, n: any) => [fmt(v, 3), n]} />
          {groups.map((g, i) => (
            <Scatter key={g} name={g} data={points.filter((p) => String(p[groupKey]) === g)} fill={color(i)} fillOpacity={0.7}
              shape={many ? (SHAPES[i % SHAPES.length] as any) : 'circle'} isAnimationActive={false} />
          ))}
          {overlay && <Scatter name="centroid" data={overlay} fill="#dce5ec" shape="cross" isAnimationActive={false} />}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  )
}

function Cluster({ dataset, method, common }: { dataset: Dataset; method: 'kmeans' | 'gmm' | 'hierarchical'; common: any }) {
  const req = useRequest<any>()
  const [k, setK] = useState('5')
  const [init, setInit] = useState('k-means++')
  const [cov, setCov] = useState('full')
  const [linkage, setLinkage] = useState('ward')
  const [metric, setMetric] = useState('euclidean')
  const [colorBy, setColorBy] = useState<'cluster' | 'label'>('cluster')

  const run = () => {
    const err = checkNumber(k, 'Number of clusters', 2, 20, true) || (!common.features.length ? 'Select at least one feature.' : null)
      || (method === 'hierarchical' && linkage === 'ward' && metric !== 'euclidean' ? 'Ward linkage requires the euclidean metric.' : null)
    if (err) return req.setError(err)
    req.run(() => api.cluster(dataset.id, { ...common, method, n_clusters: Number(k), init, covariance_type: cov, linkage, metric }))
  }
  const r = req.data
  const m = r?.metrics
  return (
    <Panel title={method === 'kmeans' ? 'K-means clustering' : method === 'gmm' ? 'Mixture of Gaussians fitted by EM' : 'Agglomerative hierarchical clustering'}
      subtitle={method === 'kmeans' ? 'Lloyd’s algorithm minimising within-cluster sum of squares J = Σ‖xₙ − μₖ‖².'
        : method === 'gmm' ? 'E-step computes responsibilities γ(zₙₖ); M-step re-estimates πₖ, μₖ, Σₖ. The log-likelihood never decreases.'
          : 'Repeatedly merges the two closest clusters under the chosen dissimilarity and linkage.'}>
      <div className="ml-form-row">
        <Field label="Clusters K"><NumberInput value={k} onChange={setK} min={2} max={20} /></Field>
        {method === 'kmeans' && <Field label="Initialisation"><Select value={init} onChange={setInit} options={['k-means++', 'random']} /></Field>}
        {method === 'gmm' && <Field label="Covariance"><Select value={cov} onChange={setCov} options={['full', 'tied', 'diag', 'spherical']} /></Field>}
        {method === 'hierarchical' && <>
          <Field label="Linkage"><Select value={linkage} onChange={setLinkage} options={['ward', 'complete', 'average', 'single']} /></Field>
          <Field label="Dissimilarity"><Select value={metric} onChange={setMetric} options={[['euclidean', 'Euclidean'], ['cityblock', 'Manhattan'], ['cosine', 'Cosine'], ['chebyshev', 'Chebyshev'], ['correlation', 'Correlation']]} /></Field>
        </>}
      </div>
      <ErrorNote error={req.error} />
      <RunButton loading={req.loading} onClick={run}>Run clustering</RunButton>
      {r && (
        <>
          <Kpis items={[
            ['Silhouette', fmt(m.silhouette), 'higher is better (−1…1)'],
            ['Davies–Bouldin', fmt(m.davies_bouldin), 'lower is better'],
            ['Inertia (WCSS)', fmt(m.inertia, 1)],
            m.adjusted_rand !== undefined ? ['Adjusted Rand vs labels', fmt(m.adjusted_rand), `NMI ${fmt(m.normalized_mutual_info)}`] : ['Calinski–Harabasz', fmt(m.calinski_harabasz, 1)],
          ]} />
          <div className="ml-grid-2">
            <div>
              {r.points[0]?.label !== undefined && (
                <div className="ml-inline-toggle"><Tabs tabs={['cluster', 'label']} value={colorBy} onChange={(v) => setColorBy(v as any)} /></div>
              )}
              <GroupScatter points={r.points} groupKey={colorBy} overlay={r.centroids_2d}
                title={`${colorBy === 'cluster' ? 'Cluster assignments' : 'Reference labels'} (PCA projection, ${pct((r.projection_variance[0] || 0) + (r.projection_variance[1] || 0))} variance)`} />
            </div>
            {method === 'kmeans' && <ScanCharts scan={r.k_scan} />}
            {method === 'gmm' && <EmCharts r={r} />}
            {method === 'hierarchical' && <Dendrogram d={r.dendrogram} coph={r.cophenetic_correlation} n={r.n_rows_used} />}
          </div>
          {method === 'gmm' && (
            <>
              <p className="ml-note">Mixing weights π = [{r.weights.map((w: number) => fmt(w, 3)).join(', ')}] · mean max responsibility {pct(r.mean_max_responsibility)} · {r.uncertain_points} points with max γ &lt; 80% · BIC {fmt(r.bic, 1)}</p>
              <Table columns={['Point', 'Assigned', ...r.weights.map((_: any, i: number) => `γ(z=${i})`)]}
                rows={r.points.slice(0, 12).map((p: any, i: number) => [i + 1, `cluster ${p.cluster}`, ...p.probabilities.map((v: number) => pct(v))])} />
            </>
          )}
          <h3 className="ml-h3">Cluster profiles (mean of original numeric features)</h3>
          <Table columns={['Cluster', 'Size', ...Object.keys(r.profiles[0]?.means || {})]}
            rows={r.profiles.map((p: any) => [`cluster ${p.cluster}`, fmt(p.size), ...Object.values(p.means).map((v: any) => fmt(v, 2))])} />
          {r.centroids_scaled && (
            <Heatmap title="Centroids μₖ in standardised feature space" rows={r.centroids_scaled.map((c: any) => `μ${c.cluster}`)}
              cols={Object.keys(r.centroids_scaled[0]).filter((c) => c !== 'cluster')}
              matrix={r.centroids_scaled.map((c: any) => Object.entries(c).filter(([key]) => key !== 'cluster').map(([, v]) => v as number))} mode="diverging" />
          )}
        </>
      )}
      {!r && !req.loading && <Empty>Choose K and run. The sample network dataset has five planted behaviour classes.</Empty>}
    </Panel>
  )
}

function ScanCharts({ scan }: { scan: any[] }) {
  return (
    <div className="ml-stack-tight">
      <div className="ml-chart-card">
        <h3>Elbow method: inertia vs K</h3>
        <ResponsiveContainer width="100%" height={140}>
          <LineChart data={scan} margin={{ left: 0, right: 10 }}>
            <CartesianGrid stroke={INK.grid} vertical={false} />
            <XAxis dataKey="k" {...axisProps} />
            <YAxis {...axisProps} tickFormatter={(v) => fmt(v, 0)} width={50} />
            <Tooltip contentStyle={tooltipStyle} formatter={(v: any) => [fmt(v, 1), 'inertia']} labelFormatter={(v) => `K = ${v}`} />
            <Line dataKey="inertia" stroke={color(0)} strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="ml-chart-card">
        <h3>Silhouette vs K</h3>
        <ResponsiveContainer width="100%" height={140}>
          <LineChart data={scan.filter((s) => s.k >= 2)} margin={{ left: 0, right: 10 }}>
            <CartesianGrid stroke={INK.grid} vertical={false} />
            <XAxis dataKey="k" {...axisProps} />
            <YAxis {...axisProps} tickFormatter={(v) => fmt(v, 2)} width={50} />
            <Tooltip contentStyle={tooltipStyle} formatter={(v: any) => [fmt(v, 3), 'silhouette']} labelFormatter={(v) => `K = ${v}`} />
            <Line dataKey="silhouette" stroke={color(1)} strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function EmCharts({ r }: { r: any }) {
  return (
    <div className="ml-stack-tight">
      <div className="ml-chart-card">
        <h3>EM: log-likelihood bound per sample by iteration ({r.converged_iterations} iterations)</h3>
        <ResponsiveContainer width="100%" height={140}>
          <LineChart data={r.em_trace} margin={{ left: 0, right: 10 }}>
            <CartesianGrid stroke={INK.grid} vertical={false} />
            <XAxis dataKey="iteration" {...axisProps} />
            <YAxis {...axisProps} domain={['auto', 'auto']} tickFormatter={(v) => fmt(v, 2)} width={50} />
            <Tooltip contentStyle={tooltipStyle} formatter={(v: any) => [fmt(v, 4), 'log-likelihood']} labelFormatter={(v) => `iteration ${v}`} />
            <Line dataKey="log_likelihood" stroke={color(0)} strokeWidth={2} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="ml-chart-card">
        <h3>Choosing K: BIC (lower is better)</h3>
        <ResponsiveContainer width="100%" height={140}>
          <LineChart data={r.model_selection} margin={{ left: 0, right: 10 }}>
            <CartesianGrid stroke={INK.grid} vertical={false} />
            <XAxis dataKey="k" {...axisProps} />
            <YAxis {...axisProps} domain={['auto', 'auto']} tickFormatter={(v) => fmt(v, 0)} width={50} />
            <Tooltip contentStyle={tooltipStyle} formatter={(v: any, n: any) => [fmt(v, 1), n]} labelFormatter={(v) => `K = ${v}`} />
            <Line dataKey="bic" name="BIC" stroke={color(1)} strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function Dendrogram({ d, coph, n }: { d: any; coph: number; n: number }) {
  const W = 520, H = 280, pad = 30
  const xs = d.icoord.flat() as number[]
  const ys = d.dcoord.flat() as number[]
  const xmax = Math.max(...xs) + 5, ymax = Math.max(...ys) * 1.05 || 1
  const sx = (v: number) => pad + (v / xmax) * (W - pad - 6)
  const sy = (v: number) => H - 24 - (v / ymax) * (H - 34)
  return (
    <div className="ml-chart-card">
      <h3>Dendrogram (last 30 merges) · cophenetic correlation {fmt(coph)} · n = {n}</h3>
      <svg viewBox={`0 0 ${W} ${H}`} className="ml-svg" role="img" aria-label="Dendrogram">
        <line x1={pad} x2={pad} y1={6} y2={H - 24} stroke={INK.grid} />
        {[0, 0.25, 0.5, 0.75, 1].map((f) => (
          <g key={f}>
            <text x={pad - 4} y={sy(f * ymax) + 3} fill={INK.axis} fontSize={9} textAnchor="end">{fmt(f * ymax, 1)}</text>
            <line x1={pad} x2={W} y1={sy(f * ymax)} y2={sy(f * ymax)} stroke={INK.grid} strokeDasharray="2 4" />
          </g>
        ))}
        {d.icoord.map((ic: number[], i: number) => (
          <polyline key={i} fill="none" stroke={color(0)} strokeWidth={1.5}
            points={ic.map((x, j) => `${sx(x)},${sy(d.dcoord[i][j])}`).join(' ')} />
        ))}
        <line x1={pad} x2={W} y1={sy(d.cut_height)} y2={sy(d.cut_height)} stroke={color(1)} strokeDasharray="5 3" />
        <text x={W - 4} y={sy(d.cut_height) - 4} fill={INK.muted} fontSize={9} textAnchor="end">cut for K clusters</text>
        {d.leaves.map((l: string, i: number) => (
          <text key={i} x={sx(5 + i * 10)} y={H - 10} fill={INK.axis} fontSize={8} textAnchor="middle">{l.replace(/[()]/g, '')}</text>
        ))}
      </svg>
      <small>Leaf labels show the number of points under truncated branches.</small>
    </div>
  )
}

function Compare({ dataset, common }: { dataset: Dataset; common: any }) {
  const req = useRequest<any>()
  const [k, setK] = useState('5')
  const run = () => {
    const err = checkNumber(k, 'K', 2, 20, true)
    if (err) return req.setError(err)
    req.run(() => api.cluster(dataset.id, { ...common, method: 'compare', n_clusters: Number(k) }))
  }
  const rows = req.data?.comparison
  const hasTruth = rows?.[0]?.adjusted_rand !== undefined
  return (
    <Panel title="Cluster comparison" subtitle="K-means, Gaussian mixture and Ward hierarchical clustering at the same K, scored with internal (and, if labels are set, external) indices.">
      <div className="ml-form-row"><Field label="Clusters K"><NumberInput value={k} onChange={setK} min={2} max={20} /></Field></div>
      <ErrorNote error={req.error} />
      <RunButton loading={req.loading} onClick={run}>Compare</RunButton>
      {rows && (
        <Table columns={['Method', 'Silhouette ↑', 'Davies–Bouldin ↓', 'Calinski–Harabasz ↑', 'Inertia ↓', ...(hasTruth ? ['ARI ↑', 'NMI ↑', 'Homogeneity'] : []), 'Time']}
          rows={rows.map((x: any) => [<b key="m">{x.method}</b>, fmt(x.silhouette), fmt(x.davies_bouldin), fmt(x.calinski_harabasz, 1), fmt(x.inertia, 1),
            ...(hasTruth ? [fmt(x.adjusted_rand), fmt(x.normalized_mutual_info), fmt(x.homogeneity)] : []), `${x.duration_ms} ms`])} />
      )}
    </Panel>
  )
}

function Decompose({ dataset, method, common }: { dataset: Dataset; method: 'pca' | 'factor_analysis'; common: any }) {
  const req = useRequest<any>()
  const [n, setN] = useState('3')
  const [rotation, setRotation] = useState('varimax')
  const run = () => {
    const err = checkNumber(n, 'Components', 1, Math.max(1, common.features.length * 4), true) || (!common.features.length ? 'Select features.' : null)
    if (err) return req.setError(err)
    req.run(() => api.decompose(dataset.id, { method, features: common.features, n_components: Number(n), scale: common.scale, color_column: common.label_column, rotation }))
  }
  const r = req.data
  const pca = method === 'pca'
  const scree = r && pca ? r.explained_variance_ratio.map((v: number, i: number) => ({ pc: `PC${i + 1}`, ratio: v, cumulative: r.cumulative_ratio[i] })) : []
  return (
    <Panel title={pca ? 'Principal component analysis' : 'Factor analysis'}
      subtitle={pca ? 'Orthogonal projection onto directions of maximal variance (eigenvectors of the covariance matrix).' : 'Latent-variable model x = Wz + μ + ε with diagonal noise Ψ, fitted by maximum likelihood.'}>
      <div className="ml-form-row">
        <Field label={pca ? 'Components' : 'Factors'}><NumberInput value={n} onChange={setN} min={1} max={50} /></Field>
        {!pca && <Field label="Rotation"><Select value={rotation} onChange={setRotation} options={['varimax', 'quartimax', 'none']} /></Field>}
      </div>
      <ErrorNote error={req.error} />
      <RunButton loading={req.loading} onClick={run}>Fit {pca ? 'PCA' : 'factor model'}</RunButton>
      {r && pca && (
        <>
          <Kpis items={[
            ['Variance retained', pct(r.retained_ratio), `${r.n_components} of ${r.features.length} dims`],
            ['Dims for 90% / 95%', `${r.dimension_choice.components_for_90pct} / ${r.dimension_choice.components_for_95pct}`],
            ['Kaiser rule', r.dimension_choice.kaiser_rule, 'eigenvalues above mean'],
            ['Scree elbow / PPCA CV', `${r.dimension_choice.elbow} / ${r.dimension_choice.ppca_best_k ?? '—'}`],
          ]} />
          <div className="ml-grid-2">
            <div className="ml-chart-card">
              <h3>Scree plot: explained variance ratio (bars) and cumulative (line)</h3>
              <ResponsiveContainer width="100%" height={240}>
                <ComposedChart data={scree} margin={{ left: -6, right: 10 }}>
                  <CartesianGrid stroke={INK.grid} vertical={false} />
                  <XAxis dataKey="pc" {...axisProps} />
                  <YAxis domain={[0, 1]} {...axisProps} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                  <Tooltip contentStyle={tooltipStyle} formatter={(v: any, nm: any) => [pct(v), nm]} cursor={{ fill: '#ffffff08' }} />
                  <Bar dataKey="ratio" name="explained" fill={color(0)} radius={[4, 4, 0, 0]} maxBarSize={28} isAnimationActive={false} />
                  <Line dataKey="cumulative" stroke={color(1)} strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
            <div className="ml-chart-card">
              <h3>Reconstruction error vs number of components</h3>
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={r.reconstruction} margin={{ left: -6, right: 10 }}>
                  <CartesianGrid stroke={INK.grid} vertical={false} />
                  <XAxis dataKey="k" {...axisProps} />
                  <YAxis {...axisProps} tickFormatter={(v) => fmt(v, 2)} />
                  <Tooltip contentStyle={tooltipStyle} formatter={(v: any) => [fmt(v, 4), 'MSE']} labelFormatter={(v) => `${v} components`} />
                  <Line dataKey="mse" stroke={color(0)} strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="ml-grid-2">
            <GroupScatter points={r.points.map((p: any) => ({ ...p, label: p.label ?? 'all points' }))} groupKey="label" xKey="c1" yKey="c2" title="2-D projection (PC1 vs PC2)" />
            {r.points[0]?.c3 !== undefined ? <Scatter3D points={r.points} /> : <Empty>Choose ≥ 3 components for the 3-D view.</Empty>}
          </div>
          <Heatmap title="Loadings (component weights per feature)" rows={r.loadings.map((l: any) => l.component)}
            cols={r.features} matrix={r.loadings.map((l: any) => r.features.map((f: string) => l[f]))} mode="diverging" />
        </>
      )}
      {r && !pca && (
        <>
          <Kpis items={[
            ['Log-likelihood / sample', fmt(r.log_likelihood_per_sample)],
            ['Best K by 3-fold CV', r.best_k ?? '—'],
            ['Features', r.features.length],
            ['Factors', r.n_components],
          ]} />
          <Heatmap title="Factor loadings W" rows={r.loadings.map((l: any) => l.factor)} cols={r.features}
            matrix={r.loadings.map((l: any) => r.features.map((f: string) => l[f]))} mode="diverging" />
          <div className="ml-grid-2">
            <Table columns={['Feature', 'Communality h²', 'Noise ψ']}
              rows={r.features.map((f: string, i: number) => [f, fmt(r.communalities[i].h2), fmt(r.noise_variance[i].psi)])} />
            <div className="ml-chart-card">
              <h3>Choosing the number of latent dimensions (held-out log-likelihood)</h3>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={r.model_selection} margin={{ left: -6, right: 10 }}>
                  <CartesianGrid stroke={INK.grid} vertical={false} />
                  <XAxis dataKey="k" {...axisProps} />
                  <YAxis {...axisProps} domain={['auto', 'auto']} tickFormatter={(v) => fmt(v, 2)} />
                  <Tooltip contentStyle={tooltipStyle} formatter={(v: any) => [fmt(v, 4), 'CV log-lik']} labelFormatter={(v) => `${v} factors`} />
                  <Line dataKey="cv_log_likelihood" stroke={color(0)} strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          <GroupScatter points={r.points.map((p: any) => ({ ...p, label: p.label ?? 'all points' }))} groupKey="label" xKey="c1" yKey={r.points[0]?.c2 !== undefined ? 'c2' : 'c1'} title="Factor scores (F1 vs F2)" xLabel="F1" yLabel="F2" />
        </>
      )}
    </Panel>
  )
}

function Scatter3D({ points }: { points: any[] }) {
  const [yaw, setYaw] = useState(35)
  const [pitch, setPitch] = useState(20)
  const W = 380, H = 300
  const groups = Array.from(new Set(points.map((p) => String(p.label ?? 'all')))).sort()
  const ext = (k: string) => Math.max(...points.map((p) => Math.abs(p[k]))) || 1
  const [ex, ey, ez] = [ext('c1'), ext('c2'), ext('c3')]
  const a = (yaw * Math.PI) / 180, b = (pitch * Math.PI) / 180
  const project = (x: number, y: number, z: number) => {
    const x1 = x * Math.cos(a) - z * Math.sin(a)
    const z1 = x * Math.sin(a) + z * Math.cos(a)
    const y1 = y * Math.cos(b) - z1 * Math.sin(b)
    const depth = y * Math.sin(b) + z1 * Math.cos(b)
    return { px: W / 2 + x1 * 110, py: H / 2 - y1 * 110, depth }
  }
  const pts = points.map((p) => ({ ...project(p.c1 / ex, p.c3 / ez, p.c2 / ey), g: groups.indexOf(String(p.label ?? 'all')) }))
    .sort((p, q) => p.depth - q.depth)
  const axes = [['PC1', [1, 0, 0]], ['PC3', [0, 1, 0]], ['PC2', [0, 0, 1]]] as const
  const o = project(0, 0, 0)
  return (
    <div className="ml-chart-card">
      <h3>3-D projection (PC1, PC2, PC3)</h3>
      <Legend items={groups.map((g, i) => ({ name: g, color: color(i) }))} />
      <svg viewBox={`0 0 ${W} ${H}`} className="ml-svg" role="img" aria-label="3-D PCA scatter">
        {axes.map(([name, [x, y, z]]) => {
          const e = project(x * 1.2, y * 1.2, z * 1.2)
          return <g key={name}><line x1={o.px} y1={o.py} x2={e.px} y2={e.py} stroke={INK.axis} /><text x={e.px} y={e.py} fill={INK.muted} fontSize={10}>{name}</text></g>
        })}
        {pts.map((p, i) => <circle key={i} cx={p.px} cy={p.py} r={2.4} fill={color(p.g)} fillOpacity={0.75} />)}
      </svg>
      <div className="ml-form-row">
        <Field label={`Rotate ${yaw}°`}><input className="ml-range" type="range" min={0} max={360} value={yaw} onChange={(e) => setYaw(Number(e.target.value))} /></Field>
        <Field label={`Tilt ${pitch}°`}><input className="ml-range" type="range" min={-90} max={90} value={pitch} onChange={(e) => setPitch(Number(e.target.value))} /></Field>
      </div>
    </div>
  )
}

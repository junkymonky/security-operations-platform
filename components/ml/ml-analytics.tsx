'use client'

import { useCallback, useEffect, useState } from 'react'
import { Database, RefreshCw } from 'lucide-react'
import { api, type Dataset } from './api'
import { ErrorNote, fmt, Tabs } from './ui'
import DataTab from './data-tab'
import ProbabilityTab from './probability-tab'
import SupervisedTab from './supervised-tab'
import ClusteringTab from './clustering-tab'
import SequenceTab from './sequence-tab'
import EnsembleTab from './ensemble-tab'
import ExperimentsTab from './experiments-tab'
import ReportTab from './report-tab'

const TABS = [
  'Project Report',
  'Datasets',
  'Unit 1 · Probability',
  'Unit 2 · Regression & SVM',
  'Unit 3 · Clustering & PCA',
  'Unit 4 · Sequences & HMM',
  'Unit 5 · Trees & Ensembles',
  'Experiments',
]

export default function MLAnalytics({ notify }: { notify: (msg: string) => void }) {
  const [tab, setTab] = useState(TABS[0])
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [selectedId, setSelectedId] = useState('')
  const [dataset, setDataset] = useState<Dataset | null>(null)
  const [error, setError] = useState('')
  const [service, setService] = useState<'checking' | 'ok' | 'down'>('checking')

  const refresh = useCallback(async (select?: string) => {
    try {
      const list = await api.datasets()
      setDatasets(list)
      setService('ok')
      setError('')
      setSelectedId((cur) => select ?? (list.some((d) => d.id === cur) ? cur : list[0]?.id ?? ''))
    } catch (e: any) {
      setService('down')
      setError(e.message)
    }
  }, [])

  useEffect(() => { refresh() }, [refresh])

  useEffect(() => {
    if (!selectedId) { setDataset(null); return }
    let live = true
    api.dataset(selectedId).then((d) => live && setDataset(d)).catch((e) => live && setError(e.message))
    return () => { live = false }
  }, [selectedId])

  const needsData = !['Datasets', 'Experiments', 'Project Report'].includes(tab) && !tab.startsWith('Unit 4')

  return (
    <div className="ml-workspace">
      <div className="panel ml-toolbar">
        <div className="ml-toolbar-row">
          <Database size={15} className="muted" />
          <span className="muted">Active dataset</span>
          <select className="ml-input ml-dataset-select" value={selectedId} onChange={(e) => setSelectedId(e.target.value)}
            disabled={!datasets.length}>
            {!datasets.length && <option value="">No datasets yet</option>}
            {datasets.map((d) => <option key={d.id} value={d.id}>{d.name} · {d.n_rows.toLocaleString()} rows</option>)}
          </select>
          {dataset && (
            <span className="ml-dim">
              {fmt(dataset.n_rows)} rows × {dataset.n_columns} columns · {dataset.source}
            </span>
          )}
          <span className={`ml-service ${service}`}>
            <i /> ML service {service === 'ok' ? 'connected' : service === 'down' ? 'unreachable' : 'checking…'}
          </span>
          <button className="filter" onClick={() => refresh()}><RefreshCw size={13} /> Reload</button>
        </div>
        <Tabs tabs={TABS} value={tab} onChange={setTab} />
      </div>

      <ErrorNote error={service === 'down' ? `${error} Start it with: pnpm ml` : error} />

      {needsData && !dataset ? (
        <div className="panel placeholder ml-empty-state">
          <h2>No dataset selected</h2>
          <p>Upload a CSV or import a sample dataset on the Datasets tab to run this unit.</p>
          <button className="primary" onClick={() => setTab('Datasets')}>Go to datasets</button>
        </div>
      ) : (
        <>
          {tab === 'Datasets' && (
            <DataTab datasets={datasets} dataset={dataset} onSelect={setSelectedId}
              onChanged={(id) => refresh(id)} notify={notify} />
          )}
          {tab === 'Unit 1 · Probability' && dataset && <ProbabilityTab dataset={dataset} />}
          {tab === 'Unit 2 · Regression & SVM' && dataset && <SupervisedTab dataset={dataset} unit={2} notify={notify} />}
          {tab === 'Unit 3 · Clustering & PCA' && dataset && <ClusteringTab dataset={dataset} />}
          {tab === 'Unit 4 · Sequences & HMM' && <SequenceTab dataset={dataset} />}
          {tab === 'Unit 5 · Trees & Ensembles' && dataset && <EnsembleTab dataset={dataset} notify={notify} />}
          {tab === 'Experiments' && <ExperimentsTab datasets={datasets} notify={notify} />}
          {tab === 'Project Report' && <ReportTab notify={notify} />}
        </>
      )}
    </div>
  )
}

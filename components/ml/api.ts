// Typed client for the Python ML service (proxied at /ml-api by next.config.mjs).

const BASE = process.env.NEXT_PUBLIC_ML_API_URL || '/ml-api'

export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

function describe(detail: unknown): string {
  if (typeof detail === 'string') return detail
  // FastAPI/Pydantic validation errors: [{loc: [...], msg}]
  if (Array.isArray(detail)) {
    return detail
      .map((d: any) => `${(d.loc || []).filter((x: any) => x !== 'body').join('.')}: ${d.msg}`)
      .join('; ')
  }
  return 'Request failed'
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${BASE}${path}`, init)
  } catch {
    throw new ApiError('Cannot reach the ML service. Is it running on port 8000?', 0)
  }
  if (res.status === 204) return undefined as T
  const text = await res.text()
  let body: any = null
  try {
    body = text ? JSON.parse(text) : null
  } catch {
    // Non-JSON (e.g. proxy error page)
  }
  if (!res.ok) {
    if (res.status >= 500 && !body) {
      throw new ApiError(
        'The ML service did not respond. Either it is not running (start it with `pnpm ml`) or it stopped ' +
        'while processing this request — check the `pnpm ml` terminal for the error.',
        res.status,
      )
    }
    throw new ApiError(describe(body?.detail) || `Request failed (${res.status})`, res.status)
  }
  return body as T
}

const post = <T>(path: string, body: unknown) =>
  request<T>(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })

export interface ColumnInfo {
  name: string
  dtype: string
  kind: 'numeric' | 'categorical'
  missing: number
  unique: number
}

export interface Dataset {
  id: string
  name: string
  original_filename: string
  source: string
  parent_id: string | null
  n_rows: number
  n_columns: number
  size_bytes: number
  created_at: string
  columns?: ColumnInfo[]
  preprocessing?: { steps: string[]; rows_before: number; rows_after: number } | null
  summary?: { numeric: Record<string, Record<string, number | null>>; categorical: Record<string, { value: string; count: number }[]> }
}

export interface Experiment {
  id: string
  group_id: string | null
  dataset_id: string | null
  dataset_name: string | null
  unit: number
  task: string
  algorithm: string
  params: Record<string, unknown>
  features: string[] | null
  target: string | null
  metrics: Record<string, unknown>
  duration_ms: number | null
  has_model: boolean
  prediction_count?: number
  created_at: string
  results?: any
}

export const api = {
  health: () => request<{ status: string; database: string }>('/health'),
  datasets: () => request<Dataset[]>('/datasets'),
  samples: () => request<{ key: string; filename: string; description: string }[]>('/datasets/samples'),
  importSample: (key: string) => request<Dataset>(`/datasets/samples/${key}`, { method: 'POST' }),
  upload: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return request<Dataset>('/datasets', { method: 'POST', body: form })
  },
  dataset: (id: string) => request<Dataset>(`/datasets/${id}`),
  preview: (id: string, offset = 0, limit = 25) =>
    request<{ columns: string[]; rows: unknown[][]; total: number; offset: number }>(
      `/datasets/${id}/preview?offset=${offset}&limit=${limit}`,
    ),
  preprocess: (id: string, body: unknown) => post<{ dataset: Dataset; steps: string[] }>(`/datasets/${id}/preprocess`, body),
  deleteDataset: (id: string) => request<void>(`/datasets/${id}`, { method: 'DELETE' }),

  describe: (id: string, columns?: string[]) => post<any>(`/datasets/${id}/stats/describe`, { columns }),
  covariance: (id: string, body: unknown) => post<any>(`/datasets/${id}/stats/covariance`, body),
  distribution: (id: string, body: unknown) => post<any>(`/datasets/${id}/stats/distribution`, body),
  bayes: (id: string, body: unknown) => post<any>(`/datasets/${id}/stats/bayes`, body),
  polynomial: (id: string, body: unknown) => post<any>(`/datasets/${id}/stats/polynomial`, body),

  algorithms: () => request<{ regression: Record<string, string>; classification: Record<string, string> }>('/algorithms'),
  train: (id: string, body: unknown) => post<any>(`/datasets/${id}/train`, body),
  bma: (id: string, body: unknown) => post<any>(`/datasets/${id}/bma`, body),

  cluster: (id: string, body: unknown) => post<any>(`/datasets/${id}/cluster`, body),
  decompose: (id: string, body: unknown) => post<any>(`/datasets/${id}/decompose`, body),

  markov: (id: string, body: unknown) => post<any>(`/datasets/${id}/markov`, body),
  hmmFit: (id: string, body: unknown) => post<any>(`/datasets/${id}/hmm`, body),
  hmmDecode: (body: unknown) => post<any>('/hmm/decode', body),
  lds: (id: string, body: unknown) => post<any>(`/datasets/${id}/lds`, body),

  experiments: (params: { dataset_id?: string; unit?: number } = {}) => {
    const q = new URLSearchParams()
    if (params.dataset_id) q.set('dataset_id', params.dataset_id)
    if (params.unit) q.set('unit', String(params.unit))
    return request<Experiment[]>(`/experiments?${q}`)
  },
  experiment: (id: string) => request<Experiment>(`/experiments/${id}`),
  deleteExperiment: (id: string) => request<void>(`/experiments/${id}`, { method: 'DELETE' }),
  predict: (id: string, records: Record<string, unknown>[]) =>
    post<{ predictions: any[]; task: string }>(`/experiments/${id}/predict`, { records }),
  predictions: (id: string) => request<any[]>(`/experiments/${id}/predictions`),

  studyStatus: (study: string) => request<any>(`/study/${study}`),
  studyRun: (study: string, extended: boolean) => post<any>(`/study/${study}/run`, { extended }),
  studyImage: (study: string, kind: 'figures' | 'code', file: string, version = '') =>
    `${BASE}/study/${study}/${kind}/${file}${version ? `?v=${encodeURIComponent(version)}` : ''}`,
  socDetect: () => post<any>('/soc/detect', {}),
}

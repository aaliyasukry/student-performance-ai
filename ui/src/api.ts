import type {
  ActiveModelsResponse,
  CohortOptionsResponse,
  CohortResponse,
  HealthResponse,
  OverallReliability,
  PredictionResponse,
  PredictionSummary,
  SubjectReliabilityResponse,
} from './types'

const API_BASE = '/ai-api'

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
      ...(options?.headers ?? {}),
    },
    ...options,
  })

  let body: unknown = null
  try {
    body = await response.json()
  } catch {
    body = null
  }

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`
    if (body && typeof body === 'object' && 'detail' in body) {
      const detail = (body as { detail?: unknown }).detail
      if (typeof detail === 'string') message = detail
      else if (detail) message = JSON.stringify(detail)
    }
    throw new Error(message)
  }

  return body as T
}

export const getHealth = () => request<HealthResponse>('/health')

export const getActiveModels = () =>
  request<ActiveModelsResponse>('/api/v1/models/active')

export const getPredictionSummary = () =>
  request<PredictionSummary>('/api/v1/predictions/summary')

export const getOverallReliability = () =>
  request<OverallReliability>('/api/v1/models/overall-reliability')

export const getSubjectReliability = () =>
  request<SubjectReliabilityResponse>('/api/v1/models/subject-reliability')

export const getIntegrationReadiness = () =>
  request<Record<string, unknown>>('/api/v1/integration/readiness')

export const getSourceContract = () =>
  request<Record<string, unknown>>('/api/v1/integration/source-contract')

export const getCohortOptions = (targetYear: number) =>
  request<CohortOptionsResponse>(
    `/api/v1/lookups/cohort-options?target_year=${encodeURIComponent(targetYear)}`,
  )

export const predictStudent = (payload: {
  student_id: number
  target_year: number
  target_term: number
  include_actual_if_available: boolean
}) =>
  request<PredictionResponse>('/api/v1/predictions/student', {
    method: 'POST',
    body: JSON.stringify({
      ...payload,
      persist: true,
      source: 'STANDALONE_UI',
      requested_by_user_id: null,
    }),
  })

export const predictCohort = (payload: {
  target_year: number
  target_term: number
  grade_id: number
  class_id: number | null
  persist: boolean
  include_actual_if_available: boolean
}) =>
  request<CohortResponse>('/api/v1/predictions/cohort', {
    method: 'POST',
    body: JSON.stringify({
      ...payload,
      source: 'BATCH',
      requested_by_user_id: null,
    }),
  })

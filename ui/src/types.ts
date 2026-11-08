export interface HealthResponse {
  status: string
  service: string
  version: string
  database: {
    connected: boolean
    message?: string
  }
  models: Record<string, unknown>
}

export interface ModelRecord {
  id: number
  model_key: string
  version: string
  task_type: string
  algorithm: string
  feature_set: string
  final_test_rows: number
  final_test_mae: number
  final_test_rmse: number
  final_test_r2: number
  final_test_within_10_pct: number
  is_active: number | boolean
  activated_at?: string | null
}

export interface ReliabilityRecord {
  subject: string
  row_count: number
  mae: number | null
  rmse: number | null
  r2: number | null
  within_10_pct: number | null
  reliability: string
  display_policy: string
}

export interface SubjectReliabilityResponse {
  items: ReliabilityRecord[]
}

export interface OverallReliability {
  model_key?: string
  version?: string
  row_count?: number
  mae?: number
  rmse?: number
  r2?: number
  within_10_pct?: number
  reliability?: string
  display_policy?: string
}

export interface ActiveModelsResponse {
  database_models: ModelRecord[]
  local_artifacts: Record<string, unknown>
}

export interface PredictionRunSummary {
  id: number
  run_uuid: string
  source: string
  status: string
  target_period: string
  total_students: number
  total_predictions: number
  requested_by_user_id: number | null
  started_at: string | null
  completed_at: string | null
}

export interface PredictionSummary {
  total_runs: number
  completed_runs: number
  total_predictions: number
  distinct_students_predicted: number
  latest_completed_at: string | null
  overall_risk_distribution: Record<string, number>
  recent_runs: PredictionRunSummary[]
}

export interface AcademicHistoryPeriod {
  year: number
  term: number
  academic_period: string
  grade: number
  class: string
  average: number | null
  numeric_subject_count: number
}

export interface AttendancePeriod {
  year: number
  term: number
  academic_period: string
  term_name: string
  class: string
  grade: string
  total_records: number
  attended_count: number
  present_count: number
  late_count: number
  absent_count: number
  excused_count: number
  attendance_pct: number | null
  attendance_level: string
}

export interface AttendanceContext {
  available: boolean
  used_by_final_predictor: boolean
  calculation_note: string
  historical_periods: AttendancePeriod[]
  historical_average_pct: number | null
  latest_historical_period: AttendancePeriod | null
  historical_trend: string
  target_period: AttendancePeriod | null
  display_period: AttendancePeriod | null
}

export interface SubjectHistoryPeriod {
  year: number
  term: number
  academic_period: string
  mark: number | null
  status: string
}

export interface PredictionSubject {
  prediction_id: number | null
  subject_key: string
  target_subject: string
  predicted_mark: number
  previous_mark: number | null
  expected_change: number | null
  subject_standing: string
  history: SubjectHistoryPeriod[]
  actual_mark: number | null
  error: number | null
  absolute_error: number | null
  academic_risk: string
  prediction_reliability: string
  display_policy: string
  performance_trend: string
  feature_count: number
}

export interface PredictionResponse {
  run_uuid: string
  persistence: {
    persisted: boolean
    prediction_run_id: number | null
  }
  student: {
    id: number
    full_name: string
    student_index: string | null
    admission_no: string | null
    status: string
  }
  target: {
    year: number
    term: number
    period: string
    term_name: string
    grade: string
    class: string
  }
  history: {
    period_count: number
    periods: Array<{
      year: number
      term: number
      academic_period: string
    }>
    latest_previous_average: number | null
    academic_history: AcademicHistoryPeriod[]
  }
  attendance_context: AttendanceContext
  model_generation: {
    label: string
    attendance_used_by_predictor: boolean
    future_note: string
  }
  models: {
    overall: { id: number; model_key: string; version: string }
    subject: { id: number; model_key: string; version: string }
  }
  overall: {
    prediction_id: number | null
    model_version_id: number
    model_version: string
    predicted_average: number
    actual_average: number | null
    error: number | null
    absolute_error: number | null
    academic_risk: string
    prediction_reliability: string
    display_policy: string
    performance_trend: string
    feature_count: number
  }
  subjects: PredictionSubject[]
  validation: {
    actual_target_data_available: boolean
    actual_subject_marks_available: number
    actual_values_were_not_used_as_features: boolean
  }
}

export interface CohortClassOption {
  class_id: number
  class_name: string
  class_code: string
  student_count: number
}

export interface CohortGradeOption {
  grade_id: number
  grade_name: string
  grade_code: string
  student_count: number
  classes: CohortClassOption[]
}

export interface CohortOptionsResponse {
  target_year: number
  grades: CohortGradeOption[]
}

export interface CohortStudentRow {
  student_id: number
  full_name: string
  student_index: string | null
  admission_no: string | null
  grade: string
  class: string
  predicted_average: number
  actual_average: number | null
  academic_risk: string
  prediction_reliability: string
  performance_trend: string
  attendance_pct: number | null
  attendance_level: string
  attendance_period: string | null
  prediction_run_id: number | null
}

export interface CohortSubjectSummary {
  subject_key: string
  target_subject: string
  student_count: number
  predicted_average: number
  high_risk_students: number
  attention_students: number
  on_track_students: number
  high_performer_students: number
  prediction_reliability: string
  display_policy: string
}

export interface CohortResponse {
  target: {
    year: number
    term: number
    period: string
    term_name: string
    grade: string
    class: string
  }
  filters: {
    grade_id: number
    class_id: number | null
  }
  summary: {
    students_found: number
    students_predicted: number
    students_failed: number
    predicted_class_average: number
    highest_predicted_average: number
    lowest_predicted_average: number
    risk_distribution: Record<string, number>
    attendance_average_pct: number | null
    attendance_distribution: Record<string, number>
    low_attendance_students: number
    high_risk_low_attendance_students: number
  }
  students: CohortStudentRow[]
  subject_summary: CohortSubjectSummary[]
  failures: Array<{
    student_id: number
    full_name: string
    class: string
    reason: string
  }>
  persistence: {
    requested: boolean
    mode: string
  }
  attendance_note: string
}

import {
  Activity,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  Database,
  Gauge,
  GraduationCap,
  LayoutDashboard,
  RefreshCw,
  Server,
  ShieldCheck,
  TriangleAlert,
  Users,
  X,
  XCircle,
} from 'lucide-react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { useEffect, useMemo, useState, type FormEvent } from 'react'
import {
  getActiveModels,
  getCohortOptions,
  getHealth,
  getIntegrationReadiness,
  getOverallReliability,
  getPredictionSummary,
  getSourceContract,
  getSubjectReliability,
  predictCohort,
  predictStudent,
} from './api'
import type {
  ActiveModelsResponse,
  CohortOptionsResponse,
  CohortResponse,
  HealthResponse,
  ModelRecord,
  OverallReliability,
  PredictionResponse,
  PredictionSubject,
  PredictionSummary,
  ReliabilityRecord,
} from './types'

type Page =
  | 'dashboard'
  | 'models'
  | 'student'
  | 'cohort'
  | 'reliability'
  | 'integration'

const navItems = [
  { id: 'dashboard' as Page, label: 'Dashboard', icon: LayoutDashboard },
  { id: 'models' as Page, label: 'Models', icon: BrainCircuit },
  { id: 'student' as Page, label: 'Student Prediction', icon: GraduationCap },
  { id: 'cohort' as Page, label: 'Cohort Forecast', icon: Users },
  { id: 'reliability' as Page, label: 'Reliability & Validation', icon: Gauge },
  { id: 'integration' as Page, label: 'Integration Status', icon: Server },
]

function number(value: number | null | undefined, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—'
  return Number(value).toFixed(digits)
}

function pretty(value?: string | null) {
  if (!value) return '—'
  const labels: Record<string, string> = {
    ATTENTION: 'Needs Attention',
    NEEDS_ATTENTION: 'Needs Attention',
    HIGH_RISK: 'High Risk',
    ON_TRACK: 'On Track',
    HIGH_PERFORMER: 'High Performer',
    INSUFFICIENT_DATA: 'Insufficient Data',
    SHOW_WITH_CAUTION: 'Show With Caution',
    SUPPRESS_BY_DEFAULT: 'Suppress By Default',
    NO_DATA: 'No Data',
  }
  if (labels[value]) return labels[value]
  return value
    .replaceAll('_', ' ')
    .toLowerCase()
    .replace(/\b\w/g, (m) => m.toUpperCase())
}

function riskClass(value?: string | null) {
  if (value === 'HIGH_RISK') return 'badge red'
  if (value === 'ATTENTION' || value === 'NEEDS_ATTENTION') return 'badge amber'
  if (value === 'HIGH_PERFORMER') return 'badge purple'
  return 'badge green'
}

function reliabilityClass(value?: string | null) {
  if (value === 'HIGH') return 'badge green'
  if (value === 'MODERATE') return 'badge blue'
  if (value === 'LOW') return 'badge amber'
  return 'badge gray'
}

function standingClass(value?: string | null) {
  if (value === 'STRONG') return 'badge green'
  if (value === 'WEAK') return 'badge red'
  return 'badge blue'
}


function StatCard({ label, value, helper }: { label: string; value: string; helper?: string }) {
  return (
    <div className="stat-card">
      <span>{label}</span>
      <strong>{value}</strong>
      {helper && <small>{helper}</small>}
    </div>
  )
}

function ModelCard({ model }: { model?: ModelRecord }) {
  if (!model) return <div className="panel"><div className="empty-block">Model metadata not available.</div></div>
  return (
    <section className="panel model-card">
      <div className="panel-title">
        <div>
          <span className="eyebrow">{pretty(model.task_type)} Predictor</span>
          <h3>{model.algorithm}</h3>
          <p>{model.model_key}</p>
        </div>
        <span className="badge green">ACTIVE · v{model.version}</span>
      </div>
      <div className="model-meta-grid">
        <div><span>Feature Set</span><strong>{pretty(model.feature_set)}</strong></div>
        <div><span>Final Test Rows</span><strong>{model.final_test_rows}</strong></div>
        <div><span>MAE</span><strong>{number(model.final_test_mae, 4)}</strong></div>
        <div><span>RMSE</span><strong>{number(model.final_test_rmse, 4)}</strong></div>
        <div><span>R²</span><strong>{number(model.final_test_r2, 4)}</strong></div>
        <div><span>Within ±10</span><strong>{number(model.final_test_within_10_pct, 2)}%</strong></div>
      </div>
    </section>
  )
}

function App() {
  const [page, setPage] = useState<Page>('dashboard')
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [models, setModels] = useState<ActiveModelsResponse | null>(null)
  const [overallReliability, setOverallReliability] = useState<OverallReliability | null>(null)
  const [subjectReliability, setSubjectReliability] = useState<ReliabilityRecord[]>([])
  const [, setSummary] = useState<PredictionSummary | null>(null)
  const [readiness, setReadiness] = useState<Record<string, unknown> | null>(null)
  const [sourceContract, setSourceContract] = useState<Record<string, unknown> | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const loadSystem = async () => {
    setLoading(true)
    setError(null)
    try {
      const [healthData, modelData, overallData, subjectData, summaryData, readinessData, contractData] =
        await Promise.all([
          getHealth(),
          getActiveModels(),
          getOverallReliability(),
          getSubjectReliability(),
          getPredictionSummary(),
          getIntegrationReadiness(),
          getSourceContract(),
        ])
      setHealth(healthData)
      setModels(modelData)
      setOverallReliability(overallData)
      setSubjectReliability(subjectData.items)
      setSummary(summaryData)
      setReadiness(readinessData)
      setSourceContract(contractData)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load AI service data.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { void loadSystem() }, [])

  const overallModel = useMemo(() => models?.database_models.find((m) =>
    m.model_key.toLowerCase().includes('overall')), [models])
  const subjectModel = useMemo(() => models?.database_models.find((m) =>
    m.model_key.toLowerCase().includes('subject')), [models])

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><BrainCircuit size={25} /></div>
          <div><strong>Student AI</strong><span>Performance Forecasting</span></div>
        </div>
        <nav>
          {navItems.map((item) => {
            const Icon = item.icon
            return (
              <button key={item.id} className={page === item.id ? 'nav-item active' : 'nav-item'} onClick={() => setPage(item.id)}>
                <Icon size={18} /><span>{item.label}</span>
              </button>
            )
          })}
        </nav>
        <div className="sidebar-footer">
          <div><span>API</span><b>{health?.version ?? '—'}</b></div>
          <div><span>Model</span><b>v{overallModel?.version ?? '—'}</b></div>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <h1>{navItems.find((item) => item.id === page)?.label}</h1>
            <p>AI-powered student performance forecasting and academic decision support</p>
          </div>
          <div className="top-actions">
            <div className={health?.status === 'ok' ? 'health healthy' : 'health unhealthy'}>
              <span />{health?.status === 'ok' ? 'System Healthy' : 'System Degraded'}
            </div>
            <button className="icon-btn" onClick={() => void loadSystem()} title="Refresh"><RefreshCw size={17} /></button>
          </div>
        </header>

        <div className="content">
          {loading && <div className="message"><RefreshCw className="spin" size={19} /> Loading live data...</div>}
          {error && <div className="message error"><TriangleAlert size={19} /><div><b>Unable to load service</b><span>{error}</span></div></div>}

          {!loading && page === 'dashboard' && (
            <Dashboard
              health={health}
              overallModel={overallModel}
              subjectModel={subjectModel}
              overallReliability={overallReliability}
              subjectReliability={subjectReliability}
            />
          )}
          {!loading && page === 'models' && (
            <ModelsPage overallModel={overallModel} subjectModel={subjectModel} overallReliability={overallReliability} />
          )}
          {!loading && page === 'student' && <StudentPrediction />}
          {!loading && page === 'cohort' && <CohortForecast />}
          {!loading && page === 'reliability' && <ReliabilityPage data={subjectReliability} />}
          {!loading && page === 'integration' && (
            <Integration health={health} readiness={readiness} sourceContract={sourceContract} />
          )}
        </div>
      </main>
    </div>
  )
}

function Dashboard({
  health,
  overallModel,
  subjectModel,
  overallReliability,
  subjectReliability,
}: {
  health: HealthResponse | null
  overallModel?: ModelRecord
  subjectModel?: ModelRecord
  overallReliability: OverallReliability | null
  subjectReliability: ReliabilityRecord[]
}) {
  const reliabilityCounts = useMemo(() => {
    const result: Record<string, number> = { HIGH: 0, MODERATE: 0, LOW: 0, INSUFFICIENT_DATA: 0 }
    subjectReliability.forEach((item) => {
      const key = item.reliability?.toUpperCase() ?? 'INSUFFICIENT_DATA'
      result[key] = (result[key] ?? 0) + 1
    })
    return result
  }, [subjectReliability])

  return (
    <>
      <section className="hero">
        <div>
          <span className="eyebrow">Production Model Monitor</span>
          <h2>Student Performance AI</h2>
          <p>Live monitoring of the deployed forecasting models, validation evidence and school-system integration.</p>
        </div>
        <BrainCircuit size={70} />
      </section>

      <div className="status-grid">
        <div className="status-card"><Activity /><div><span>API Service</span><strong>{health?.status === 'ok' ? 'Healthy' : 'Degraded'}</strong></div></div>
        <div className="status-card"><Database /><div><span>Database</span><strong>{health?.database.connected ? 'Connected' : 'Disconnected'}</strong></div></div>
        <div className="status-card"><BrainCircuit /><div><span>Overall Model</span><strong>Active v{overallModel?.version ?? '—'}</strong></div></div>
        <div className="status-card"><Gauge /><div><span>Subject Model</span><strong>Active v{subjectModel?.version ?? '—'}</strong></div></div>
      </div>

      <div className="section-title"><div><h2>Final Model Performance</h2><p>Protected out-of-time evaluation evidence</p></div></div>
      <div className="two-col">
        <section className="panel">
          <div className="panel-title"><div><span className="eyebrow">Overall Predictor</span><h3>Overall Average</h3></div><span className={reliabilityClass(overallReliability?.reliability)}>{overallReliability?.reliability ?? 'HIGH'}</span></div>
          <div className="metrics-grid">
            <StatCard label="MAE" value={number(overallModel?.final_test_mae, 4)} />
            <StatCard label="RMSE" value={number(overallModel?.final_test_rmse, 4)} />
            <StatCard label="R²" value={number(overallModel?.final_test_r2, 4)} />
            <StatCard label="Within ±10" value={`${number(overallModel?.final_test_within_10_pct, 2)}%`} />
          </div>
        </section>
        <section className="panel">
          <div className="panel-title"><div><span className="eyebrow">Subject Predictor</span><h3>Subject Marks</h3></div><span className="badge blue">17 Subjects</span></div>
          <div className="metrics-grid">
            <StatCard label="MAE" value={number(subjectModel?.final_test_mae, 4)} />
            <StatCard label="RMSE" value={number(subjectModel?.final_test_rmse, 4)} />
            <StatCard label="R²" value={number(subjectModel?.final_test_r2, 4)} />
            <StatCard label="Within ±10" value={`${number(subjectModel?.final_test_within_10_pct, 2)}%`} />
          </div>
        </section>
      </div>

      <section className="panel">
        <div className="panel-title"><div><span className="eyebrow">Decision Support</span><h3>Subject Reliability Overview</h3></div></div>
        <div className="reliability-summary">
          <div><strong>{reliabilityCounts.HIGH}</strong><span>High</span></div>
          <div><strong>{reliabilityCounts.MODERATE}</strong><span>Moderate</span></div>
          <div><strong>{reliabilityCounts.LOW}</strong><span>Low</span></div>
          <div><strong>{reliabilityCounts.INSUFFICIENT_DATA}</strong><span>Insufficient</span></div>
        </div>
      </section>
    </>
  )
}

function ModelsPage({ overallModel, subjectModel, overallReliability }: {
  overallModel?: ModelRecord
  subjectModel?: ModelRecord
  overallReliability: OverallReliability | null
}) {
  return (
    <>
      <div className="section-title"><div><h2>Deployed Models</h2><p>Frozen production models currently used by the prediction service.</p></div></div>
      <div className="models-stack">
        <ModelCard model={overallModel} />
        <ModelCard model={subjectModel} />
      </div>
      <div className="info-panel">
        <ShieldCheck size={22} />
        <div>
          <strong>Overall prediction reliability</strong>
          <p>Current reliability: <b>{overallReliability?.reliability ?? 'HIGH'}</b>. Reliability represents validation evidence and is separate from student academic risk.</p>
        </div>
      </div>
    </>
  )
}

function ReliabilityPage({ data }: { data: ReliabilityRecord[] }) {
  const chartData = [...data]
    .filter((item) => item.mae !== null && Number.isFinite(Number(item.mae)))
    .sort((a, b) => Number(a.mae) - Number(b.mae))
    .map((item) => ({ subject: pretty(item.subject), mae: Number(item.mae) }))

  return (
    <>
      <div className="section-title"><div><h2>Reliability & Validation</h2><p>Subject-level held-out evaluation evidence from the final test period.</p></div></div>
      <section className="panel reliability-chart-panel">
        <div className="panel-title"><div><span className="eyebrow">Error Analysis</span><h3>MAE by Subject</h3></div></div>
        <div className="chart reliability-chart">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} layout="vertical" margin={{ top: 10, right: 25, bottom: 10, left: 80 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis dataKey="subject" type="category" width={120} tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="mae" fill="#5667d8" radius={[0, 5, 5, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>
      <section className="panel">
        <div className="panel-title"><div><span className="eyebrow">Validation Evidence</span><h3>Subject Reliability Matrix</h3></div></div>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Subject</th><th>Rows</th><th>MAE</th><th>RMSE</th><th>R²</th><th>Within ±10</th><th>Reliability</th><th>Display Policy</th></tr></thead>
            <tbody>{data.map((item) => (
              <tr key={item.subject}>
                <td><strong>{pretty(item.subject)}</strong></td>
                <td>{item.row_count}</td>
                <td>{number(item.mae, 4)}</td>
                <td>{number(item.rmse, 4)}</td>
                <td>{number(item.r2, 4)}</td>
                <td>{number(item.within_10_pct, 2)}%</td>
                <td><span className={reliabilityClass(item.reliability)}>{item.reliability}</span></td>
                <td>{pretty(item.display_policy)}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </section>
    </>
  )
}

function StudentPrediction() {
  const [studentId, setStudentId] = useState('1025')
  const [year, setYear] = useState('2026')
  const [term, setTerm] = useState('2')
  const [includeActual, setIncludeActual] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [selectedSubject, setSelectedSubject] = useState<PredictionSubject | null>(null)

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setLoading(true); setError(null); setResult(null); setSelectedSubject(null)
    try {
      setResult(await predictStudent({
        student_id: Number(studentId),
        target_year: Number(year),
        target_term: Number(term),
        include_actual_if_available: includeActual,
      }))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <div className="section-title"><div><h2>Individual Student Forecast</h2><p>Generate an overall and subject-wise forecast from historical School Portal records.</p></div></div>
      <form className="panel form-panel" onSubmit={submit}>
        <div className="form-grid">
          <label><span>Student ID</span><input type="number" min="1" value={studentId} onChange={(e) => setStudentId(e.target.value)} required /></label>
          <label><span>Target Year</span><input type="number" min="2025" value={year} onChange={(e) => setYear(e.target.value)} required /></label>
          <label><span>Target Term</span><select value={term} onChange={(e) => setTerm(e.target.value)}><option value="1">Term 1</option><option value="2">Term 2</option><option value="3">Term 3</option></select></label>
        </div>
        <div className="form-actions">
          <label className="check"><input type="checkbox" checked={includeActual} onChange={(e) => setIncludeActual(e.target.checked)} /><span>Include actual results if available</span></label>
          <button className="primary" disabled={loading}>{loading ? <RefreshCw className="spin" size={17} /> : <GraduationCap size={17} />}{loading ? 'Generating...' : 'Generate Prediction'}</button>
        </div>
      </form>
      {error && <div className="message error"><TriangleAlert size={18} /><span>{error}</span></div>}
      {result && <StudentResult result={result} onSubject={setSelectedSubject} />}
      {selectedSubject && <SubjectDrawer subject={selectedSubject} overall={result?.overall.predicted_average ?? 0} onClose={() => setSelectedSubject(null)} />}
    </>
  )
}

function StudentResult({ result, onSubject }: { result: PredictionResponse; onSubject: (subject: PredictionSubject) => void }) {
  const academicData = result.history.academic_history.map((p) => ({ period: p.academic_period, average: p.average }))
  academicData.push({ period: `${result.target.period} (Pred)`, average: result.overall.predicted_average })

  return (
    <div className="results-stack">
      <section className="panel student-header">
        <div><span className="eyebrow">Student</span><h2>{result.student.full_name}</h2><p>ID {result.student.id} · Index {result.student.student_index ?? '—'} · Admission {result.student.admission_no ?? '—'}</p></div>
        <div className="target-box"><span>Forecast Target</span><b>{result.target.year} · {result.target.term_name}</b><small>{result.target.grade} · {result.target.class}</small></div>
      </section>

      <section className="forecast-banner">
        <div className="big-score"><span>Predicted Average</span><strong>{number(result.overall.predicted_average, 1)}</strong><small>Previous: {number(result.history.latest_previous_average, 1)}</small></div>
        <div className="forecast-cards">
          <div><span>Predicted Risk</span><b className={riskClass(result.overall.academic_risk)}>{pretty(result.overall.academic_risk)}</b></div>
          <div><span>Reliability</span><b className={reliabilityClass(result.overall.prediction_reliability)}>{pretty(result.overall.prediction_reliability)}</b></div>
          <div><span>Academic Trend</span><b>{pretty(result.overall.performance_trend)}</b></div>
          <div><span>History Used</span><b>{result.history.period_count} terms</b></div>
          {result.overall.actual_average !== null && <div><span>Actual Average</span><b>{number(result.overall.actual_average, 1)}</b></div>}
          {result.overall.absolute_error !== null && <div><span>Absolute Error</span><b>{number(result.overall.absolute_error, 1)}</b></div>}
        </div>
      </section>

      <section className="panel">
        <div className="panel-title"><div><span className="eyebrow">Academic history</span><h3>Overall Performance Trend</h3></div></div>
        <div className="chart medium"><ResponsiveContainer width="100%" height="100%"><LineChart data={academicData}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="period" tick={{ fontSize: 10 }} /><YAxis domain={[0, 100]} /><Tooltip /><Line type="monotone" dataKey="average" stroke="#5968d8" strokeWidth={3} connectNulls /></LineChart></ResponsiveContainer></div>
      </section>

      <section className="panel">
        <div className="panel-title"><div><span className="eyebrow">Click a subject</span><h3>Subject Forecasts</h3><p>Open any card for history, strength/weakness, risk and validation details.</p></div></div>
        <div className="subject-grid">
          {result.subjects.map((subject) => (
            <button key={subject.subject_key} className="subject-card" onClick={() => onSubject(subject)}>
              <div className="subject-top"><b>{pretty(subject.target_subject || subject.subject_key)}</b><ChevronRight size={17} /></div>
              <div className="subject-score">{number(subject.predicted_mark, 1)}</div>
              <div className="subject-badges"><span className={standingClass(subject.subject_standing)}>{pretty(subject.subject_standing)}</span><span className={riskClass(subject.academic_risk)}>{pretty(subject.academic_risk)}</span></div>
              <div className="subject-foot"><span>Prev {number(subject.previous_mark, 1)}</span><span>{pretty(subject.performance_trend)}</span></div>
            </button>
          ))}
        </div>
      </section>

      <section className="panel meta-row">
        <div><span>Model Generation</span><b>{result.model_generation.label}</b></div>
        <div><span>Prediction Run</span><b>#{result.persistence.prediction_run_id ?? '—'}</b></div>
        <div><span>Overall Model</span><b>v{result.models.overall.version}</b></div>
        <div><span>Subject Model</span><b>v{result.models.subject.version}</b></div>
      </section>
    </div>
  )
}

function SubjectDrawer({ subject, overall, onClose }: { subject: PredictionSubject; overall: number; onClose: () => void }) {
  const history = subject.history.map((h) => ({ period: h.academic_period, mark: h.mark }))
  history.push({ period: 'Predicted', mark: subject.predicted_mark })
  if (subject.actual_mark !== null) history.push({ period: 'Actual', mark: subject.actual_mark })

  const message = subject.subject_standing === 'STRONG'
    ? 'This is one of the student’s stronger subjects relative to the predicted overall average.'
    : subject.subject_standing === 'WEAK'
      ? 'This is one of the student’s weaker subjects relative to the predicted overall average and may need attention.'
      : 'This subject is broadly consistent with the student’s predicted overall performance.'

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside className="drawer" onClick={(e) => e.stopPropagation()}>
        <div className="drawer-head"><div><span className="eyebrow">Subject analysis</span><h2>{pretty(subject.target_subject || subject.subject_key)}</h2></div><button className="icon-btn" onClick={onClose}><X size={18} /></button></div>
        <div className="drawer-score"><span>Predicted Mark</span><strong>{number(subject.predicted_mark, 1)}</strong><b className={standingClass(subject.subject_standing)}>{pretty(subject.subject_standing)} Subject</b></div>
        <div className="drawer-grid">
          <StatCard label="Previous Mark" value={number(subject.previous_mark, 1)} />
          <StatCard label="Expected Change" value={subject.expected_change === null ? '—' : `${subject.expected_change >= 0 ? '+' : ''}${number(subject.expected_change, 1)}`} />
          <StatCard label="Predicted Risk" value={pretty(subject.academic_risk)} />
          <StatCard label="Reliability" value={pretty(subject.prediction_reliability)} />
          <StatCard label="Trend" value={pretty(subject.performance_trend)} />
          <StatCard label="Actual Mark" value={number(subject.actual_mark, 1)} />
        </div>
        <div className="insight"><b>{pretty(subject.subject_standing)} subject</b><p>{message} Predicted subject performance is compared with the student’s predicted overall average of {number(overall, 1)}.</p></div>
        <div className="panel-title"><div><span className="eyebrow">Term history</span><h3>Subject Performance</h3></div></div>
        <div className="chart medium"><ResponsiveContainer width="100%" height="100%"><LineChart data={history}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="period" tick={{ fontSize: 10 }} /><YAxis domain={[0, 100]} /><Tooltip /><Line type="monotone" dataKey="mark" stroke="#5968d8" strokeWidth={3} connectNulls /></LineChart></ResponsiveContainer></div>
        {subject.display_policy !== 'SHOW' && <div className="message warning"><TriangleAlert size={18} /><span>Display policy: {pretty(subject.display_policy)}. Interpret this subject forecast with the corresponding reliability caution.</span></div>}
      </aside>
    </div>
  )
}

function CohortForecast() {
  const [year, setYear] = useState('2026')
  const [term, setTerm] = useState('2')
  const [options, setOptions] = useState<CohortOptionsResponse | null>(null)
  const [gradeId, setGradeId] = useState('')
  const [classId, setClassId] = useState('')
  const [includeActual, setIncludeActual] = useState(true)
  const [persist, setPersist] = useState(false)
  const [loadingOptions, setLoadingOptions] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<CohortResponse | null>(null)

  useEffect(() => {
    const load = async () => {
      setLoadingOptions(true)
      try {
        const data = await getCohortOptions(Number(year))
        setOptions(data)
        const first = data.grades[0]
        setGradeId(first ? String(first.grade_id) : '')
        setClassId('')
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Could not load cohort options.')
      } finally {
        setLoadingOptions(false)
      }
    }
    void load()
  }, [year])

  const selectedGrade = options?.grades.find((g) => String(g.grade_id) === gradeId)

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    if (!gradeId) return
    setLoading(true); setError(null); setResult(null)
    try {
      setResult(await predictCohort({
        target_year: Number(year),
        target_term: Number(term),
        grade_id: Number(gradeId),
        class_id: classId ? Number(classId) : null,
        persist,
        include_actual_if_available: includeActual,
      }))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Cohort prediction failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <div className="section-title"><div><h2>Class / Grade Forecast</h2><p>Run the same V1 prediction pipeline across every eligible student in a selected cohort.</p></div></div>
      <form className="panel form-panel" onSubmit={submit}>
        <div className="form-grid four-fields">
          <label><span>Target Year</span><input type="number" min="2025" value={year} onChange={(e) => setYear(e.target.value)} /></label>
          <label><span>Target Term</span><select value={term} onChange={(e) => setTerm(e.target.value)}><option value="1">Term 1</option><option value="2">Term 2</option><option value="3">Term 3</option></select></label>
          <label><span>Grade</span><select value={gradeId} onChange={(e) => { setGradeId(e.target.value); setClassId('') }} disabled={loadingOptions}><option value="">Select grade</option>{options?.grades.map((g) => <option key={g.grade_id} value={g.grade_id}>{g.grade_name} ({g.student_count})</option>)}</select></label>
          <label><span>Class</span><select value={classId} onChange={(e) => setClassId(e.target.value)} disabled={!selectedGrade}><option value="">All classes</option>{selectedGrade?.classes.map((c) => <option key={c.class_id} value={c.class_id}>{c.class_name} ({c.student_count})</option>)}</select></label>
        </div>
        <div className="form-actions">
          <div className="checks"><label className="check"><input type="checkbox" checked={includeActual} onChange={(e) => setIncludeActual(e.target.checked)} /><span>Include actuals if available</span></label><label className="check"><input type="checkbox" checked={persist} onChange={(e) => setPersist(e.target.checked)} /><span>Persist each student forecast</span></label></div>
          <button className="primary" disabled={loading || !gradeId}>{loading ? <RefreshCw className="spin" size={17} /> : <Users size={17} />}{loading ? 'Running cohort...' : 'Run Cohort Forecast'}</button>
        </div>
      </form>
      {error && <div className="message error"><TriangleAlert size={18} /><span>{error}</span></div>}
      {result && <CohortResult result={result} />}
    </>
  )
}

function CohortResult({ result }: { result: CohortResponse }) {
  const riskData = [
    { name: 'High Risk', count: result.summary.risk_distribution.HIGH_RISK ?? 0 },
    { name: 'Needs Attention', count: result.summary.risk_distribution.ATTENTION ?? 0 },
    { name: 'On Track', count: result.summary.risk_distribution.ON_TRACK ?? 0 },
    { name: 'High Performer', count: result.summary.risk_distribution.HIGH_PERFORMER ?? 0 },
  ]

  return (
    <div className="results-stack">
      <section className="panel student-header"><div><span className="eyebrow">Cohort</span><h2>{result.target.grade} · {result.target.class}</h2><p>Forecast target: {result.target.year} {result.target.term_name}</p></div><div className="target-box"><span>Students Predicted</span><b>{result.summary.students_predicted}</b><small>{result.summary.students_failed} unable to predict</small></div></section>

      <div className="stats-grid">
        <StatCard label="Predicted Class Avg" value={number(result.summary.predicted_class_average, 1)} />
        <StatCard label="High Risk" value={String(result.summary.risk_distribution.HIGH_RISK ?? 0)} />
        <StatCard label="Needs Attention" value={String(result.summary.risk_distribution.ATTENTION ?? 0)} />
        <StatCard label="On Track" value={String(result.summary.risk_distribution.ON_TRACK ?? 0)} />
        <StatCard label="High Performer" value={String(result.summary.risk_distribution.HIGH_PERFORMER ?? 0)} />
      </div>

      <section className="panel"><div className="panel-title"><div><span className="eyebrow">Academic outlook</span><h3>Risk Distribution</h3></div></div><div className="chart medium"><ResponsiveContainer width="100%" height="100%"><BarChart data={riskData}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="name" tick={{ fontSize: 10 }} /><YAxis allowDecimals={false} /><Tooltip /><Bar dataKey="count" fill="#5968d8" radius={[6, 6, 0, 0]} /></BarChart></ResponsiveContainer></div></section>

      <section className="panel">
        <div className="panel-title"><div><span className="eyebrow">Class subject view</span><h3>Subject Forecast Summary</h3><p>Average predicted mark and number of high-risk students by subject.</p></div></div>
        <div className="subject-summary-grid">
          {result.subject_summary.map((s) => (
            <div className="cohort-subject" key={s.subject_key}>
              <div><b>{pretty(s.target_subject || s.subject_key)}</b><span className={reliabilityClass(s.prediction_reliability)}>{pretty(s.prediction_reliability)}</span></div>
              <strong>{number(s.predicted_average, 1)}</strong>
              <small>{s.high_risk_students} high risk · {s.attention_students} attention</small>
            </div>
          ))}
        </div>
      </section>

      <section className="panel">
        <div className="panel-title"><div><span className="eyebrow">Student list</span><h3>Predicted Student Outcomes</h3><p>Sorted from lowest to highest predicted overall average.</p></div></div>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Student</th><th>Class</th><th>Pred. Avg</th><th>Actual</th><th>Predicted Risk</th><th>Trend</th></tr></thead>
            <tbody>{result.students.map((s) => (
              <tr key={s.student_id} className={s.academic_risk === 'HIGH_RISK' ? 'risk-row' : ''}>
                <td><b>{s.full_name}</b><small className="subtext">ID {s.student_id}</small></td><td>{s.class}</td><td><b>{number(s.predicted_average, 1)}</b></td><td>{number(s.actual_average, 1)}</td><td><span className={riskClass(s.academic_risk)}>{pretty(s.academic_risk)}</span></td><td>{pretty(s.performance_trend)}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </section>

      {result.failures.length > 0 && <details className="panel"><summary>{result.failures.length} students could not be predicted</summary><div className="failure-list">{result.failures.map((f) => <div key={f.student_id}><b>{f.full_name}</b><span>{f.reason}</span></div>)}</div></details>}
    </div>
  )
}

function Integration({ health, readiness, sourceContract }: { health: HealthResponse | null; readiness: Record<string, unknown> | null; sourceContract: Record<string, unknown> | null }) {
  const ok = health?.status === 'ok'
  return (
    <>
      <div className="section-title"><div><h2>Integration Status</h2><p>Runtime connection between the School Portal database and the deployed AI service.</p></div></div>
      <div className="status-grid">
        <div className="status-card"><Server /><div><span>FastAPI</span><strong className="inline">{ok ? <CheckCircle2 size={16} /> : <XCircle size={16} />}{ok ? 'Online' : 'Degraded'}</strong></div></div>
        <div className="status-card"><Database /><div><span>MySQL</span><strong className="inline">{health?.database.connected ? <CheckCircle2 size={16} /> : <XCircle size={16} />}{health?.database.connected ? 'Connected' : 'Disconnected'}</strong></div></div>
        <div className="status-card"><BrainCircuit /><div><span>Prediction Runtime</span><strong>{ok ? 'Ready' : 'Check Service'}</strong></div></div>
        <div className="status-card"><ShieldCheck /><div><span>V1 Feature Policy</span><strong>Academic history only</strong></div></div>
      </div>
      <div className="flow"><div>School Portal MySQL</div><span>→</span><div>AI Source Views</div><span>→</span><div>Historical Features</div><span>→</span><div>Random Forest V1</div><span>→</span><div>FastAPI</div></div>
      <div className="two-col">
        <section className="panel"><div className="panel-title"><div><span className="eyebrow">Live database</span><h3>Integration Readiness</h3></div></div><pre>{JSON.stringify(readiness, null, 2)}</pre></section>
        <section className="panel"><div className="panel-title"><div><span className="eyebrow">Data sources</span><h3>Source Contract</h3></div></div><pre>{JSON.stringify(sourceContract, null, 2)}</pre></section>
      </div>
    </>
  )
}


export default App
